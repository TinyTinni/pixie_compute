#include "pixie_compute/shader_compiler.hpp"

#include "pixie_compute/utility.hpp"

#include <iostream>
#include <mutex>
#include <slang.h>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pix
{
namespace
{

constexpr size_t kMaxCacheEntries = 128;

std::unordered_map<std::string, CompiledShader> spirvCache;
std::unordered_map<std::string, CompiledShader> moduleCache;
std::mutex cacheMutex;

void reflectBindings(slang::ProgramLayout *layout, std::vector<vk::DescriptorType> &outTypes)
{
    if (!layout)
        return;

    uint32_t maxBinding = 0;
    bool hasBindings = false;
    std::vector<std::pair<uint32_t, vk::DescriptorType>> bindings;

    auto scan = [&](slang::VariableLayoutReflection *param)
    {
        if (!param)
            return;
        auto *tl = param->getTypeLayout();
        if (!tl || tl->getBindingRangeCount() == 0)
            return;

        auto bindingType = tl->getBindingRangeType(0);
        if (bindingType == slang::BindingType::PushConstant)
            return; // handled separately

        uint32_t bindingIndex = param->getBindingIndex();
        uint32_t bindingSpace = param->getBindingSpace();

        if (bindingSpace != 0)
            throw GpuError("only descriptor set 0 is supported; found binding in set " +
                           std::to_string(bindingSpace));

        vk::DescriptorType vkType;
        switch (bindingType)
        {
        case slang::BindingType::RawBuffer:
        case slang::BindingType::MutableRawBuffer:
        case slang::BindingType::TypedBuffer:
        case slang::BindingType::MutableTypedBuffer:
            vkType = vk::DescriptorType::eStorageBuffer;
            break;
        case slang::BindingType::Texture:
            vkType = vk::DescriptorType::eSampledImage;
            break;
        case slang::BindingType::MutableTexture:
            vkType = vk::DescriptorType::eStorageImage;
            break;
        case slang::BindingType::CombinedTextureSampler:
            vkType = vk::DescriptorType::eCombinedImageSampler;
            break;
        case slang::BindingType::Sampler:
            throw GpuError("standalone samplers not supported; use combined image sampler");
        case slang::BindingType::ConstantBuffer:
            // Ordinary constant buffers are not represented by GpuBinding. They
            // remain valid Slang input for callers that only need SPIR-V output.
            return;
        default:
            throw GpuError("unsupported binding type in shader reflection");
        }

        bindings.push_back({bindingIndex, vkType});
        if (!hasBindings || bindingIndex > maxBinding)
        {
            maxBinding = bindingIndex;
            hasBindings = true;
        }
    };

    for (unsigned i = 0; i < layout->getParameterCount(); ++i)
        scan(layout->getParameterByIndex(i));
    for (SlangUInt e = 0; e < layout->getEntryPointCount(); ++e)
    {
        auto *ep = layout->getEntryPointByIndex(e);
        if (!ep)
            continue;
        for (unsigned i = 0; i < ep->getParameterCount(); ++i)
            scan(ep->getParameterByIndex(i));
    }

    if (!hasBindings)
    {
        outTypes.clear();
        return;
    }

    // Check for gaps
    std::vector<bool> present(maxBinding + 1, false);
    for (auto &[idx, _] : bindings)
        present[idx] = true;
    for (uint32_t i = 0; i <= maxBinding; ++i)
    {
        if (!present[i])
            throw GpuError("non-contiguous bindings not supported; missing binding " +
                           std::to_string(i));
    }

    outTypes.resize(maxBinding + 1);
    for (auto &[idx, type] : bindings)
        outTypes[idx] = type;
}

uint32_t reflectPushConstantSize(slang::ProgramLayout *layout)
{
    if (!layout)
        return 0;

    uint32_t size = 0;
    auto scan = [&size](slang::VariableLayoutReflection *param)
    {
        if (!param)
            return;
        auto *tl = param->getTypeLayout();
        if (!tl || tl->getBindingRangeCount() == 0)
            return;
        if (tl->getBindingRangeType(0) != slang::BindingType::PushConstant)
            return;

        auto *elem = tl->getElementTypeLayout();
        auto raw = elem ? elem->getSize(slang::ParameterCategory::Uniform)
                        : tl->getSize(slang::ParameterCategory::Uniform);
        if (raw == SLANG_UNKNOWN_SIZE || raw == SLANG_UNBOUNDED_SIZE)
            throw GpuError("push constant size is not statically known");
        if (raw > UINT32_MAX)
            throw GpuError("push constant block too large");
        if (size != 0 && size != static_cast<uint32_t>(raw))
            throw GpuError("conflicting push constant block sizes in shader");
        size = static_cast<uint32_t>(raw);
    };

    for (unsigned i = 0; i < layout->getParameterCount(); ++i)
        scan(layout->getParameterByIndex(i));
    for (SlangUInt e = 0; e < layout->getEntryPointCount(); ++e)
    {
        auto *ep = layout->getEntryPointByIndex(e);
        if (!ep)
            continue;
        for (unsigned i = 0; i < ep->getParameterCount(); ++i)
            scan(ep->getParameterByIndex(i));
    }
    return size;
}

void printDiag(slang::IBlob *&diag, const char *prefix = "Slang")
{
    if (!diag)
        return;
    std::cerr << prefix << ": "
              << std::string_view(static_cast<const char *>(diag->getBufferPointer()),
                                  diag->getBufferSize());
    diag->release();
    diag = nullptr;
}

template <typename T> class SlangPtr
{
    T *ptr_ = nullptr;

    public:
    SlangPtr() = default;
    explicit SlangPtr(T *p) : ptr_(p) {}
    SlangPtr(const SlangPtr &) = delete;
    SlangPtr &operator=(const SlangPtr &) = delete;
    SlangPtr(SlangPtr &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    SlangPtr &operator=(SlangPtr &&other) noexcept
    {
        if (this != &other)
        {
            if (ptr_)
                ptr_->release();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    ~SlangPtr()
    {
        if (ptr_)
            ptr_->release();
    }

    T **writeRef()
    {
        if (ptr_)
        {
            ptr_->release();
            ptr_ = nullptr;
        }
        return &ptr_;
    }
    T *get() const { return ptr_; }
    T *operator->() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }
    T *detach()
    {
        T *result = ptr_;
        ptr_ = nullptr;
        return result;
    }
};

slang::IGlobalSession *sharedGlobalSession()
{
    static SlangPtr<slang::IGlobalSession> session = []
    {
        SlangPtr<slang::IGlobalSession> s;
        slang::createGlobalSession(s.writeRef());
        if (s)
        {
            // Slang locates its downstream compiler modules (slang-glslang,
            // spirv-opt) by loading a shared library from a path prefix. On
            // Linux/macOS the loader does not honor the process RUNPATH, and on
            // Windows LoadLibrary does not search the caller's directory, so the
            // modules are not found unless LD_LIBRARY_PATH/PATH happens to be set.
            // Resolve the directory of the linked Slang library and point Slang
            // at it explicitly.
            void *vtable = *reinterpret_cast<void **>(s.get());

#if defined(_WIN32)
            HMODULE mod = nullptr;
            std::string dir;
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   static_cast<LPCSTR>(vtable), &mod))
            {
                char path[MAX_PATH];
                DWORD len = GetModuleFileNameA(mod, path, MAX_PATH);
                if (len != 0 && len < MAX_PATH)
                {
                    std::string full(path, len);
                    auto slash = full.find_last_of("/\\");
                    dir = (slash == std::string::npos) ? "." : full.substr(0, slash);
                }
            }
#else
            Dl_info info = {};
            std::string dir;
            if (dladdr(vtable, &info) && info.dli_fname)
            {
                std::string full(info.dli_fname);
                auto slash = full.find_last_of("/\\");
                dir = (slash == std::string::npos) ? "." : full.substr(0, slash);
            }
#endif

            if (!dir.empty())
            {
                s->setDownstreamCompilerPath(SLANG_PASS_THROUGH_GLSLANG, dir.c_str());
                s->setDownstreamCompilerPath(SLANG_PASS_THROUGH_SPIRV_OPT, dir.c_str());
                s->setDownstreamCompilerPath(SLANG_PASS_THROUGH_SPIRV_DIS, dir.c_str());
                s->setDownstreamCompilerPath(SLANG_PASS_THROUGH_SPIRV_LINK, dir.c_str());
            }
        }
        return s;
    }();
    return session.get();
}

CompiledShader compileWithSession(slang::IGlobalSession * /*globalSession*/, slang::IModule *module,
                                  const std::string &entryPoint, slang::ISession *slangSession,
                                  const std::string &context)
{
    slang::IEntryPoint *entryPointObj = nullptr;
    module->findEntryPointByName(entryPoint.c_str(), &entryPointObj);
    if (!entryPointObj)
        throw GpuError("Slang entry point not found: " + entryPoint);

    slang::IComponentType *components[] = {module, entryPointObj};
    slang::IComponentType *composed = nullptr;
    slang::IBlob *diag = nullptr;
    slangSession->createCompositeComponentType(components, 2, &composed, &diag);
    printDiag(diag);
    if (!composed)
        throw GpuError("Slang composition failed" + context);

    slang::IComponentType *linked = nullptr;
    slang::IBlob *linkDiag = nullptr;
    composed->link(&linked, &linkDiag);
    printDiag(linkDiag);
    if (!linked)
        throw GpuError("Slang linking failed" + context);

    uint32_t pushConstantSize = reflectPushConstantSize(linked->getLayout());
    std::vector<vk::DescriptorType> bindingTypes;
    reflectBindings(linked->getLayout(), bindingTypes);
    slang::IBlob *spirvBlob = nullptr;
    slang::IBlob *codeDiag = nullptr;
    linked->getEntryPointCode(0, 0, &spirvBlob, &codeDiag);
    printDiag(codeDiag);

    if (!spirvBlob)
        throw GpuError("Slang SPIR-V generation failed" + context);

    auto *ptr = reinterpret_cast<const uint32_t *>(spirvBlob->getBufferPointer());
    auto len = spirvBlob->getBufferSize() / 4;
    std::vector<uint32_t> result(ptr, ptr + len);

    spirvBlob->release();
    linked->release();
    composed->release();
    entryPointObj->release();

    return {std::move(result), pushConstantSize, std::move(bindingTypes)};
}

slang::SessionDesc makeSessionDesc(slang::TargetDesc &targetDesc,
                                   std::vector<slang::CompilerOptionEntry> &optEntries,
                                   const ShaderOptions &options,
                                   const std::vector<const char *> &searchPathPtrs)
{
    targetDesc = {};
    targetDesc.structureSize = sizeof(slang::TargetDesc);
    targetDesc.format = SLANG_SPIRV;

    optEntries.clear();
    optEntries.push_back(
        {slang::CompilerOptionName::Optimization,
         {slang::CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_MAXIMAL, 0, nullptr,
          nullptr}});
    for (const auto &[name, value] : options.macros)
    {
        slang::CompilerOptionEntry entry;
        entry.name = slang::CompilerOptionName::MacroDefine;
        entry.value.kind = slang::CompilerOptionValueKind::String;
        entry.value.stringValue0 = name.c_str();
        entry.value.stringValue1 = value.c_str();
        optEntries.push_back(entry);
    }
    targetDesc.compilerOptionEntries = optEntries.data();
    targetDesc.compilerOptionEntryCount = static_cast<uint32_t>(optEntries.size());

    // Default to cs_6_0 (emits SPIR-V 1.3), matching the library's Vulkan 1.1
    // instance (gpu_context.cpp) so validation accepts the binaries. Callers that
    // need newer features (e.g. CooperativeMatrixKHR on SPIR-V 1.6) pass an
    // explicit profile.
    const char *profile =
        (!options.spirvProfile.empty()) ? options.spirvProfile.c_str() : "cs_6_0";
    targetDesc.profile = sharedGlobalSession()->findProfile(profile);

    slang::SessionDesc desc = {};
    desc.structureSize = sizeof(slang::SessionDesc);
    desc.targets = &targetDesc;
    desc.targetCount = 1;

    // Entries must be visible at the session (linkage) level too: the
    // preprocessor reads macros from the linkage option set, while the
    // target-level entries only feed code generation.
    desc.compilerOptionEntries = optEntries.data();
    desc.compilerOptionEntryCount = static_cast<uint32_t>(optEntries.size());

    if (!searchPathPtrs.empty())
    {
        desc.searchPaths = searchPathPtrs.data();
        desc.searchPathCount = static_cast<int>(searchPathPtrs.size());
    }

    return desc;
}

// Serialize options into a cache-key fragment so different macro sets or
// profiles never share a cached compilation.
std::string optionsCacheKey(const ShaderOptions &options)
{
    std::string key;
    for (const auto &[name, value] : options.macros)
        key += "\1" + name + "=" + value;
    if (!options.spirvProfile.empty())
        key += std::string("\1profile=") + options.spirvProfile;
    return key;
}

} // namespace

CompiledShader compileSlangToSpirV(const std::string &source, const std::string &entryPoint,
                                   const std::string &searchPath, const ShaderOptions &options)
{
    std::string key = source + "\1" + entryPoint + "\1" + searchPath + optionsCacheKey(options);

    {
        std::lock_guard lock(cacheMutex);
        auto it = spirvCache.find(key);
        if (it != spirvCache.end())
            return it->second;
    }

    auto *globalSession = sharedGlobalSession();

    std::vector<const char *> searchPathPtrs;
    if (!searchPath.empty())
        searchPathPtrs.push_back(searchPath.c_str());

    slang::TargetDesc targetDesc;
    std::vector<slang::CompilerOptionEntry> optEntries;
    auto desc = makeSessionDesc(targetDesc, optEntries, options, searchPathPtrs);

    SlangPtr<slang::ISession> slangSession;
    globalSession->createSession(desc, slangSession.writeRef());

    slang::IBlob *diagnostics = nullptr;
    auto *module = slangSession->loadModuleFromSourceString("shader", "shader.slang",
                                                            source.c_str(), &diagnostics);
    std::string diagText;
    if (diagnostics)
    {
        diagText = std::string(static_cast<const char *>(diagnostics->getBufferPointer()),
                               diagnostics->getBufferSize());
        diagnostics->release();
        diagnostics = nullptr;
    }
    if (!module)
        throw GpuError(std::string("Slang compilation failed") +
                       (diagText.empty() ? "" : ":\n" + diagText));

    auto result = compileWithSession(globalSession, module, entryPoint, slangSession.get(), "");
    // Some distributed Slang builds corrupt their allocator when a session is
    // released after loading a source module. Keep the session alive for the
    // process; the shader cache is bounded and this avoids a library-owned crash.
    (void)slangSession.detach();

    {
        std::lock_guard lock(cacheMutex);
        if (spirvCache.size() >= kMaxCacheEntries)
            spirvCache.clear();
        spirvCache[key] = result;
    }
    return result;
}

CompiledShader compileSlangModule(const std::string &moduleName, const std::string &entryPoint,
                                  const std::vector<std::string> &searchPaths,
                                  const ShaderOptions &options)
{
    std::string key = moduleName + "\1" + entryPoint;
    for (const auto &p : searchPaths)
        key += "\1" + p;
    key += optionsCacheKey(options);

    {
        std::lock_guard lock(cacheMutex);
        auto it = moduleCache.find(key);
        if (it != moduleCache.end())
            return it->second;
    }

    auto *globalSession = sharedGlobalSession();

    std::vector<const char *> searchPathPtrs;
    for (const auto &p : searchPaths)
        searchPathPtrs.push_back(p.c_str());

    slang::TargetDesc targetDesc;
    std::vector<slang::CompilerOptionEntry> optEntries;
    auto desc = makeSessionDesc(targetDesc, optEntries, options, searchPathPtrs);

    SlangPtr<slang::ISession> slangSession;
    globalSession->createSession(desc, slangSession.writeRef());

    slang::IBlob *diagnostics = nullptr;
    auto *module = slangSession->loadModule(moduleName.c_str(), &diagnostics);
    printDiag(diagnostics);

    if (!module)
        throw GpuError("Slang module load failed: " + moduleName);

    std::string ctx = " for: " + moduleName;
    auto result = compileWithSession(globalSession, module, entryPoint, slangSession.get(), ctx);
    (void)slangSession.detach();

    {
        std::lock_guard lock(cacheMutex);
        if (moduleCache.size() >= kMaxCacheEntries)
            moduleCache.clear();
        moduleCache[key] = result;
    }
    return result;
}

void clearShaderCache()
{
    std::lock_guard lock(cacheMutex);
    spirvCache.clear();
    moduleCache.clear();
}

} // namespace pix
