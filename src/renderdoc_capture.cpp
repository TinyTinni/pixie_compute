#include "pixie_compute/renderdoc_capture.hpp"

#include "renderdoc_abi.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace pix
{

namespace
{

const char *renderdocLibraryName()
{
#if defined(_WIN32)
    return "renderdoc.dll";
#elif defined(__ANDROID__)
    return "libVkLayer_GLES_RenderDoc.so";
#elif defined(__APPLE__)
    return "librenderdoc.dylib";
#else
    return "librenderdoc.so";
#endif
}

void *loadLibrary(const char *name, bool loadIfMissing)
{
#if defined(_WIN32)
    if (loadIfMissing)
        return reinterpret_cast<void *>(LoadLibraryA(name));
    return reinterpret_cast<void *>(GetModuleHandleA(name));
#else
    return loadIfMissing ? dlopen(name, RTLD_NOW) : dlopen(name, RTLD_NOW | RTLD_NOLOAD);
#endif
}

void *resolveSymbol(void *library, const char *symbol)
{
#if defined(_WIN32)
    return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(library), symbol));
#else
    return dlsym(library, symbol);
#endif
}

} // namespace

RenderDocCapture::RenderDocCapture(bool loadIfMissing)
    : m_library(loadLibrary(renderdocLibraryName(), loadIfMissing))
{
    if (!m_library)
        return;
    auto getApi = reinterpret_cast<RenderDocGetApi>(resolveSymbol(m_library, "RENDERDOC_GetAPI"));
    if (!getApi)
        return;
    void *api = nullptr;
    if (getApi(kRenderDocApiVersion_1_6_0, &api) != 1)
        return;
    m_api = api;
}

RenderDocCapture::~RenderDocCapture()
{
#if defined(_WIN32)
    if (m_library)
        FreeLibrary(static_cast<HMODULE>(m_library));
#else
    // Intentionally do not dlclose(): RenderDoc hooks the Vulkan entry points, and
    // unloading the library would leave dangling hooks behind and can crash. Keeping
    // it mapped for the rest of the process lifetime is safe.
#endif
    m_library = nullptr;
    m_api = nullptr;
}

bool RenderDocCapture::isLoadedInProcess()
{
    return loadLibrary(renderdocLibraryName(), false) != nullptr;
}

bool RenderDocCapture::isCapturing() const
{
    if (!m_api)
        return false;
    return static_cast<RenderDocApi *>(m_api)->IsFrameCapturing() != 0;
}

void RenderDocCapture::startCapture()
{
    if (!m_api)
        return;
    static_cast<RenderDocApi *>(m_api)->StartFrameCapture(nullptr, nullptr);
}

void RenderDocCapture::endCapture()
{
    if (!m_api)
        return;
    static_cast<RenderDocApi *>(m_api)->EndFrameCapture(nullptr, nullptr);
}

void RenderDocCapture::triggerCapture()
{
    if (!m_api)
        return;
    static_cast<RenderDocApi *>(m_api)->TriggerCapture();
}

void RenderDocCapture::triggerMultiFrameCapture(uint32_t numFrames)
{
    if (!m_api)
        return;
    static_cast<RenderDocApi *>(m_api)->TriggerMultiFrameCapture(numFrames);
}

bool RenderDocCapture::targetControlConnected() const
{
    if (!m_api)
        return false;
    return static_cast<RenderDocApi *>(m_api)->IsTargetControlConnected() != 0;
}

void RenderDocCapture::launchReplayUI() const
{
    if (!m_api)
        return;
    static_cast<RenderDocApi *>(m_api)->LaunchReplayUI(1, nullptr);
}

void RenderDocCapture::setCaptureFilePathTemplate(const std::string &pathTemplate)
{
    if (!m_api)
        return;
    static_cast<RenderDocApi *>(m_api)->SetCaptureFilePathTemplate(pathTemplate.c_str());
}

} // namespace pix
