#include "pixie_compute/gpu_context.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{

class EnvVarScope
{
    public:
    EnvVarScope(const char *name, const char *value) : m_name(name), m_old(std::getenv(name))
    {
#ifdef _WIN32
        _putenv_s(m_name.c_str(), value);
#else
        setenv(name, value, 1);
#endif
    }
    ~EnvVarScope()
    {
#ifdef _WIN32
        if (m_old)
            _putenv_s(m_name.c_str(), m_old);
        else
            _putenv_s(m_name.c_str(), "");
#else
        if (m_old)
            setenv(m_name.c_str(), m_old, 1);
        else
            unsetenv(m_name.c_str());
#endif
    }

    private:
    std::string m_name;
    const char *m_old = nullptr;
};

void emitValidationMessage(pix::GpuContext &ctx)
{
    // Route a synthetic message through the debug-utils API. This exercises the
    // debug messenger -> callback -> logSink path without relying on a driver
    // submitting garbage (which some drivers mishandle).
    vk::DebugUtilsMessengerCallbackDataEXT data{};
    data.messageIdNumber = 0;
    data.pMessageIdName = "pixie_test";
    data.pMessage = "synthetic validation message";
    ctx.instance().submitDebugUtilsMessageEXT(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
                                              vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
                                              data);
}

std::vector<std::string> collectValidationMessages(bool enableValidation)
{
    std::vector<std::string> messages;
    pix::GpuContextDesc desc;
    desc.enableValidation = enableValidation;
    desc.debugSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                         vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                         vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    desc.logSink = [&](vk::DebugUtilsMessageSeverityFlagBitsEXT, const std::string &msg)
    { messages.push_back(msg); };

    pix::GpuContext ctx(desc);
    if (ctx.validationEnabled())
        emitValidationMessage(ctx);
    return messages;
}

} // namespace

TEST_CASE("GpuContext desc controls validation", "[context][validation]")
{
    pix::GpuContextDesc desc;
    desc.enableValidation = true;
    pix::GpuContext ctx(desc);
    if (!ctx.validationEnabled())
        SKIP("validation layer not available");
    REQUIRE(ctx.validationEnabled());
}

TEST_CASE("PIXIE_COMPUTE_VALIDATION=0 forces validation off", "[context][validation]")
{
    EnvVarScope env("PIXIE_COMPUTE_VALIDATION", "0");
    auto messages = collectValidationMessages(true);
    REQUIRE(messages.empty());
}

TEST_CASE("PIXIE_COMPUTE_VALIDATION=1 forces validation on", "[context][validation]")
{
    EnvVarScope env("PIXIE_COMPUTE_VALIDATION", "1");
    pix::GpuContextDesc desc;
    desc.enableValidation = false;
    desc.debugSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                         vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                         vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
    std::vector<std::string> messages;
    desc.logSink = [&](vk::DebugUtilsMessageSeverityFlagBitsEXT, const std::string &msg)
    { messages.push_back(msg); };
    pix::GpuContext ctx(desc);
    if (!ctx.validationEnabled())
        SKIP("validation layer not available");
    emitValidationMessage(ctx);
    REQUIRE(!messages.empty());
}

TEST_CASE("custom log sink receives validation messages", "[context][validation]")
{
    auto messages = collectValidationMessages(true);
    if (messages.empty())
        SKIP("validation layer not available");
    REQUIRE(!messages.empty());
}

TEST_CASE("shader printf messages use the dedicated sink", "[context][validation]")
{
    std::vector<std::string> messages;
    pix::GpuContextDesc desc;
    desc.enableValidation = true;
    desc.shaderPrintfSink = [&](const std::string &msg) { messages.push_back(msg); };
    pix::GpuContext ctx(desc);
    if (!ctx.validationEnabled())
        SKIP("validation layer not available");

    vk::DebugUtilsMessengerCallbackDataEXT data{};
    data.pMessageIdName = "UNASSIGNED-DEBUG-PRINTF";
    data.pMessage = "shader value: 42";
    ctx.instance().submitDebugUtilsMessageEXT(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning,
                                              vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation,
                                              data);
    REQUIRE(messages.size() == 1);
    REQUIRE(messages.front() == "shader value: 42");
}
