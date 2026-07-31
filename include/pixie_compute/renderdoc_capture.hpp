#pragma once

#include <cstdint>
#include <string>

namespace pix
{

/// Optional integration with the RenderDoc capture tool. RenderDoc is never linked
/// against; the library is probed and, optionally, loaded at runtime. The RenderDoc
/// ABI is mirrored locally (src/renderdoc_abi.hpp) so no RenderDoc SDK header is
/// vendored or required.
///
/// Construct before creating a GpuContext so RenderDoc can hook the instance and
/// device creation. When RenderDoc is not installed, every method is a safe no-op
/// and available() returns false. The library is kept mapped for the lifetime of
/// the process (never unloaded on POSIX; on Windows the destructor frees it), since
/// RenderDoc's Vulkan hooks must stay valid as long as the instance exists.
///
/// For headless compute (no presentation), programmatic captures work: wrap the
/// work you want captured in startCapture()/endCapture() or a ScopedCapture. A
/// connected RenderDoc UI is only needed for target control features.
class RenderDocCapture
{
    public:
    /// Probe for RenderDoc. When loadIfMissing is true and the library is not already
    /// loaded in this process, it is loaded explicitly (e.g. dlopen("librenderdoc.so")).
    explicit RenderDocCapture(bool loadIfMissing = true);
    ~RenderDocCapture();

    RenderDocCapture(const RenderDocCapture &) = delete;
    RenderDocCapture &operator=(const RenderDocCapture &) = delete;
    RenderDocCapture(RenderDocCapture &&) = delete;
    RenderDocCapture &operator=(RenderDocCapture &&) = delete;

    /// True if the RenderDoc API was found and captures will actually work.
    bool available() const noexcept { return m_api != nullptr; }

    /// True if a frame capture is currently in progress.
    bool isCapturing() const;

    /// Start/stop a capture around a frame's GPU work. No-ops when unavailable.
    void startCapture();
    void endCapture();

    /// Trigger a capture on the next frame(s) (as if by the RenderDoc hotkey).
    void triggerCapture();
    void triggerMultiFrameCapture(uint32_t numFrames);

    /// True when the RenderDoc UI is connected to this process (target control).
    bool targetControlConnected() const;

    /// Ask the connected RenderDoc UI to open its replay window.
    void launchReplayUI() const;

    /// Set the capture file path template (see RenderDoc docs for the format, e.g.
    /// "<project>_<frame>.rdc").
    void setCaptureFilePathTemplate(const std::string &pathTemplate);

    /// True if RenderDoc is already loaded in this process (passive probe, never loads).
    static bool isLoadedInProcess();

    private:
    void *m_library = nullptr;
    void *m_api = nullptr;
};

/// RAII guard: starts a capture on construction and ends it on destruction.
/// No-op when the capture object is unavailable.
class ScopedCapture
{
    public:
    explicit ScopedCapture(RenderDocCapture &capture) : m_capture(&capture)
    {
        capture.startCapture();
    }
    ~ScopedCapture()
    {
        if (m_capture)
            m_capture->endCapture();
    }

    ScopedCapture(const ScopedCapture &) = delete;
    ScopedCapture &operator=(const ScopedCapture &) = delete;
    ScopedCapture(ScopedCapture &&other) noexcept : m_capture(other.m_capture)
    {
        other.m_capture = nullptr;
    }
    ScopedCapture &operator=(ScopedCapture &&) = delete;

    private:
    RenderDocCapture *m_capture = nullptr;
};

} // namespace pix
