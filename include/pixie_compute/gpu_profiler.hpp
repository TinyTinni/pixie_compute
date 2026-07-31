#pragma once

#include "pixie_compute/detail/vulkan_include.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace pix
{

class GpuContext;

/// One named timing from a single profiled frame.
struct GpuProfilerFrameSection
{
    std::string name;
    double ms;
};

/// Aggregated statistics for one named section across frames.
struct GpuProfilerStats
{
    double lastMs = 0.0;
    double avgMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
    uint64_t frameCount = 0;
};

/// GPU timestamp profiler for named, per-frame sections.
///
/// Usage:
///   GpuProfiler profiler(ctx);
///   GpuCommandBuffer cmd(ctx);
///   cmd.begin();
///   profiler.beginFrame(cmd.handle());
///   {
///       auto section = profiler.begin(cmd.handle(), "pass1");
///       cmd.bind(pipeline);
///       cmd.dispatch();
///   }
///   cmd.submitAndWait();
///   auto sections = profiler.endFrame();      // blocks until timestamps available
///   auto stats = profiler.stats();
///
/// Sections may nest. Begin and end timestamps use the top- and bottom-of-pipe
/// stages, so every section that is properly closed gets a valid duration even
/// without compute work inside it.
class GpuProfiler
{
    public:
    /// RAII guard that closes the section on destruction. Move-only.
    class ScopedSection
    {
        public:
        ScopedSection(GpuProfiler &profiler, vk::CommandBuffer cmd, const std::string &name);
        ~ScopedSection();

        ScopedSection(const ScopedSection &) = delete;
        ScopedSection &operator=(const ScopedSection &) = delete;
        ScopedSection(ScopedSection &&other) noexcept;
        ScopedSection &operator=(ScopedSection &&) = delete;

        private:
        GpuProfiler *m_profiler = nullptr;
        vk::CommandBuffer m_cmd;
    };

    /// @param maxSections  Maximum number of sections per frame.
    /// Throws GpuError if timestamp queries are not supported by the device.
    explicit GpuProfiler(GpuContext &ctx, uint32_t maxSections = 64);

    GpuProfiler(const GpuProfiler &) = delete;
    GpuProfiler &operator=(const GpuProfiler &) = delete;
    GpuProfiler(GpuProfiler &&) = delete;
    GpuProfiler &operator=(GpuProfiler &&) = delete;

    /// Begin a new frame: reset the query pool and clear this frame's sections.
    /// Call while recording, before any begin()/beginSection(). Requires that the
    /// previous frame was closed with endFrame().
    void beginFrame(vk::CommandBuffer cmd);

    /// Begin a named section; the returned guard writes the matching end timestamp
    /// when it goes out of scope.
    ScopedSection begin(vk::CommandBuffer cmd, const std::string &name);

    /// Manual section control. Every beginSection() must have a matching endSection()
    /// before endFrame().
    void beginSection(vk::CommandBuffer cmd, const std::string &name);
    void endSection(vk::CommandBuffer cmd);

    /// Block until all of this frame's timestamps are available and update the
    /// aggregates. Returns this frame's sections. Throws if any section was left
    /// open (which would otherwise hang the query read).
    std::vector<GpuProfilerFrameSection> endFrame();

    /// Aggregates across all frames since construction or reset().
    const std::map<std::string, GpuProfilerStats> &stats() const noexcept { return m_stats; }

    /// Export aggregate timings as dependency-free JSON or CSV text.
    std::string toJson() const;
    std::string toCsv() const;

    /// Clear all aggregate history (does not affect the current frame).
    void reset();

    private:
    double readMs(uint32_t pairIndex) const;

    vk::Device m_device;
    float m_timestampPeriod;
    uint32_t m_maxSections;
    vk::UniqueQueryPool m_queryPool;

    std::vector<std::string> m_names;
    std::vector<uint32_t> m_stack;
    uint32_t m_next = 0;
    bool m_inFrame = false;
    std::map<std::string, GpuProfilerStats> m_stats;
};

} // namespace pix
