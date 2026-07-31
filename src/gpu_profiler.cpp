#include "pixie_compute/gpu_profiler.hpp"

#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/utility.hpp"
#include "timestamp_util.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace pix
{

GpuProfiler::GpuProfiler(GpuContext &ctx, uint32_t maxSections)
    : m_device(ctx.device()), m_timestampPeriod(ctx.timestampPeriod()), m_maxSections(maxSections)
{
    if (maxSections == 0)
        throw GpuError("GpuProfiler maxSections must be > 0");
    if (!ctx.limits().timestampComputeAndGraphics)
        throw GpuError("GpuProfiler requires timestampComputeAndGraphics support");

    vk::QueryPoolCreateInfo poolInfo{};
    poolInfo.queryType = vk::QueryType::eTimestamp;
    poolInfo.queryCount = maxSections * 2;
    m_queryPool = m_device.createQueryPoolUnique(poolInfo);
}

void GpuProfiler::beginFrame(vk::CommandBuffer cmd)
{
    if (m_inFrame)
        throw GpuError("beginFrame() called before endFrame() of the previous frame");
    cmd.resetQueryPool(*m_queryPool, 0, m_maxSections * 2);
    m_names.clear();
    m_stack.clear();
    m_next = 0;
    m_inFrame = true;
}

GpuProfiler::ScopedSection GpuProfiler::begin(vk::CommandBuffer cmd, const std::string &name)
{
    return ScopedSection(*this, cmd, name);
}

void GpuProfiler::beginSection(vk::CommandBuffer cmd, const std::string &name)
{
    if (!m_inFrame)
        throw GpuError("beginSection() called before beginFrame()");
    if (m_next >= m_maxSections)
        throw GpuError("GpuProfiler: too many sections in one frame");
    uint32_t index = m_next++;
    m_names.push_back(name);
    m_stack.push_back(index);
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *m_queryPool, index * 2);
}

void GpuProfiler::endSection(vk::CommandBuffer cmd)
{
    if (m_stack.empty())
        throw GpuError("endSection() without a matching beginSection()");
    uint32_t index = m_stack.back();
    m_stack.pop_back();
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *m_queryPool, index * 2 + 1);
}

std::vector<GpuProfilerFrameSection> GpuProfiler::endFrame()
{
    if (!m_inFrame)
        throw GpuError("endFrame() called without beginFrame()");
    if (!m_stack.empty())
        throw GpuError("endFrame() with unbalanced sections (open section would hang the read)");
    m_inFrame = false;

    std::vector<GpuProfilerFrameSection> sections;
    sections.reserve(m_names.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_names.size()); ++i)
    {
        double ms = readMs(i);
        const auto &name = m_names[i];
        sections.push_back({name, ms});

        auto &s = m_stats[name];
        s.lastMs = ms;
        if (s.frameCount == 0)
        {
            s.avgMs = ms;
            s.minMs = ms;
            s.maxMs = ms;
        }
        else
        {
            s.avgMs = (s.avgMs * static_cast<double>(s.frameCount) + ms) /
                      static_cast<double>(s.frameCount + 1);
            s.minMs = std::min(s.minMs, ms);
            s.maxMs = std::max(s.maxMs, ms);
        }
        ++s.frameCount;
    }
    return sections;
}

std::string GpuProfiler::toJson() const
{
    auto escape = [](const std::string &value)
    {
        std::string result;
        for (char raw : value)
        {
            const unsigned char c = static_cast<unsigned char>(raw);
            switch (c)
            {
            case '\\':
                result += "\\\\";
                break;
            case '"':
                result += "\\\"";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                result += "\\r";
                break;
            case '\t':
                result += "\\t";
                break;
            default:
                result += static_cast<char>(c);
                break;
            }
        }
        return result;
    };

    std::ostringstream out;
    out << "{\"sections\":[";
    bool first = true;
    out << std::setprecision(12);
    for (const auto &[name, stats] : m_stats)
    {
        if (!first)
            out << ',';
        first = false;
        out << "{\"name\":\"" << escape(name) << "\",\"lastMs\":" << stats.lastMs
            << ",\"avgMs\":" << stats.avgMs << ",\"minMs\":" << stats.minMs
            << ",\"maxMs\":" << stats.maxMs << ",\"frameCount\":" << stats.frameCount << '}';
    }
    out << "]}";
    return out.str();
}

std::string GpuProfiler::toCsv() const
{
    auto escapeCsv = [](const std::string &value)
    {
        std::string result = "\"";
        for (char c : value)
        {
            if (c == '"')
                result += "\"\"";
            else
                result += c;
        }
        result += '"';
        return result;
    };

    std::ostringstream out;
    out << "name,last_ms,avg_ms,min_ms,max_ms,frame_count\n";
    out << std::setprecision(12);
    for (const auto &[name, stats] : m_stats)
        out << escapeCsv(name) << ',' << stats.lastMs << ',' << stats.avgMs << ',' << stats.minMs
            << ',' << stats.maxMs << ',' << stats.frameCount << '\n';
    return out.str();
}

double GpuProfiler::readMs(uint32_t pairIndex) const
{
    return detail::readTimestampMs(m_device, *m_queryPool, pairIndex, m_timestampPeriod);
}

void GpuProfiler::reset()
{
    m_stats.clear();
}

GpuProfiler::ScopedSection::ScopedSection(GpuProfiler &profiler, vk::CommandBuffer cmd,
                                          const std::string &name)
    : m_profiler(&profiler), m_cmd(cmd)
{
    profiler.beginSection(cmd, name);
}

GpuProfiler::ScopedSection::~ScopedSection()
{
    if (m_profiler)
        m_profiler->endSection(m_cmd);
}

GpuProfiler::ScopedSection::ScopedSection(ScopedSection &&other) noexcept
    : m_profiler(other.m_profiler), m_cmd(other.m_cmd)
{
    other.m_profiler = nullptr;
}

} // namespace pix
