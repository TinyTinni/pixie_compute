#pragma once

#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_context.hpp"

namespace pix
{
namespace detail
{

/// RAII guard returning a staging buffer to the GpuContext pool on destruction,
/// including on exception paths. Stack-only; the guard must outlive any GPU work
/// using the buffer.
class StagingScoped
{
    public:
    StagingScoped(GpuContext &ctx, StagingBuffer buf) noexcept : m_ctx(&ctx), m_buf(buf) {}
    ~StagingScoped() { reset(); }

    StagingScoped(const StagingScoped &) = delete;
    StagingScoped &operator=(const StagingScoped &) = delete;
    StagingScoped(StagingScoped &&) = delete;
    StagingScoped &operator=(StagingScoped &&) = delete;

    /// The underlying staging buffer.
    StagingBuffer get() const noexcept { return m_buf; }

    /// Return the buffer to the pool now (idempotent). Safe once the GPU work
    /// referencing the buffer has completed.
    void reset() noexcept
    {
        if (m_buf.buffer && m_ctx)
        {
            m_ctx->releaseStagingBuffer(m_buf);
            m_buf = {};
        }
    }

    private:
    GpuContext *m_ctx;
    StagingBuffer m_buf;
};

} // namespace detail
} // namespace pix
