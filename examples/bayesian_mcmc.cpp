#include "pixie_compute/gpu_buffer.hpp"
#include "pixie_compute/gpu_command_buffer.hpp"
#include "pixie_compute/gpu_compute_pipeline.hpp"
#include "pixie_compute/gpu_context.hpp"
#include "pixie_compute/utility.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

using namespace pix;

struct ChainState
{
    float x;
    float log_prob;
    uint32_t accepted;
    uint32_t rng_state;
};

static constexpr const char *shader = R"(
struct ChainState {
    float x;
    float log_prob;
    uint  accepted;
    uint  rng_state;
};

RWStructuredBuffer<ChainState> chains     : register(u0);
RWStructuredBuffer<float>     samples    : register(u1);
RWStructuredBuffer<float>     obs_data   : register(u2);

struct Params {
    float prior_mean;
    float prior_var;
    float likelihood_var;
    uint  n_obs;
    uint  n_samples;
    uint  n_burnin;
    float step_size;
    uint  total_steps;
};
[[vk::push_constant]] cbuffer PC : register(b0) { Params params; };

uint xorshift(uint s)
{
    s ^= s << 13u;
    s ^= s >> 17u;
    s ^= s << 5u;
    return s;
}

float uniform01(inout uint state)
{
    state = xorshift(state);
    return float(state) / 4294967295.0;
}

float box_muller_gauss(inout uint state)
{
    float u1 = uniform01(state);
    float u2 = uniform01(state);
    float r = sqrt(-2.0 * log(max(u1, 1e-20)));
    return r * cos(6.2831853 * u2);
}

float compute_log_prior(float x, float mean, float var)
{
    float dx = x - mean;
    return -0.5 * dx * dx / var;
}

float compute_log_likelihood(float x, uint n_obs, float lvar)
{
    float lp = 0.0;
    for (uint i = 0u; i < n_obs; ++i)
    {
        float dx = x - obs_data[i];
        lp += -0.5 * dx * dx / lvar;
    }
    return lp;
}

[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint chain_idx = tid.x;
    ChainState st = chains[chain_idx];

    for (uint step = 0u; step < params.total_steps; ++step)
    {
        float proposal = st.x + params.step_size * box_muller_gauss(st.rng_state);

        float log_prior_new = compute_log_prior(proposal, params.prior_mean, params.prior_var);
        float log_lik_new   = compute_log_likelihood(proposal, params.n_obs, params.likelihood_var);
        float log_post_new  = log_prior_new + log_lik_new;

        float log_accept = log_post_new - st.log_prob;
        float u = log(max(uniform01(st.rng_state), 1e-20));

        if (u < log_accept)
        {
            st.x = proposal;
            st.log_prob = log_post_new;
            st.accepted++;
        }

        if (step >= params.n_burnin)
        {
            uint sample_idx = chain_idx * params.n_samples + (step - params.n_burnin);
            samples[sample_idx] = st.x;
        }
    }

    chains[chain_idx] = st;
}
)";

int main()
{
    try
    {
        GpuContext ctx;
        std::printf("[bayesian_mcmc] GPU: %s\n", ctx.deviceName().c_str());

        constexpr uint32_t nChains = 128;
        constexpr uint32_t nData = 50;
        constexpr uint32_t nBurnin = 1000;
        constexpr uint32_t nSamples = 2000;
        constexpr uint32_t totalSteps = nBurnin + nSamples;

        constexpr float trueMean = 3.0f;
        constexpr float trueVar = 1.0f;
        constexpr float priorMean = 0.0f;
        constexpr float priorVar = 100.0f;
        constexpr float stepSize = 0.5f;

        std::mt19937 rng(42);
        std::normal_distribution<float> dataDist(trueMean, std::sqrt(trueVar));
        std::vector<float> data(nData);
        for (auto &d : data)
            d = dataDist(rng);

        float dataMean = 0.0f;
        for (auto d : data)
            dataMean += d;
        dataMean /= float(nData);

        float sampleVar = 0.0f;
        for (auto d : data)
        {
            float dx = d - dataMean;
            sampleVar += dx * dx;
        }
        sampleVar /= float(nData);

        float postVar = 1.0f / (1.0f / priorVar + float(nData) / trueVar);
        float postMean = postVar * (priorMean / priorVar + float(nData) * dataMean / trueVar);

        std::printf("[bayesian_mcmc] Analytical posterior: N(%.4f, %.4f)\n", postMean, postVar);

        std::vector<ChainState> initChains(nChains);
        std::mt19937 seedRng(123);
        for (uint32_t i = 0; i < nChains; ++i)
        {
            initChains[i].x = 0.0f;
            initChains[i].log_prob = -1e30f;
            initChains[i].accepted = 0;
            initChains[i].rng_state = seedRng();
            if (initChains[i].rng_state == 0)
                initChains[i].rng_state = 1;
        }

        GpuBuffer buf_chains(ctx, nChains * sizeof(ChainState), GpuBuffer::Type::HostCoherent);
        GpuBuffer buf_samples(ctx, nChains * nSamples * sizeof(float),
                              GpuBuffer::Type::HostCoherent);
        GpuBuffer buf_data(ctx, nData * sizeof(float), GpuBuffer::Type::HostCoherent);

        buf_chains.upload(std::span<const ChainState>(initChains));
        buf_data.upload(std::span<const float>(data));

        struct PushConstants
        {
            float prior_mean;
            float prior_var;
            float likelihood_var;
            uint32_t n_obs;
            uint32_t n_samples;
            uint32_t n_burnin;
            float step_size;
            uint32_t total_steps;
        } pc{priorMean, priorVar, trueVar, nData, nSamples, nBurnin, stepSize, totalSteps};

        GpuComputePipeline pipeline(ctx, shader,
                                    {
                                        buf_chains.descriptorInfo(),
                                        buf_samples.descriptorInfo(),
                                        buf_data.descriptorInfo(),
                                    },
                                    sizeof(pc), 1, 1, 1);

        GpuCommandBuffer cmd(ctx);
        cmd.begin();
        cmd.bind(pipeline);
        cmd.pushConstants(pipeline, &pc, sizeof(pc));
        cmd.dispatch(pipeline, nChains, 1, 1);
        cmd.end();
        cmd.submitAndWait();

        std::vector<ChainState> finalChains(nChains);
        buf_chains.download(std::span<ChainState>(finalChains));

        std::vector<float> allSamples(nChains * nSamples);
        buf_samples.download(std::span<float>(allSamples));

        uint32_t totalAccepted = 0;
        for (auto &c : finalChains)
            totalAccepted += c.accepted;

        double sampleMean = 0.0;
        for (auto s : allSamples)
            sampleMean += double(s);
        sampleMean /= double(allSamples.size());

        double sampleVar2 = 0.0;
        for (auto s : allSamples)
        {
            double dx = double(s) - sampleMean;
            sampleVar2 += dx * dx;
        }
        sampleVar2 /= double(allSamples.size() - 1);

        double acceptRate = double(totalAccepted) / double(nChains * totalSteps);
        double meanError = std::fabs(sampleMean - double(postMean));
        double varError = std::fabs(sampleVar2 - double(postVar));

        std::printf("[bayesian_mcmc] chains=%u  samples=%u  accept_rate=%.2f%%\n", nChains,
                    nChains * nSamples, acceptRate * 100.0);
        std::printf("[bayesian_mcmc] posterior mean:  GPU=%.4f  exact=%.4f  err=%.4f\n", sampleMean,
                    postMean, meanError);
        std::printf("[bayesian_mcmc] posterior var:   GPU=%.4f  exact=%.4f  err=%.4f\n", sampleVar2,
                    postVar, varError);

        bool pass = meanError < 0.15 && varError < 0.15;
        std::printf("[bayesian_mcmc] %s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }
    catch (const GpuError &e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
