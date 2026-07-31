#pragma once

// Minimal, self-contained mirror of the RenderDoc in-application API (ABI 1.6.0,
// https://renderdoc.org/docs/in_application_api.html). It replaces the vendored
// renderdoc_app.h so the library does not need the RenderDoc SDK headers.
//
// RenderDoc only exports a single symbol, RENDERDOC_GetAPI, which fills a struct
// of function pointers. The struct layout is append-only across versions: newer
// members are appended at the end, so the prefix up to API 1.6.0 is stable as
// long as we request that version. This file declares only that prefix; the
// static_asserts below pin the offsets of the members we call.

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define PIXIE_RENDERDOC_CC __cdecl
#else
#define PIXIE_RENDERDOC_CC
#endif

namespace pix
{

constexpr int kRenderDocApiVersion_1_6_0 = 10600;

using RenderDocGetApi = int(PIXIE_RENDERDOC_CC *)(int version, void **outApiPointers);

/// First 23 pointer slots of RENDERDOC_API_1_6_0. Members we do not call are kept
/// as opaque void* placeholders to preserve the layout.
struct RenderDocApi
{
    void *GetAPIVersion;                                                            //  0
    void *SetCaptureOptionU32;                                                      //  1
    void *SetCaptureOptionF32;                                                      //  2
    void *GetCaptureOptionU32;                                                      //  3
    void *GetCaptureOptionF32;                                                      //  4
    void *SetFocusToggleKeys;                                                       //  5
    void *SetCaptureKeys;                                                           //  6
    void *GetOverlayBits;                                                           //  7
    void *MaskOverlayBits;                                                          //  8
    void *RemoveHooks;                                                              //  9
    void *UnloadCrashHandler;                                                       // 10
    void(PIXIE_RENDERDOC_CC *SetCaptureFilePathTemplate)(const char *pathTemplate); // 11
    void *GetCaptureFilePathTemplate;                                               // 12
    void *GetNumCaptures;                                                           // 13
    void *GetCapture;                                                               // 14
    void(PIXIE_RENDERDOC_CC *TriggerCapture)();                                     // 15
    uint32_t(PIXIE_RENDERDOC_CC *IsTargetControlConnected)();                       // 16
    uint32_t(PIXIE_RENDERDOC_CC *LaunchReplayUI)(uint32_t connectTargetControl,
                                                 const char *cmdline);               // 17
    void *SetActiveWindow;                                                           // 18
    void(PIXIE_RENDERDOC_CC *StartFrameCapture)(void *device, void *windowHandle);   // 19
    uint32_t(PIXIE_RENDERDOC_CC *IsFrameCapturing)();                                // 20
    uint32_t(PIXIE_RENDERDOC_CC *EndFrameCapture)(void *device, void *windowHandle); // 21
    void(PIXIE_RENDERDOC_CC *TriggerMultiFrameCapture)(uint32_t numFrames);          // 22
};

static_assert(offsetof(RenderDocApi, SetCaptureFilePathTemplate) == 11 * sizeof(void *));
static_assert(offsetof(RenderDocApi, TriggerCapture) == 15 * sizeof(void *));
static_assert(offsetof(RenderDocApi, IsTargetControlConnected) == 16 * sizeof(void *));
static_assert(offsetof(RenderDocApi, LaunchReplayUI) == 17 * sizeof(void *));
static_assert(offsetof(RenderDocApi, StartFrameCapture) == 19 * sizeof(void *));
static_assert(offsetof(RenderDocApi, IsFrameCapturing) == 20 * sizeof(void *));
static_assert(offsetof(RenderDocApi, EndFrameCapture) == 21 * sizeof(void *));
static_assert(offsetof(RenderDocApi, TriggerMultiFrameCapture) == 22 * sizeof(void *));

} // namespace pix
