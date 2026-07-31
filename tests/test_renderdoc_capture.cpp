#include "pixie_compute/renderdoc_capture.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace pix;

TEST_CASE("RenderDocCapture is graceful when unavailable", "[renderdoc]")
{
    RenderDocCapture capture(false);

    // Never throws and never reports an in-progress capture without a start.
    REQUIRE_FALSE(capture.isCapturing());
    REQUIRE_NOTHROW(capture.startCapture());
    REQUIRE_NOTHROW(capture.endCapture());
    REQUIRE_NOTHROW(capture.triggerCapture());
    REQUIRE_NOTHROW(capture.triggerMultiFrameCapture(2));
    REQUIRE_NOTHROW(capture.launchReplayUI());
    REQUIRE_NOTHROW(capture.setCaptureFilePathTemplate("test.rdc"));

    // available() can only be true when the library is actually loaded in-process.
    bool loaded = RenderDocCapture::isLoadedInProcess();
    bool ok = !capture.available() || loaded;
    REQUIRE(ok);
}

TEST_CASE("RenderDocCapture default construction is safe without a library", "[renderdoc]")
{
    REQUIRE_NOTHROW(RenderDocCapture(true));
    RenderDocCapture capture(true);
    REQUIRE_FALSE(capture.isCapturing());
}

TEST_CASE("ScopedCapture is graceful when unavailable", "[renderdoc]")
{
    RenderDocCapture capture(false);
    REQUIRE_NOTHROW(ScopedCapture(capture));
}
