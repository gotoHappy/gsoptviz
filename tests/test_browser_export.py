"""Optional real-browser regression check for standalone startup and seeking."""
from io import BytesIO
import os

import numpy as np
from PIL import Image
import pytest

from examples.make_demo import make_demo
from gsoptviz import Sequence
from gsoptviz.viewer import export_html


@pytest.mark.skipif(not os.environ.get("GSOPTVIZ_BROWSER"),
                    reason="Set GSOPTVIZ_BROWSER to a Chromium executable to run browser checks")
def test_offline_first_frame_and_seeking_without_network(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    output = tmp_path / "animation.html"
    export_html(Sequence(make_demo(tmp_path / "sequence")), output)
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ["GSOPTVIZ_BROWSER"], headless=True,
            args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
        )
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("https://**/*", lambda route: route.abort())
            page.route("http://**/*", lambda route: route.abort())
            page.goto(output.as_uri())
            page.locator('svg[class*="player-play"]').wait_for()
            page.wait_for_timeout(2000)
            assert float(page.locator("input").first.input_value()) == 0

            def rendered_pixels():
                rgb = np.asarray(Image.open(BytesIO(page.screenshot())).convert("RGB"))
                return rgb[200:650, 300:1000]

            first = rendered_pixels()
            # Check the actual colored splats/cameras, not merely a canvas element.
            assert np.count_nonzero(first.max(axis=-1) - first.min(axis=-1) > 50) > 1000
            page.locator("input").first.fill("1.8")
            page.wait_for_timeout(1000)
            last = rendered_pixels()
            assert np.count_nonzero(np.any(first != last, axis=-1)) > 1000
            page.locator("input").first.fill("0")
            page.wait_for_timeout(1000)
            back = rendered_pixels()
            assert np.count_nonzero(back.max(axis=-1) - back.min(axis=-1) > 50) > 1000
            assert not errors
        finally:
            browser.close()
