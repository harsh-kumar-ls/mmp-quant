"""Browser tests of the web UI (20 Oct 2026, part of the regression loop).

Real clicks in a real (headless) Chromium, against the real server, on a
temporary copy of two real runs -- so what is checked is what the user sees:
the window that "got stuck" (05 Oct), numbers on screen matching result.json,
tabs, keyboard, phone width, and no JavaScript errors.

Run with `pytest -m e2e` (tools/loop_check.sh does). Skipped automatically
where the runs are not available (e.g. a fresh clone: runs/ is not in git).
"""

from __future__ import annotations

import json
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

RUNS = Path("runs")
REDUNDANCY_VIDEO, COHERENCE_VIDEO = "Test_16", "test6"

playwright = pytest.importorskip("playwright.sync_api")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    for v in (REDUNDANCY_VIDEO, COHERENCE_VIDEO):
        if not (RUNS / v / "result.json").exists():
            pytest.skip(f"runs/{v}/result.json not available")
    import uvicorn

    from mmpquant.webapp import server

    runs = tmp_path_factory.mktemp("runs")
    for v in (REDUNDANCY_VIDEO, COHERENCE_VIDEO):
        (runs / v).mkdir()
        shutil.copy(RUNS / v / "result.json", runs / v / "result.json")
    old_runs = server.RUNS_DIR
    server.RUNS_DIR = runs
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port,
                                        log_level="error"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}", "runs": runs}
    srv.should_exit = True
    thread.join(timeout=10)
    server.RUNS_DIR = old_runs


@pytest.fixture
def page(site):
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page(viewport={"width": 1280, "height": 800})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.goto(site["url"])
        yield pg
        browser.close()
        assert not errors, f"JavaScript errors on the page: {errors}"


def _result(site, vid) -> dict:
    return json.loads((site["runs"] / vid / "result.json").read_text())


def _open(page, vid):
    page.locator(".video-list").get_by_text(vid, exact=True).click()
    page.locator(".ring-value strong").wait_for()


class TestReport:
    def test_header_score_matches_result_json(self, page, site):
        _open(page, REDUNDANCY_VIDEO)
        score = _result(site, REDUNDANCY_VIDEO)["result"]["score"]
        assert page.locator(".ring-value strong").inner_text() == f"{score:.1f}"

    def test_every_segment_row_shows_its_score(self, page, site):
        _open(page, REDUNDANCY_VIDEO)
        segs = _result(site, REDUNDANCY_VIDEO)["segments"]
        shown = page.locator(".chart-row .chart-bar-value").all_inner_texts()
        assert shown == [f"{s['score']:.2f}" for s in segs]


class TestSegmentDrawer:
    """The 05 Oct bug: the pop-up was 3,000+ px tall and froze the page."""

    def test_opens_scrolls_navigates_and_closes(self, page, site):
        _open(page, REDUNDANCY_VIDEO)
        segs = _result(site, REDUNDANCY_VIDEO)["segments"]
        page.locator('.chart-row[data-segment="0"]').click()
        drawer = page.locator(".drawer")
        drawer.wait_for()
        assert page.evaluate("document.documentElement.classList.contains('drawer-open')")
        assert drawer.locator(".drawer-score strong").inner_text() == f"{segs[0]['score']:.2f}"

        body = drawer.locator(".drawer-body")
        body.evaluate("el => { for (const d of el.querySelectorAll('details')) d.open = true; }")
        assert body.evaluate("el => el.scrollHeight > el.clientHeight"), "nothing to scroll"
        # Scroll the way a person does -- the mouse wheel over the panel. (A
        # scripted scrollTo() works even where the user cannot scroll, which
        # let a deliberately broken panel pass this test on 20 Oct.)
        b = body.bounding_box()
        page.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
        page.mouse.wheel(0, 600)
        page.wait_for_timeout(300)
        assert body.evaluate("el => el.scrollTop") > 0, "drawer body does not scroll"
        assert drawer.locator(".drawer-head").bounding_box()["y"] >= 0, "header scrolled away"

        drawer.locator('[aria-label="Next segment"]').click()
        assert f"segment 2 of {len(segs)}" in drawer.locator(".drawer-kicker").inner_text().lower()
        page.keyboard.press("ArrowLeft")
        assert f"segment 1 of {len(segs)}" in drawer.locator(".drawer-kicker").inner_text().lower()

        page.keyboard.press("Escape")
        assert page.locator(".drawer").count() == 0
        assert not page.evaluate("document.documentElement.classList.contains('drawer-open')")

    def test_segment_formula_reproduces_the_segment_score(self, page, site):
        _open(page, REDUNDANCY_VIDEO)
        seg = next(s for s in _result(site, REDUNDANCY_VIDEO)["segments"] if s["score"])
        page.locator(f'.chart-row[data-segment="{seg["index"]}"]').click()
        formula = page.locator(".drawer .formula").first.inner_text()
        assert formula.strip().endswith(f"= {seg['score']:.2f}")


class TestCoherenceTab:
    def test_tab_shows_coherence_and_its_drawer(self, page, site):
        _open(page, COHERENCE_VIDEO)
        coh = _result(site, COHERENCE_VIDEO)["coherence"]
        page.locator('[data-principle="coherence"]').click()
        assert page.locator(".ring-value strong").inner_text() == f"{coh['result']['score']:.1f}"
        assert page.locator(".lanes").count() == 1
        page.locator('.chart-row[data-principle-of="coherence"]').first.click()
        assert page.locator(".drawer").get_attribute("data-principle") == "coherence"


class TestPhoneWidth:
    def test_drawer_fits_and_page_never_scrolls_sideways(self, page):
        page.set_viewport_size({"width": 375, "height": 812})
        _open(page, REDUNDANCY_VIDEO)
        page.locator('.chart-row[data-segment="0"]').click()
        box = page.locator(".drawer").bounding_box()
        assert box["width"] <= 375 + 1
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
