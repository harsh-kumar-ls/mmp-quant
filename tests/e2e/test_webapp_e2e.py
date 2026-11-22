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
import re
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

RUNS = Path("runs")
REDUNDANCY_VIDEO, COHERENCE_VIDEO = "Test_16", "test6"
TIMING_VIDEO = "animation_london_subtitled"  # the user's 21 Oct timing case

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
    for v in (REDUNDANCY_VIDEO, COHERENCE_VIDEO, TIMING_VIDEO):
        if (RUNS / v / "result.json").exists():
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
        pg.goto(site["url"] + "/classic/")   # the original UI (10 Nov 2026: / is the React UI)
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


class TestTiming:
    """21 Oct 2026 (user: "catching the narration timestamp and the on-screen
    text with timestamp is the most crucial part of our project")."""

    def _window(self, page, site, start):
        if not (site["runs"] / TIMING_VIDEO / "result.json").exists():
            pytest.skip(f"runs/{TIMING_VIDEO} not available")
        windows = _result(site, TIMING_VIDEO)["result"]["windows"]
        w = next(x for x in windows if x["start"] == start)
        checks = (w.get("evidence") or {}).get("sentence_checks") or []
        if not any("time_weight" in c for c in checks):
            pytest.fail(f"runs/{TIMING_VIDEO} predates the time-weighted score: re-score it")
        _open(page, TIMING_VIDEO)
        page.locator('.chart-row[data-segment="0"]').click()
        win = page.locator("details.win").filter(
            has=page.locator(".win-time", has_text=re.compile(rf"^0:{int(start):02d}")))
        win.first.evaluate("d => d.open = true")
        return w, win.first

    def test_words_spoken_outside_the_window_are_shown_faded(self, page, site):
        _, win = self._window(page, site, 0.0)
        voice = win.locator(".evidence-col.voice")
        # "It's 135 metres" is spoken at 0:11 -- in the margin, not in 0:00-0:10
        assert voice.locator(".w-outside", has_text="135").count() == 1
        assert voice.locator(".w-in", has_text="135").count() == 0
        assert voice.locator(".tstamp").count() >= 1

    def test_each_sentence_counts_for_its_time_on_screen(self, page, site):
        w, win = self._window(page, site, 10.0)
        big = win.locator("tr", has_text="The big wheel").last
        high = win.locator("tr", has_text="135m high").last
        assert "1.5 s in this window" in big.inner_text()
        assert "5.5 s in this window" in high.inner_text()
        formula = win.locator(".formula").inner_text()
        assert formula.strip().endswith(f"= {w['score']:.2f}")


class TestRatingSet:
    """10 Nov 2026: the tab listing the 10 videos chosen for the professor."""

    def test_lists_ten_videos_warns_and_opens_a_report(self, page, site):
        import glob

        if not glob.glob("rating_package*/selection.csv"):
            pytest.skip("no rating package in this checkout (it is gitignored)")
        page.locator("#nav-rating").click()
        page.locator(".rs-row").first.wait_for()
        assert page.locator(".rs-row").count() == 10
        assert "Student only" in page.locator(".banner").first.inner_text()
        targets = page.locator(".rs-target").all_inner_texts()
        assert targets == [str(t) for t in range(0, 100, 10)]


def test_root_serves_the_react_ui_and_classic_stays_available(site):
    """10 Nov 2026: / is the React UI when it is built; the original is at /classic/."""
    import urllib.request

    from mmpquant.webapp import server
    root = urllib.request.urlopen(site["url"] + "/").read().decode()
    classic = urllib.request.urlopen(site["url"] + "/classic/").read().decode()
    assert 'id="video-list"' in classic and 'src="app.js"' in classic
    if server.REACT_DIST.is_dir():
        assert 'id="root"' in root and "/react/assets/" in root
    else:
        assert root == classic


class TestReactUI:
    """The React UI (10 Nov 2026, webapp-react/), served built at /react/ by
    the same server. Same checks as the original UI: numbers on screen equal
    result.json, the drawer's working adds up, timing is shown."""

    @pytest.fixture
    def react(self, page, site):
        from mmpquant.webapp import server
        if not server.REACT_DIST.is_dir():
            pytest.skip("webapp-react/dist not built (npm run build)")
        page.goto(site["url"] + "/react/")
        return page

    def _open(self, page, vid):
        page.get_by_test_id("video-list").get_by_text(vid, exact=True).click()
        page.get_by_test_id("ring-value").wait_for()

    def test_header_and_segments_match_result_json(self, react, site):
        self._open(react, REDUNDANCY_VIDEO)
        res = _result(site, REDUNDANCY_VIDEO)
        assert react.get_by_test_id("ring-value").inner_text() == f"{res['result']['score']:.1f}"
        shown = react.get_by_test_id("segment-value").all_inner_texts()
        assert shown == [f"{s['score']:.2f}" for s in res["segments"]]

    def test_report_says_where_it_was_scored(self, react, site):
        # 12 Nov 2026: the cloud Space scores uploads with RapidOCR; Mac runs are
        # the thesis numbers -- every report says which
        self._open(react, REDUNDANCY_VIDEO)
        backend = _result(site, REDUNDANCY_VIDEO)["config"]["ocr"]["backend"]
        label = react.get_by_test_id("scored-with").inner_text()
        assert ("on the Mac" in label) == (backend == "apple")

    def test_instructor_score_is_100_minus_the_mean_of_both(self, react, site):
        # 12 Nov 2026: 100 - (R + C) / 2, computed from result.json's two scores
        self._open(react, COHERENCE_VIDEO)
        res = _result(site, COHERENCE_VIDEO)
        r, c = res["result"]["score"], res["coherence"]["result"]["score"]
        expected = round(100 - (r + c) / 2, 2)
        assert react.get_by_test_id("instructor-value").inner_text() == f"{expected:.1f}"

    def test_drawer_formula_navigation_and_close(self, react, site):
        self._open(react, REDUNDANCY_VIDEO)
        segs = _result(site, REDUNDANCY_VIDEO)["segments"]
        seg = next(i for i, s in enumerate(segs) if s["score"])
        react.get_by_test_id("segment-row").nth(seg).click()
        drawer = react.get_by_test_id("drawer")
        assert drawer.get_by_test_id("drawer-score").inner_text() == f"{segs[seg]['score']:.2f}"
        assert drawer.get_by_test_id("segment-formula").inner_text().strip().endswith(f"= {segs[seg]['score']:.2f}")
        body = drawer.get_by_test_id("drawer-body")
        b = body.bounding_box()
        react.mouse.move(b["x"] + b["width"] / 2, b["y"] + b["height"] / 2)
        react.mouse.wheel(0, 600)
        react.wait_for_timeout(300)
        assert body.evaluate("el => el.scrollTop") > 0, "drawer body does not scroll"
        if seg + 1 < len(segs):
            drawer.get_by_label("Next segment").click()
            assert f"segment {seg + 2} of {len(segs)}" in drawer.get_by_test_id("drawer-kicker").inner_text().lower()
        react.keyboard.press("Escape")
        assert react.get_by_test_id("drawer").count() == 0

    def test_timing_margin_and_window_formula(self, react, site):
        if not (site["runs"] / TIMING_VIDEO / "result.json").exists():
            pytest.skip(f"runs/{TIMING_VIDEO} not available")
        w = next(x for x in _result(site, TIMING_VIDEO)["result"]["windows"] if x["start"] == 10.0)
        self._open(react, TIMING_VIDEO)
        react.get_by_test_id("segment-row").first.click()
        wins = react.get_by_test_id("win")
        first = wins.filter(has=react.locator(".win-time", has_text=re.compile(r"^0:00")))
        assert first.locator(".w-outside", has_text="135").count() == 1
        second = wins.filter(has=react.locator(".win-time", has_text=re.compile(r"^0:10"))).first
        second.evaluate("d => d.open = true")
        assert "1.5 s in this window" in second.locator("tr", has_text="The big wheel").last.inner_text()
        assert second.get_by_test_id("formula").inner_text().strip().endswith(f"= {w['score']:.2f}")

    def test_coherence_tab_and_drawer(self, react, site):
        self._open(react, COHERENCE_VIDEO)
        coh = _result(site, COHERENCE_VIDEO)["coherence"]
        react.get_by_test_id("tab-coherence").click()
        assert react.get_by_test_id("ring-value").inner_text() == f"{coh['result']['score']:.1f}"
        react.locator('[data-principle-of="coherence"]').first.click()
        assert react.get_by_test_id("drawer").get_attribute("data-principle") == "coherence"

    def test_rating_set(self, react):
        import glob

        if not glob.glob("rating_package*/selection.csv"):
            pytest.skip("no rating package in this checkout (it is gitignored)")
        react.get_by_test_id("nav-rating").click()
        react.get_by_test_id("rs-row").first.wait_for()
        assert react.get_by_test_id("rs-target").all_inner_texts() == [str(t) for t in range(0, 100, 10)]
        assert "Student only" in react.get_by_role("note").first.inner_text()

    def test_phone_width_never_scrolls_sideways(self, page, site):
        from mmpquant.webapp import server
        if not server.REACT_DIST.is_dir():
            pytest.skip("webapp-react/dist not built")
        page.set_viewport_size({"width": 375, "height": 812})
        page.goto(site["url"] + "/react/")
        self._open(page, REDUNDANCY_VIDEO)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")


class TestReadOnlyCopy:
    """The read-only copy (13 Nov 2026, tools/build_static_site.py): the same UI
    reading pre-built JSON, no server. Built fresh here so it cannot be stale."""

    @pytest.fixture(scope="class")
    def static_site(self, tmp_path_factory):
        import functools
        import http.server
        import subprocess
        import sys

        if not (RUNS / REDUNDANCY_VIDEO / "result.json").exists():
            pytest.skip("runs not available")
        if shutil.which("npx") is None:
            pytest.skip("Node not installed")
        out = tmp_path_factory.mktemp("static-site")
        subprocess.run([sys.executable, "tools/build_static_site.py", str(out)], check=True,
                       capture_output=True)
        type(self)._out = str(out)
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", _free_port()), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{srv.server_address[1]}/"
        srv.shutdown()

    def test_reports_match_result_json_with_no_upload_or_rating_set(self, page, static_site):
        page.goto(static_site)
        page.get_by_test_id("video-list").get_by_text(REDUNDANCY_VIDEO, exact=True).click()
        page.get_by_test_id("ring-value").wait_for()
        res = json.loads((RUNS / REDUNDANCY_VIDEO / "result.json").read_text())
        assert page.get_by_test_id("ring-value").inner_text() == f"{res['result']['score']:.1f}"
        page.get_by_text("the live site for marking your own video is offline").wait_for()
        assert page.get_by_test_id("live-link").count() == 0      # no live.json: offline
        assert page.get_by_test_id("nav-rating").count() == 0
        assert page.get_by_label("Mark a video (upload)").count() == 0

    def test_button_follows_the_published_live_link(self, page, static_site, site):
        # 13 Nov 2026: share_site.sh publishes live.json; the button opens that link
        # only while it answers (the live test server stands in for the Mac)
        live = Path(self._out) / "live.json"
        try:
            live.write_text(json.dumps({"url": site["url"]}))
            page.goto(static_site)
            link = page.get_by_test_id("live-link")
            link.wait_for()
            assert link.get_attribute("href") == site["url"]
            assert link.get_attribute("target") == "_blank"
            live.write_text(json.dumps({"url": None}))      # share_site.sh stopped
            page.goto(static_site)
            page.get_by_text("the live site for marking your own video is offline").wait_for()
            assert page.get_by_test_id("live-link").count() == 0
        finally:
            live.write_text(json.dumps({"url": None}))
