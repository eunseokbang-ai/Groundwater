"""Smoke test of the dashboard: open the demo site, run the analysis, open every view, reload the saved result.

It only checks that every screen draws without an error -- enough to notice a name lost or an import broken when
dashboard code is moved between modules. The app runs in this process (Streamlit's AppTest) with the home and
temp folders pointed at tmp_path, so the user's dashboard settings and popup pages are not touched."""
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "groundwater" / "viz" / "dashboard" / "app.py"
DEMO = ROOT / "data" / "raw" / "demo_site"
RESULT_DIR = "분석결과"


def _program_modules() -> dict:
    return {k: v for k, v in sys.modules.items() if k == "groundwater" or k.startswith("groundwater.")}


@pytest.fixture
def demo_dashboard(tmp_path, monkeypatch):
    """The dashboard on a copy of the demo site (inputs only), as (app, site folder)."""
    if not DEMO.is_dir():
        pytest.skip("demo site data not present")
    home, tmp = tmp_path / "home", tmp_path / "tmp"
    home.mkdir()
    tmp.mkdir()
    for var in ("USERPROFILE", "HOME"):
        monkeypatch.setenv(var, str(home))
    for var in ("TEMP", "TMP"):
        monkeypatch.setenv(var, str(tmp))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    site = tmp_path / "demo_site"
    shutil.copytree(DEMO, site, ignore=shutil.ignore_patterns(RESULT_DIR))
    # the app reloads the program modules on its first run in a process: give the other tests theirs back afterwards
    before = _program_modules()
    st.cache_resource.clear()
    app = AppTest.from_file(str(APP), default_timeout=600)
    app.session_state["site_folder"] = str(site)
    try:
        yield app, site
    finally:
        for name in _program_modules():
            del sys.modules[name]
        sys.modules.update(before)
        st.cache_resource.clear()


def _button(app, label: str):
    found = [b for b in app.button if b.label.startswith(label)]
    assert found, f"no button '{label}' among {[b.label for b in app.button]}"
    return found[0]


def _ok(app, what: str):
    assert not app.exception, f"{what}: {[e.value for e in app.exception]}"
    assert not app.error, f"{what}: {[e.value for e in app.error]}"


def _open_every_view(app) -> list:
    """Open each view of each group in turn; returns the (group, view) pairs that were opened."""
    opened = []
    groups = list(app.segmented_control(key="nav_group").options)
    for group in groups:
        app.segmented_control(key="nav_group").set_value(group).run()
        _ok(app, group)
        key = f"nav_view_{group}"
        for view in list(app.pills(key=key).options):
            app.pills(key=key).set_value(view).run()
            _ok(app, f"{group} / {view}")
            opened.append((group, view))
    return opened


def test_demo_site_run_and_every_view(demo_dashboard):
    app, site = demo_dashboard
    app.run()
    _ok(app, "first page")
    assert not app.metric, "no result yet: the header numbers must not be shown"

    _button(app, "분석 실행").click().run()            # no earlier result: the dialog starts the run at once
    _ok(app, "analysis run")
    assert (site / RESULT_DIR / "result.pkl").exists()
    assert len(app.metric) == 5                       # the result's header
    opened = _open_every_view(app)
    assert len({g for g, _ in opened}) == 5 and len(opened) >= 15

    # a new browser session on the same folder: the saved result, loaded without running again
    again = AppTest.from_file(str(APP), default_timeout=600)
    again.session_state["site_folder"] = str(site)
    again.run()
    _ok(again, "second session")
    _button(again, "이전 결과 보기").click().run()
    _ok(again, "load saved result")
    assert len(again.metric) == 5
    assert again.session_state["loaded_from"]
