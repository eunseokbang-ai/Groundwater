"""The run record of the dashboard (viz/dashboard/run.py): what happens to a finished or failed analysis."""
import pytest

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from groundwater.viz.dashboard import run  # noqa: E402


class _Result:
    def __init__(self):
        self.timings = {"load": 1.0, "kinematics": 0.5}


def _page():
    import streamlit as st

    from groundwater.viz.dashboard import run

    if st.session_state.pop("go", False):
        run._run_analysis({"paths": {"output_dir": st.session_state["out"]}}, {})
    st.session_state["running"] = run._take_run(st.session_state["out"])
    run._unsaved_banner()
    st.session_state["on_screen"] = st.session_state.get("result") is not None


@pytest.fixture
def app(tmp_path):
    st.cache_resource.clear()                  # the run records of the process
    a = AppTest.from_function(_page)
    a.session_state["out"] = str(tmp_path / "out")
    yield a
    st.cache_resource.clear()


def _button(app, label):
    return next(b for b in app.button if b.label == label)


def test_failed_save_keeps_the_result(app, monkeypatch):
    def locked(result, out_dir):
        raise PermissionError(13, "Permission denied", "discontinuity_sets.csv")

    monkeypatch.setattr(run, "run_pipeline", lambda cfg, progress=None: _Result())
    monkeypatch.setattr(run, "save_results", locked)
    app.session_state["go"] = True
    app.run()
    assert not app.exception
    assert app.session_state["on_screen"]                            # the analysis is not thrown away
    assert any("저장하지 못했습니다" in e.value and "Permission denied" in e.value for e in app.error)
    saved = []
    monkeypatch.setattr(run, "save_results", lambda result, out_dir: saved.append(out_dir))
    _button(app, "다시 저장").click().run()
    assert saved == [app.session_state["out"]] and not app.error
    assert any("저장했습니다" in s.value for s in app.success)
    assert not run._jobs()                                           # the record is closed


def test_failed_analysis_stays_on_the_page_until_closed(app, monkeypatch):
    def broken(cfg, progress=None):
        progress("joints", "normals", 2.0)
        raise ValueError("no points left")

    monkeypatch.setattr(run, "run_pipeline", broken)
    app.session_state["go"] = True
    app.run()
    assert not app.exception and not app.session_state["on_screen"]
    assert any("절리면·절리군" in e.value for e in app.error)          # the stage it stopped in
    app.run()                                                        # any click: the reason is still there
    assert app.error and "no points left" in app.code[0].value
    _button(app, "닫기").click().run()
    assert not app.error


def test_page_opened_again_finds_the_running_analysis(app, tmp_path):
    """A reloaded page is a new session: it must see the run on its folder, not offer to start a second one."""
    import time

    job = {"state": "running", "stage": "normals", "since": time.time(), "started": time.time(),
           "out_dir": app.session_state["out"], "folder": None}
    run._jobs()[run._folder_id(job["out_dir"])] = job
    app.run()
    assert not app.exception and app.session_state["running"]
    assert any("분석이 진행 중입니다" in m.value for m in app.markdown)
    assert run._other_run(str(tmp_path / "another")) is job          # and no second run on another folder
    assert run._other_run(job["out_dir"]) is None
    job.update(state="done", result=_Result(), finished=time.time())
    app.run()
    assert app.session_state["on_screen"] and not app.session_state["running"]
