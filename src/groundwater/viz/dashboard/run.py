"""The analysis run: the dialog that asks and shows the stages, the run record that survives a click during the
run, and the stage times."""
from pathlib import Path

import pandas as pd
import streamlit as st

from groundwater.pipeline import RESULT_PICKLE, run_pipeline, save_results
from groundwater.viz.dashboard.state import _RESULT_STATE, _folder_id


STAGE_NAMES = {"load": "점군 읽기", "frame": "기준선·분석 범위", "auto_scale": "자동 설정 (점 간격)",
               "voxel": "식생 제외·복셀", "masks": "낙석방지망·평탄 지면 제외", "lidar": "라이다로 망·식생 뒤 암반 보강",
               "normals": "법선·곡률", "joints": "절리면·절리군",
               "facade": "입면 영상 (원본 전체 점)", "planar": "고해상도 평면 자료 (색·트레이스)",
               "cleaning": "식생 위의 면 제외", "kinematics": "안정성 검토", "save": "결과 저장"}


def _dur(seconds: float) -> str:
    """'36분 27초' / '14초'."""
    s = int(round(seconds))
    return f"{s // 60}분 {s % 60:02d}초" if s >= 60 else f"{s}초"


def _last_timings(output_dir) -> dict:
    """Seconds per stage of the run that made the result on screen, if that result belongs to `output_dir` (a guide
    for the next run of the same folder); else {}."""
    result = st.session_state.get("result")
    shown = st.session_state.get("output_dir")
    if result is None or not shown or _folder_id(shown) != _folder_id(output_dir):
        return {}
    return dict(getattr(result, "timings", None) or {})


def _run_notes(prev, out_dir) -> list:
    """What a new run replaces (asked about before it starts): the saved result of this folder, the result on
    screen; with the duration of the last run when known. [] when nothing is replaced."""
    import datetime

    notes = []
    if prev:
        when = datetime.datetime.fromtimestamp((Path(prev) / RESULT_PICKLE).stat().st_mtime)
        notes.append(f"이 폴더에는 **{when:%Y-%m-%d %H:%M}**에 저장된 분석 결과가 있습니다."
                     + (" 새 분석이 끝나면 그 결과를 **덮어씁니다** (남겨 두려면 먼저 분석결과 폴더의 이름을 바꾸세요)."
                        if _folder_id(prev) == _folder_id(out_dir) else ""))
    if st.session_state.get("result") is not None:
        notes.append("지금 화면의 결과는 닫힙니다.")
    total = sum(_last_timings(out_dir).values())
    if notes and total:
        notes.append(f"지난 분석은 **{_dur(total)}** 걸렸습니다.")
    return notes


def _run_go():
    st.session_state["_run_go"] = True


@st.dialog("분석 실행", icon=":material/play_circle:")
def _run_dialog(cfg: dict, notes: list, last: dict):
    """The analysis run, in a dialog: first the question when a result would be replaced (`notes`), then the
    stages as they go. A full wall takes over an hour, so the run is not started by one stray click, and the dialog
    covers the page while it runs. Everything sits in one slot: the question is replaced by the progress at once,
    not left greyed for the whole run."""
    slot = st.empty()
    if notes and not st.session_state.pop("_run_go", False):
        with slot.container():
            st.markdown("\n".join(f"- {n}" for n in notes))
            with st.container(horizontal=True, horizontal_alignment="right"):
                if st.button("취소", key="run_cancel"):
                    st.rerun()
                st.button("분석 실행", type="primary", icon=":material/play_arrow:", key="run_confirm", on_click=_run_go)
        return
    with slot.container():
        st.caption("분석이 끝날 때까지 이 창을 열어 두세요. 창을 닫고 다른 곳을 눌러도 분석은 계속되어 저장되고, "
                   "진행 상황은 화면 위쪽에 나옵니다.")
        shown = _run_analysis(cfg, last)
    if shown:
        st.rerun()                             # closes the dialog; _take_run puts the result on screen


def _run_analysis(cfg: dict, last: dict) -> bool:
    """Run the pipeline and save the result, with the stages listed as they end and the running one in the box's
    title (`last` = seconds per stage of the previous run of this folder, shown as a guide). The result is left in
    the session's run record (`_run_job`), from where the next script run takes it (_take_run).

    When a widget is touched during the run, Streamlit stops this script at its next st.* call or Session State
    access and starts a new one -- which would drop an analysis of up to an hour. So from that moment this run
    shows nothing more and touches no Session State: it only finishes, saves and fills the record it already holds.
    Returns whether this script is still the one on screen."""
    import datetime
    import time
    import traceback

    from streamlit.runtime.scriptrunner import RerunException, StopException

    out_dir = cfg["paths"].get("output_dir", "data/processed/output")
    job = {"state": "running", "stage": "load", "since": time.time(), "started": time.time(), "out_dir": out_dir,
           "folder": st.session_state.get("_settings_folder")}
    for k in _RESULT_STATE:
        st.session_state.pop(k, None)
    st.session_state["_run_job"] = job
    shown = [True]

    def tell(fn):
        if shown[0]:
            try:
                fn()
            except (RerunException, StopException):
                shown[0] = False

    box = st.status("분석을 시작하는 중...", expanded=True)

    def progress(stage, finished, seconds):
        job.update(stage=stage, since=time.time())
        if finished:
            tell(lambda: box.write(f":material/check: {STAGE_NAMES.get(finished, finished)} — {_dur(seconds)}"))
        usual = f" · 지난번 {_dur(last[stage])}" if last.get(stage) else ""
        tell(lambda: box.update(label=f"{STAGE_NAMES.get(stage, stage)} ({datetime.datetime.now():%H:%M} 시작{usual})",
                                expanded=True))             # a label update alone closes the box

    try:
        result = run_pipeline(cfg, progress=progress)
        progress("save", "kinematics", result.timings.get("kinematics", 0.0))
        t_save = time.perf_counter()
        save_results(result, out_dir)
        result.timings["save"] = round(time.perf_counter() - t_save, 1)
    except Exception:
        job.update(state="error", error=traceback.format_exc())
        tell(lambda: box.update(label="분석이 중단되었습니다", state="error", expanded=True))
        return shown[0]
    job.update(state="done", result=result, finished=time.time())
    tell(lambda: box.update(label=f"분석 완료 — 모두 {_dur(job['finished'] - job['started'])}", state="complete",
                            expanded=True))
    return shown[0]


def _take_run() -> bool:
    """The run record of this session (_run_analysis): a finished run's result is put on screen, a failed one's
    error shown, a running one announced (the page of a run that lost its display). Returns whether one is running."""
    import datetime

    job = st.session_state.get("_run_job")
    if not job:
        return False
    if job["state"] == "running":
        hm = lambda t: f"{datetime.datetime.fromtimestamp(t):%H:%M}"  # noqa: E731
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            st.markdown(f":material/progress_activity: **분석이 진행 중입니다** ({hm(job['started'])} 시작) — 지금 단계: "
                        f"{STAGE_NAMES.get(job['stage'], job['stage'])} ({hm(job['since'])}부터). 끝나면 결과가 저장되고, "
                        "'상태 확인'을 누르면 열립니다.")
            st.button("상태 확인", icon=":material/refresh:", key="run_poll")
        return True
    del st.session_state["_run_job"]
    if job["state"] == "error":
        st.error("분석이 오류로 중단되었습니다.")
        st.code(job.get("error", ""))
        return False
    if job["folder"] != st.session_state.get("_settings_folder"):       # another site or section was opened meanwhile
        st.success(f"분석이 끝나 저장되었습니다: {job['out_dir']} — 그 폴더를 열고 '이전 결과 보기'로 불러오세요.")
        return False
    for k in _RESULT_STATE:
        st.session_state.pop(k, None)
    st.session_state["result"] = job.pop("result")
    st.session_state["output_dir"] = job["out_dir"]
    st.session_state["result_key"] = ("run", job["out_dir"], job["finished"])
    st.session_state["loaded_from"] = None
    return False


def _run_times(result):
    """Seconds per analysis stage of the run that made this result (results saved before this have none)."""
    t = getattr(result, "timings", None) or {}
    if not t:
        return
    with st.sidebar.expander(f"분석 소요 시간 {sum(t.values()) / 60:.1f}분", icon=":material/timer:"):
        st.dataframe(pd.DataFrame({"단계": [STAGE_NAMES.get(k, k) for k in t], "초": list(t.values())}),
                     hide_index=True, width="stretch")
        n = (result.config.get("load_thinning") or {}).get("points_total")
        if n:
            st.caption(f"원본 점 {n:,}개. 입면 영상은 원본의 모든 점으로, 절리 분석은 얇게 읽은 점으로 만듭니다.")
