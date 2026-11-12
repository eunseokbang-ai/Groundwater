"""Streamlit dashboard of the discontinuity-extraction pipeline and the views built on its results (seepage,
thermal, grouting, kinematics, LiDAR sections, site-wide joints).

Run with:
    streamlit run src/groundwater/viz/dashboard/app.py

This file only starts the program. The dashboard itself is in the modules beside it:
    page.py         title, sidebar, run / load, the result's header, the navigation between the views
    sidebar.py      the site-folder input and the advanced settings (-> analysis configuration)
    run.py          the analysis run (dialog, stages, the run record)
    state.py        remembered options (per site folder), session caches, helpers about the result on screen
    components.py   building blocks shared by the views (figure + popup window, 3D view, path fields, tools)
    views/          one module per view
"""
import os
import sys
from pathlib import Path

import streamlit as st

# Allow running via `streamlit run` without an editable install.
_SRC_ROOT = Path(__file__).resolve().parents[2].parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
# Streamlit reloads an edited module only if it lies in this script's folder or on PYTHONPATH: without this, a
# change to groundwater/viz/*.py (or any module outside dashboard/) is ignored until the server restarts, while
# app.py itself is reloaded -- the two then disagree (e.g. a call with a changed signature fails).
_pp = os.environ.get("PYTHONPATH", "")
if str(_SRC_ROOT) not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (str(_SRC_ROOT), _pp) if p)

_DASHBOARD = "groundwater.viz.dashboard"


@st.cache_resource(show_spinner=False)
def _module_times() -> dict:
    """File time of every program module as it was when loaded, kept for the life of the server process."""
    return {"new": True, "seen": {}}


def _program_modules(prefix: str = "groundwater") -> list:
    return [n for n in list(sys.modules) if n == prefix or n.startswith(prefix + ".")]


def _refresh_program_modules():
    """Drop the program modules (groundwater.*) from memory when any of their files changed since they were
    loaded, so this run imports them afresh. Streamlit's own watcher misses a change made while no browser
    session was watching that file (each session watches only from its own first run), and then keeps the old
    module until the server restarts -- app.py itself is always rerun, so the two disagree (a missing function,
    a changed signature). All of them go together: they import from each other.

    A change in the dashboard's own modules drops only those: nothing else imports them, and the rest (the classes
    of the result held in the session) stays as loaded. Streamlit's watcher drops just the one edited module, which
    would leave the other dashboard modules with the old functions they imported from it -- so a dashboard module
    gone from memory counts as changed too."""
    state = _module_times()
    everything = state["new"]                  # first run of this check in a server that may have run for hours
    state["new"] = False
    dashboard = False
    for name, t in list(state["seen"].items()):
        ours = name == _DASHBOARD or name.startswith(_DASHBOARD + ".")
        mod = sys.modules.get(name)
        if mod is None:                        # Streamlit's watcher already dropped it: it loads afresh anyway
            del state["seen"][name]
            dashboard = dashboard or ours
            continue
        f = getattr(mod, "__file__", None)
        try:
            if f and os.path.getmtime(f) != t:
                if not ours:
                    everything = True
                    break
                dashboard = True
        except OSError:
            pass
    if everything:
        for name in _program_modules():
            sys.modules.pop(name, None)
        state["seen"].clear()
    elif dashboard:
        for name in _program_modules(_DASHBOARD):
            sys.modules.pop(name, None)
            state["seen"].pop(name, None)


def _note_program_modules():
    """Record the file time of every program module loaded so far (end of each run)."""
    seen = _module_times()["seen"]
    for name in _program_modules():
        if name not in seen:
            f = getattr(sys.modules[name], "__file__", None)
            try:
                if f:
                    seen[name] = os.path.getmtime(f)
            except OSError:
                pass


_refresh_program_modules()

st.set_page_config(page_title="암벽 불연속면 분석", page_icon=":material/landscape:", layout="wide")

from groundwater.viz.dashboard import page  # noqa: E402


def main():
    try:
        page.main()
    finally:
        _note_program_modules()


if __name__ == "__main__":
    main()
