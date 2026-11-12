"""Session and settings state of the dashboard: the options remembered between sessions (per site folder), the
per-session caches, and small helpers about the result on screen."""
from pathlib import Path

import streamlit as st

from groundwater import hydro
from groundwater.pipeline import load_results


# The src/ folder: the program runs from here without being installed.
_SRC_ROOT = Path(__file__).resolve().parents[3]


def _reg_crs(result):
    """The result's map CRS with its height datum (heights.reg_crs): '+KNGeoid18' for orthometric results."""
    from groundwater import heights

    return heights.reg_crs(result.config)


@st.cache_resource(max_entries=2, show_spinner=False)
def _load_results_cached(path: str, mtime: float):
    """A saved result, shared by all browser tabs / reloads until the file changes (it is ~0.5 GB)."""
    return load_results(path)


def _result_key():
    """Identity of the result on screen (path + file time, or the run time), for caches."""
    return st.session_state.get("result_key", ("none",))


def _flow_sets(result):
    """hydro.joint_sets_for_flow for the result on screen (a copy: callers may add columns), made once."""
    return _session_cache("_flow_sets", _result_key(), lambda: hydro.joint_sets_for_flow(result)).copy()


def _session_cache(name: str, key, make):
    """Value cached in the session under `name` while `key` stays the same."""
    c = st.session_state.get(name)
    if c is not None and c[0] == key:
        return c[1]
    v = make()
    st.session_state[name] = (key, v)
    return v


def _fragment(fn):
    """st.fragment: a widget inside reruns only this part of the page (a 3D view with its own controls), not the
    whole app -- the map and the other open 3D views are not rebuilt and sent again. The remembered options are
    saved at the end of such a partial run too (the full run saves them in main)."""
    import functools

    @functools.wraps(fn)
    def body(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            if st.session_state.get("_input_mode") == "현장 폴더":
                _store_settings()

    return st.fragment(body)


def _range_state(key, lo, hi, default):
    """A remembered (lo, hi) range slider state, reset when the saved value is not a valid range."""
    _remember(key, site=True)                  # a chainage range belongs to the site
    v = st.session_state.get(key, _saved(key, None))
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        v = default
    st.session_state[key] = tuple(float(min(max(x, lo), hi)) for x in v)
    return key


DEMO_FOLDER = str(_SRC_ROOT.parent / "data" / "raw" / "demo_site")
NONE = "(사용 안 함)"

# Settings remembered between sessions: the last site folder and, per folder, the dashboard options.
# An option is remembered by drawing its widget with _w(key, default): there is no list of keys to keep. GENERAL
# options carry over to a folder opened for the first time; SITE options (site=True: files, analysis limits,
# planar / mesh data) belong to one site only. A widget that sets its own state calls _remember(key) instead.
SETTINGS_FILE = Path.home() / ".groundwater" / "dashboard_settings.json"
_PLANAR_WIDGETS = ("adv_planar", "planar_dir_input", "planar_redetect_dir")
_LINKED_WIDGETS = {"planar_dir": list(_PLANAR_WIDGETS), "mesh_path": ["mesh_path_input"]}
_RESULT_STATE = ("result", "result_key", "output_dir", "loaded_from", "_figs", "_seep", "_seep_key", "_seep_default",
                 "_g3d", "_g3d_key", "_g3d_raw", "_joints_cache", "_ref_cache", "_sec_cache", "_3d_sample", "_3d_fig",
                 "_facade_disp", "_sm_disp", "_sm_sample", "_sm_cloudfig", "_sm_joints", "_sm_isample", "_sm_surface", "_sm_dense", "_sm_major",
                 "_sm_ifig", "_flow_sets", "_fam_sets",
                 "_kin_stereo", "_kin_map", "_kin_3dfig", "_kin_domains", "_th_imgs", "_th_folders", "_pop_thermal", "_ta", "_ta_smap", "_seep_ortho", "_seep_facade")


def _clean_path(text) -> str:
    """A path as typed or pasted: without the spaces and the quotes Explorer's 'copy as path' puts around it."""
    return str(text or "").strip().strip('"')


def _site_dir() -> Path:
    """The site folder of the sidebar."""
    return Path(_clean_path(st.session_state.get("site_folder", "")))


def _settings_file() -> dict:
    import json

    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _folder_id(folder) -> str:
    try:
        return str(Path(_clean_path(folder)).resolve())
    except OSError:
        return str(folder)


def _saved(key, default):
    """Remembered value of an option for the current folder (else the last general value, else `default`)."""
    return st.session_state.get("_saved_values", {}).get(key, default)


def _remember(key, site: bool = False):
    """Mark the option `key` (a Session State key) as remembered between sessions: for every site (general), or
    with site=True for this site folder only. The session keeps the list, so it holds exactly the options this
    session has drawn; the others keep the value they have in the settings file."""
    st.session_state.setdefault("_remembered", {})[key] = bool(site)


def _w(key, default, site: bool = False):
    """Keyword args for a remembered widget: its state starts from the saved value (see _remember for `site`).

    The value is assigned again in every run, just before the widget: a value put into Session State in an earlier
    run, before the widget existed (a restored setting in a view opened later), is kept by the server but the
    widget shows its own default instead (Streamlit 1.62)."""
    _remember(key, site)
    st.session_state[key] = st.session_state.get(key, _saved(key, default))
    return {"key": key}


def _w_clamped(key, default, lo, hi, site: bool = False):
    """_w for a slider whose limits come from the data: the remembered value is brought inside them."""
    _remember(key, site)
    st.session_state[key] = float(min(max(st.session_state.get(key, _saved(key, default)), lo), hi))
    return {"key": key}


def _restore_session():
    """Once per browser session: reopen the last site folder."""
    if st.session_state.get("_settings_restored"):
        return
    st.session_state["_settings_restored"] = True
    last = _settings_file().get("last_folder")
    if last and Path(last).is_dir() and "site_folder" not in st.session_state:
        st.session_state["site_folder"] = last


def _site_key(folder, section=None) -> str:
    """Settings key of a site folder, or of one section of it (each section keeps its own options)."""
    return _folder_id(folder) + (f"::{section}" if section else "")


def _apply_folder_settings(folder, section=None):
    """When the site folder or section changes (and on start), load its options into the widget states."""
    fid = _site_key(folder, section)
    prev = st.session_state.get("_settings_folder")
    if prev == fid:
        return
    if prev is not None:                       # another site: the result on screen belongs to the old one
        for k in _RESULT_STATE:
            st.session_state.pop(k, None)
    st.session_state["_settings_folder"] = fid
    data = _settings_file()
    values = {**(data.get("general") or {}), **(data.get("folders", {}).get(fid) or {})}
    # the options of the folder shown until now: those drawn in this session and those loaded from its settings
    for k in set(st.session_state.get("_remembered", {})) | set(st.session_state.get("_saved_values", {})):
        st.session_state.pop(k, None)
    for linked in sum(_LINKED_WIDGETS.values(), []):
        st.session_state.pop(linked, None)
    st.session_state["_saved_values"] = values
    for k, v in values.items():                # views also read each other's options straight from Session State
        st.session_state[k] = v


def _store_settings():
    """Write the current options, merged with the saved ones (widgets not drawn this run keep their value)."""
    import json

    folder = st.session_state.get("site_folder")
    if not folder or not Path(_clean_path(folder)).is_dir():
        return
    fid = _site_key(folder, st.session_state.get("_section"))
    general, site = {}, {}
    for k, of_site in st.session_state.get("_remembered", {}).items():
        if k in st.session_state:
            v = st.session_state[k]
            (site if of_site else general)[k] = list(v) if isinstance(v, tuple) else v
    st.session_state["_saved_values"] = {**st.session_state.get("_saved_values", {}), **general, **site}
    data = _settings_file()
    folders = data.get("folders") or {}
    last_sec = {**(data.get("last_section") or {}), _folder_id(folder): st.session_state.get("_section")}
    new = {**data, "last_folder": _folder_id(folder), "last_section": last_sec,
           "general": {**(data.get("general") or {}), **general},
           "folders": {**folders, fid: {**(folders.get(fid) or {}), **site}}}
    if new != data:
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass


def _reference_folder(result):
    """Planar folder that holds reference_overlays.json (site-specific reference images), if any."""
    from groundwater.facade import reference_overlays as ro

    for d in ((result.config.get("planar") or {}).get("folder"), st.session_state.get("planar_dir")):
        if d and Path(str(d)).is_dir() and (Path(str(d)) / ro.OVERLAY_FILE).exists():
            return str(d)
    return None


def _chainage_range(result):
    d = result.facade_depth
    u0 = float(d.transform.c)
    return u0, u0 + d.array.shape[2] * abs(d.transform.a)
