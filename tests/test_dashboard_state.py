"""The dashboard's remembered options (viz/dashboard/state.py): an option is remembered by drawing its widget with
_w(key, default) -- general ones for every site, site ones per site folder."""
import json

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from groundwater.viz.dashboard import state  # noqa: E402


def _page():
    """A page with one general and one site option, as the views draw them."""
    import streamlit as st

    from groundwater.viz.dashboard import state

    state._apply_folder_settings(st.session_state["site_folder"])
    st.slider("general", 0, 10, **state._w("opt_general", 3))
    st.slider("site", 0, 10, **state._w("opt_site", 4, site=True))
    state._store_settings()


def _open(folder) -> AppTest:
    app = AppTest.from_function(_page)
    app.session_state["site_folder"] = str(folder)
    return app.run()


def _values(app) -> tuple:
    assert not app.exception
    return app.slider(key="opt_general").value, app.slider(key="opt_site").value


@pytest.fixture
def settings(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    monkeypatch.setattr(state, "SETTINGS_FILE", f)
    return f


def test_options_remembered_by_scope(tmp_path, settings):
    a, b = tmp_path / "site_a", tmp_path / "site_b"
    a.mkdir()
    b.mkdir()
    app = _open(a)
    assert _values(app) == (3, 4)                                  # the defaults
    app.slider(key="opt_general").set_value(7)
    app.slider(key="opt_site").set_value(9)
    app.run()
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["general"] == {"opt_general": 7}
    assert data["folders"] == {state._site_key(a): {"opt_site": 9}}
    assert data["last_folder"] == state._folder_id(a)

    again = _open(a)                                               # a new session on the same folder
    assert _values(again) == (7, 9)
    again.session_state["site_folder"] = str(b)                    # another site: the general option carries over
    assert _values(again.run()) == (7, 4)
    again.slider(key="opt_site").set_value(1).run()
    again.session_state["site_folder"] = str(a)                    # and back: each site kept its own
    assert _values(again.run()) == (7, 9)
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["folders"][state._site_key(b)] == {"opt_site": 1}


def test_options_not_drawn_keep_their_saved_value(tmp_path, settings):
    """Options of views this session never opened are not in its list: they must stay in the file as they were."""
    a = tmp_path / "site_a"
    a.mkdir()
    fid = state._site_key(a)
    settings.write_text(json.dumps({"general": {"other_view": "x", "opt_general": 5},
                                    "folders": {fid: {"other_range": [1.0, 2.0]}, "elsewhere": {"opt_site": 8}}}),
                        encoding="utf-8")
    app = _open(a)
    assert _values(app) == (5, 4)
    assert app.session_state["other_view"] == "x"                  # views read each other's options from the session
    app.slider(key="opt_site").set_value(6).run()
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["general"] == {"other_view": "x", "opt_general": 5}
    assert data["folders"] == {fid: {"other_range": [1.0, 2.0], "opt_site": 6}, "elsewhere": {"opt_site": 8}}


def test_clean_path():
    assert state._clean_path('  "C:\\data\\site A" ') == "C:\\data\\site A"
    assert state._clean_path(None) == ""
