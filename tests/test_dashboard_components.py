"""The dashboard's shared 'this view on the tiled model' button (viz/dashboard/components.py)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from groundwater.viz import tiles3d  # noqa: E402
from groundwater.viz.dashboard import components  # noqa: E402


def _page():
    import streamlit as st

    from groundwater.viz.dashboard import components

    components._tiles_scene_button("RESULT", st.session_state.get("tiles"), "kin", "kinematic_scene_tiles.json",
                                   "제목", lambda: [{"name": "view image"}, None], sets="SETS",
                                   joints=lambda: "JOINTS", smap="SMAP")


def test_tiles_scene_button(tmp_path, monkeypatch):
    calls = {}

    def scene_json(*args, **kwargs):
        calls["scene"] = (args, kwargs)
        return {"overlays": [{"name": "reference image"}]}

    def write_scene(path, scene):
        calls["written"] = (Path(path), scene)
        return Path(path)

    monkeypatch.setattr(tiles3d, "scene_json", scene_json)
    monkeypatch.setattr(tiles3d, "write_scene", write_scene)
    monkeypatch.setattr(components, "_reference_folder", lambda result: "REF")
    monkeypatch.setattr(components, "_open_tiles_viewer", lambda path: calls.setdefault("opened", path) and "http://viewer")

    app = AppTest.from_function(_page)
    app.session_state["output_dir"] = str(tmp_path)
    app.run()
    assert app.button(key="kin_tiles_open").disabled            # no tiled model yet
    app.session_state["tiles"] = "TILES"
    app.run()
    app.button(key="kin_tiles_open").click().run()
    assert not app.exception
    args, kwargs = calls["scene"]
    assert args == ("RESULT", [], "SETS", "TILES", "JOINTS", None)
    assert kwargs == {"title": "제목", "joint_shade": "none", "smap": "SMAP", "overlay_folder": "REF"}
    path, scene = calls["written"]
    assert path == tmp_path / "kinematic_scene_tiles.json"
    assert [o["name"] for o in scene["overlays"]] == ["view image", "reference image"]   # the view's own image first
    assert calls["opened"] == path
    assert any("http://viewer" in m.value for m in app.markdown)
