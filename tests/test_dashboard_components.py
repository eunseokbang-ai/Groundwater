"""The dashboard's shared building blocks (viz/dashboard/components.py): the 'this view on the tiled model' button,
tables with labelled columns, the choice control and the explanation behind a button."""
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


def test_table_labels_and_units(monkeypatch):
    seen = {}
    monkeypatch.setattr(components.st, "dataframe", lambda df, **kw: seen.update(kw, df=df))
    rows = [{"set_id": 1, "mean_dip": 87.04, "est_dip": None, "trace_only": True, "flow_share": 0.4, "mine": 2.5,
             "unknown": "x"}]
    components._table(rows, {"mine": ("내 값 (m)", "%.1f", "설명")})
    cfg = seen["column_config"]
    assert cfg["set_id"]["label"] == "절리군"
    assert cfg["mean_dip"]["label"] == "평균 경사 (°)" and cfg["mean_dip"]["type_config"]["format"] == "%.1f"
    assert cfg["est_dip"]["label"] == "추정 경사 (°)" and cfg["est_dip"]["help"]      # no numbers: label and help only
    assert "type_config" not in cfg["est_dip"] and "type_config" not in cfg["trace_only"]
    assert cfg["flow_share"]["type_config"]["type"] == "progress"                       # a ready column, as it is
    assert cfg["mine"]["label"] == "내 값 (m)" and cfg["mine"]["help"] == "설명"
    assert "unknown" not in cfg                                                         # shown under its own name
    assert seen["hide_index"] is True and seen["width"] == "stretch"
    assert list(seen["df"].columns) == list(rows[0])                                    # the data keep their names


def _choice_page():
    import streamlit as st

    from groundwater.viz.dashboard import components

    st.session_state["picked"] = components._choice("표시", ["점군", "표면"], key="shown")
    st.session_state["free"] = components._choice("바탕", ["정사영상", "DSM"], default="DSM")
    components._note("한 줄 설명", "긴 설명")


def test_choice_always_has_one_option():
    app = AppTest.from_function(_choice_page)
    app.session_state["shown"] = "지금은 없는 값"                    # e.g. remembered from an older version
    app.run()
    assert not app.exception
    assert app.session_state["picked"] == "점군" and app.session_state["free"] == "DSM"
    app.segmented_control(key="shown").set_value("표면").run()
    assert app.session_state["picked"] == "표면"
    assert [c.value for c in app.caption] == ["한 줄 설명"]          # the note: one line, the rest behind the button
    assert any(m.value == "긴 설명" for m in app.markdown)
