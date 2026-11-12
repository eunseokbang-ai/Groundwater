"""RGB and thermal orthomosaics of a wall on one chainage x elevation grid (facade.planar.wall_image), blended for
the dashboard and as a popup page with a live opacity slider, a left / right swipe, wheel zoom and drag pan.
"""
import base64
import io
import json

import numpy as np

from ..facade import planar
from ..facade import thermal as TH


def pair_images(result, rgb_folder, thermal_folder, crs: str, u_range=None, v_range=None, max_px: int = 8000) -> dict:
    """{"rgb", "thermal" (H, W, 3 uint8, top row highest, white = no data), "ext" [u0, u1, v0, v1], "res",
    "temperature" (H, W) degC or None, "t_range" (low, high) of its colours}. With a temperature mosaic the
    thermal image is the temperatures in the inferno map (2-98 % range), else the palette mosaic itself."""
    _, ortho_r = planar.find_planar_pair(rgb_folder)
    _, ortho_t = planar.find_planar_pair(thermal_folder)
    temp_path = TH.temperature_ortho(thermal_folder)
    Tr = np.array(planar.load_registration(rgb_folder, crs)["matrix"])
    reg_t = TH.load(thermal_folder, crs)
    Tt = np.array(reg_t["matrix"])
    rgb, ext, res = planar.wall_image(result, ortho_r, Tr, u_range, v_range, max_px)
    temp, t_range = None, None
    if temp_path is not None:
        temp, _, _ = planar.wall_image(result, temp_path, Tt, u_range, v_range, max_px, res=res,
                                       correction=reg_t.get("local_correction"), values=True)
        temp[(temp < -40) | (temp > 80)] = np.nan
        th, t_range = colour_temperature(temp)
    else:
        th, _, _ = planar.wall_image(result, ortho_t, Tt, u_range, v_range, max_px, res=res,
                                     correction=reg_t.get("local_correction"))
    h, w = min(rgb.shape[0], th.shape[0]), min(rgb.shape[1], th.shape[1])
    return {"rgb": rgb[:h, :w], "thermal": th[:h, :w], "ext": ext, "res": res,
            "temperature": None if temp is None else temp[:h, :w], "t_range": t_range}


def colour_temperature(temp: np.ndarray, low=None, high=None):
    """Temperatures (NaN = none) as an inferno image (white where none) and its (low, high) degC range."""
    import matplotlib

    ok = np.isfinite(temp)
    if low is None or high is None:
        if ok.any():                                   # robust: sky-facing surfaces (-10 degC and below) left out
            v = temp[ok]
            med = np.median(v)
            mad = 1.4826 * np.median(np.abs(v - med)) or 1.0
            v = v[np.abs(v - med) <= 5 * mad]
            low, high = np.percentile(v, [2, 98])
        else:
            low, high = 0.0, 1.0
    t = np.clip((np.nan_to_num(temp) - low) / max(high - low, 1e-6), 0, 1)
    img = (matplotlib.colormaps["inferno"](t)[..., :3] * 255).astype(np.uint8)
    img[~ok] = 255
    return img, (float(low), float(high))


def _has_data(img: np.ndarray) -> np.ndarray:
    return ~np.all(img == 255, axis=-1)


def blend(rgb: np.ndarray, thermal: np.ndarray, opacity: float) -> np.ndarray:
    """RGB with the thermal image over it at `opacity` (only where the thermal mosaic has data)."""
    a = (opacity * _has_data(thermal))[..., None]
    return (rgb * (1 - a) + thermal * a).astype(np.uint8)


def _uri(img: np.ndarray, alpha: np.ndarray | None = None) -> str:
    from PIL import Image

    buf = io.BytesIO()
    if alpha is None:
        Image.fromarray(img).save(buf, "JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    Image.fromarray(np.dstack([img, alpha.astype(np.uint8) * 255])).save(buf, "WEBP", quality=85)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 html,body{margin:0;height:100%;background:#111827;font-family:'Malgun Gothic',sans-serif;color:#e5e7eb;overflow:hidden}
 #bar{position:fixed;left:0;right:0;top:0;height:46px;display:flex;gap:18px;align-items:center;padding:0 14px;
      background:#1f2937;z-index:5;font-size:14px}
 #bar b{color:#fff} #bar input[type=range]{width:200px;vertical-align:middle}
 #view{position:absolute;top:46px;left:0;right:0;bottom:0;cursor:grab}
 #view.drag{cursor:grabbing}
 #stage{position:absolute;left:0;top:0;transform-origin:0 0}
 #stage img{position:absolute;left:0;top:0;image-rendering:auto;user-select:none;-webkit-user-drag:none}
 #swipe{position:absolute;top:0;bottom:0;width:2px;background:#fde047;display:none;pointer-events:none}
 #pos{margin-left:auto;font-variant-numeric:tabular-nums;color:#cbd5e1}
</style></head><body>
<div id="bar"><b>__TITLE__</b>
 <label>열화상 불투명도 <input id="op" type="range" min="0" max="100" value="__OP__"> <span id="opv">__OP__%</span></label>
 <label><input id="sw" type="checkbox"> 좌우 비교 (오른쪽 = 열화상)</label>
 <label id="swl" style="display:none">경계 <input id="sx" type="range" min="0" max="100" value="50"></label>
 <span style="color:#94a3b8">휠 = 확대·축소, 드래그 = 이동, 더블클릭 = 전체</span>
 <span id="pos"></span></div>
<div id="view"><div id="stage"><img id="rgb" src="__RGB__"><img id="th" src="__TH__"></div><div id="swipe"></div></div>
<script>
const E=__EXT__, W=__W__, H=__H__;
const view=document.getElementById('view'), stage=document.getElementById('stage'), th=document.getElementById('th');
const op=document.getElementById('op'), opv=document.getElementById('opv'), sw=document.getElementById('sw'),
      sx=document.getElementById('sx'), swl=document.getElementById('swl'), line=document.getElementById('swipe'),
      pos=document.getElementById('pos');
let s=1, x=0, y=0;
function fit(){const r=view.getBoundingClientRect(); s=Math.min(r.width/W, r.height/H); x=(r.width-W*s)/2; y=(r.height-H*s)/2; draw();}
function draw(){stage.style.transform=`translate(${x}px,${y}px) scale(${s})`; clip();}
function clip(){
  th.style.opacity=op.value/100; opv.textContent=op.value+'%';
  if(sw.checked){const px=W*sx.value/100; th.style.clipPath=`inset(0 0 0 ${px}px)`;
    line.style.display='block'; line.style.left=(x+px*s)+'px';}
  else{th.style.clipPath='none'; line.style.display='none';}}
op.oninput=clip; sx.oninput=clip; sw.onchange=()=>{swl.style.display=sw.checked?'inline':'none'; clip();};
view.addEventListener('wheel',e=>{e.preventDefault(); const r=view.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const k=Math.exp(-e.deltaY*0.0015), ns=Math.min(Math.max(s*k,0.02),40); x=mx-(mx-x)*ns/s; y=my-(my-y)*ns/s; s=ns; draw();},{passive:false});
let drag=null;
view.onmousedown=e=>{drag=[e.clientX-x,e.clientY-y]; view.classList.add('drag');};
window.onmouseup=()=>{drag=null; view.classList.remove('drag');};
window.onmousemove=e=>{const r=view.getBoundingClientRect(); if(drag){x=e.clientX-drag[0]; y=e.clientY-drag[1]; draw();}
  const ix=(e.clientX-r.left-x)/s, iy=(e.clientY-r.top-y)/s;
  if(ix>=0&&ix<W&&iy>=0&&iy<H){pos.textContent=`측점 ${(E[0]+(E[1]-E[0])*ix/W).toFixed(2)} m · 표고 ${(E[3]-(E[3]-E[2])*iy/H).toFixed(2)} m`;}};
view.ondblclick=fit; window.onresize=fit; fit();
</script></body></html>"""


def overlay_page(pair: dict, title: str, opacity: float = 0.5) -> str:
    """The popup page: both images stacked on the same grid, the thermal one with an opacity slider / swipe."""
    rgb, th = pair["rgb"], pair["thermal"]
    H, W = rgb.shape[:2]
    return (PAGE.replace("__TITLE__", title).replace("__OP__", str(int(round(opacity * 100))))
            .replace("__RGB__", _uri(rgb)).replace("__TH__", _uri(th, _has_data(th)))
            .replace("__EXT__", json.dumps([float(e) for e in pair["ext"]])).replace("__W__", str(W)).replace("__H__", str(H)))


ANOMALY_COLOURS = {"차가움": "#2563eb", "따뜻함": "#dc2626"}
ANOMALY_FILL = {"차가움": "rgba(37,99,235,0.45)", "따뜻함": "rgba(220,38,38,0.45)"}


def anomaly_figure(result, an, background, ext, title: str = ""):
    """Plotly map: the facade photo, the mapped traces (thin), the thermal anomalies filled (cold blue, warm red)
    with their ids; hover gives each anomaly's row of the table."""
    import cv2
    import plotly.graph_objects as go

    from .seepage_view import _jpeg_uri

    H, W = background.shape[:2]
    dx, dy = (ext[1] - ext[0]) / W, (ext[3] - ext[2]) / H
    fig = go.Figure(go.Image(source=_jpeg_uri(background), x0=ext[0] + dx / 2, dx=dx, y0=ext[2] + dy / 2, dy=dy,
                             hoverinfo="skip"))
    xs, ys = [], []
    for t in result.traces:
        xs += [float(t.coords[0, 0]), float(t.coords[-1, 0]), None]
        ys += [float(t.coords[0, 1]), float(t.coords[-1, 1]), None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(250,204,21,0.75)", width=1),
                             name="절리 트레이스", hoverinfo="skip"))
    rows = {int(r["id"]): r for _, r in an.table.iterrows()} if len(an.table) else {}
    near_col = f"절리 {an.params['near_m']:g} m 이내 (%)"
    shown = set()
    for sign, kind in ((1, "차가움"), (-1, "따뜻함")):
        ids = np.unique(an.labels[an.labels * sign > 0]) * sign
        for i in ids:
            mask = (an.labels == i * sign).astype(np.uint8)
            cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            r = rows.get(int(i))
            if r is None:
                continue
            dt = f" · ΔT {r['ΔT 평균 (°C)']:+.1f} °C" if "ΔT 평균 (°C)" in r else ""
            tip = (f"#{int(i)} {kind} · z {r['z 평균']}{dt} · {r['면적 (m²)']} m²<br>절리 {an.params['near_m']:g} m 이내 "
                   f"{r[near_col]:.0f}% · {r['트레이스 방향군']} · {r['절리군']} · 유출 가능성 {r['유출 가능성']}"
                   + (" · 그늘 의심" if r["그늘 의심"] else ""))
            for c in cs:
                p = c[:, 0, :].astype(float)
                if len(p) < 3:
                    continue
                u = an.u_min + (p[:, 0] + 0.5) * an.res
                v = an.v_max - (p[:, 1] + 0.5) * an.res
                fig.add_trace(go.Scatter(x=np.r_[u, u[:1]], y=np.r_[v, v[:1]], mode="lines", fill="toself",
                                         line=dict(color=ANOMALY_COLOURS[kind], width=1.5),
                                         fillcolor=ANOMALY_FILL[kind],
                                         name=f"{kind} 이상", legendgroup=kind, showlegend=kind not in shown,
                                         text=tip, hoverinfo="text", hoveron="fills"))
                shown.add(kind)
            fig.add_annotation(x=r["u"], y=r["v"], text=f"{int(i)}", showarrow=False, yshift=12,
                               font=dict(size=10, color=ANOMALY_COLOURS[kind]), bgcolor="rgba(255,255,255,0.75)")
    fig.update_layout(title=title or "열화상 이상 부분 × 절리 트레이스", legend=dict(orientation="h"))
    fig.update_xaxes(title="측점 (m)", range=[ext[0], ext[1]])
    fig.update_yaxes(title="표고 (m)", range=[ext[2], ext[3]])
    return fig


def tiles_overlay(result, pair: dict, max_px: int = 4096, max_vertices: int = 400_000) -> dict | None:
    """The thermal image of `pair` (degC in the inferno map, or the palette mosaic) draped on the wall surface, as
    an overlay entry for tiles3d.scene_json; the drape grid is as fine as `max_vertices` allows (0.1 m at most)."""
    import matplotlib

    from . import setmap3d

    th = pair["thermal"]
    k = max(1, int(np.ceil(max(th.shape[:2]) / max_px)))
    img = th[::k, ::k]
    rgba = np.dstack([img, (_has_data(img) * 255).astype(np.uint8)])
    ext, res = pair["ext"], pair["res"] * k
    U, V = rgba.shape[1] * res, rgba.shape[0] * res
    step = max(0.1, float(np.sqrt(U * V / max_vertices)))
    tr = pair.get("t_range")
    if tr:
        cm = matplotlib.colormaps["inferno"]
        colors = ["#%02x%02x%02x" % tuple(int(255 * c) for c in cm(t)[:3]) for t in (0.1, 0.5, 0.9)]
        name = f"열화상 온도 {tr[0]:.1f}~{tr[1]:.1f} °C (낮음→높음)"
    else:
        colors, name = None, "열화상 (카메라 팔레트, 상대값)"
    return setmap3d._image_overlay(result, rgba, ext[0], ext[3], res, res, name, "thermal", step, 0.12, colors=colors)
