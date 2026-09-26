"""Presentation helpers: palette (validated reference palette), badges, pydeck maps, Altair charts.

Colour follows the job: magnitude = one sequential blue ramp; model identity = categorical slots 1-3 in fixed
order (GLM blue, LightGBM orange, TabPFN aqua); state = the reserved status palette, always with an icon + label.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk

SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]
SERIES = {"glm": "#2a78d6", "lgbm": "#eb6834", "tabpfn": "#1baf7a"}
SERIES_LABEL = {"glm": "GLM (active)", "lgbm": "LightGBM", "tabpfn": "TabPFN v2 (benchmark)"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
       "surface": "#fcfcfb"}
REC_ICON = {"DE-ENERGIZE": "🔴 DE-ENERGIZE", "WATCH": "🟡 WATCH", "JUDGMENT": "⚪ JUDGMENT"}

CARTO_POSITRON = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
STUDY_VIEW = pdk.ViewState(latitude=26.72, longitude=-81.95, zoom=8.6, pitch=0)
FL_VIEW = pdk.ViewState(latitude=27.6, longitude=-82.6, zoom=5.7, pitch=0)


def rgb(hex_: str, a: int = 255) -> list[int]:
    h = hex_.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


def seq_rgb(v: float, a: int = 215) -> list[int]:
    """Sequential blue for a value in [0, 1] (light = near zero, dark = high)."""
    v = 0.0 if v is None or np.isnan(v) else float(min(max(v, 0.0), 1.0))
    x = v * (len(SEQ) - 1)
    i = int(np.floor(x))
    j = min(i + 1, len(SEQ) - 1)
    c1, c2 = np.array(rgb(SEQ[i])[:3]), np.array(rgb(SEQ[j])[:3])
    c = (c1 + (c2 - c1) * (x - i)).round().astype(int).tolist()
    return c + [a]


def deck(layers, view, tooltip=None, height=460) -> pdk.Deck:
    # Carto Positron: no token needed. Given as a URL so every renderer (Streamlit, HTML export) resolves the same style.
    return pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto", map_style=CARTO_POSITRON,
                    tooltip=tooltip or False, height=height)


TOOLTIP_STYLE = {"backgroundColor": "#ffffff", "color": INK["primary"], "fontSize": "12px",
                 "border": "1px solid rgba(11,11,11,0.10)", "borderRadius": "6px", "padding": "6px 8px"}


def fmt(x, d=0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:,.{d}f}"


def altair_theme(chart: alt.Chart) -> alt.Chart:
    return (chart.configure_view(strokeWidth=0)
            .configure_axis(gridColor=INK["grid"], domainColor=INK["axis"], tickColor=INK["axis"], labelColor=INK["secondary"],
                            titleColor=INK["secondary"], labelFontSize=11, titleFontSize=11, titleFontWeight="normal")
            .configure_legend(labelColor=INK["secondary"], titleColor=INK["secondary"], orient="top", labelFontSize=11)
            .configure_title(color=INK["primary"], fontSize=13, anchor="start", fontWeight="normal"))


def pred_vs_obs(df: pd.DataFrame, model: str) -> alt.Chart:
    """Single series (one model): observed vs predicted P50 with the P10-P90 interval; y = x reference."""
    color = SERIES[model]
    base = alt.Chart(df)
    lim = float(max(df["obs_out"].max(), df["pred_out_p90"].max())) * 1.1
    ref = alt.Chart(pd.DataFrame({"x": [0, lim]})).mark_line(color=INK["muted"], strokeDash=[4, 4], strokeWidth=1).encode(
        x="x:Q", y="x:Q")
    scale = alt.Scale(type="sqrt", domain=[0, lim], nice=False)
    bars = base.mark_rule(color=color, strokeWidth=2, opacity=0.45).encode(
        x=alt.X("obs_out:Q", scale=scale, title="Observed peak customers out (Milton)"),
        y=alt.Y("pred_out_p10:Q", scale=scale, title="Predicted at T-72h, P50 with P10-P90"), y2="pred_out_p90:Q")
    pts = base.mark_circle(size=70, color=color, opacity=0.95, stroke=INK["surface"], strokeWidth=2).encode(
        x=alt.X("obs_out:Q", scale=scale), y=alt.Y("pred_out_p50:Q", scale=scale),
        tooltip=[alt.Tooltip("county:N", title="County"), alt.Tooltip("obs_out:Q", title="Observed", format=",.0f"),
                 alt.Tooltip("pred_out_p10:Q", title="P10", format=",.0f"), alt.Tooltip("pred_out_p50:Q", title="P50", format=",.0f"),
                 alt.Tooltip("pred_out_p90:Q", title="P90", format=",.0f")])
    return altair_theme((ref + bars + pts).properties(
        height=380, title=f"{SERIES_LABEL[model]}: Milton at T-72h, all 67 counties (square-root axes; dashed = perfect)"))


def leadtime(df: pd.DataFrame, metric: str, title: str, target: float | None) -> alt.Chart:
    order = [m for m in ("glm", "lgbm", "tabpfn") if m in set(df["model"])]
    df = df.assign(series=df["model"].map(SERIES_LABEL))
    color = alt.Color("series:N", title=None, sort=[SERIES_LABEL[m] for m in order],
                      scale=alt.Scale(domain=[SERIES_LABEL[m] for m in order], range=[SERIES[m] for m in order]))
    x = alt.X("hours_to_landfall:Q", title="Hours before landfall (T-)", scale=alt.Scale(reverse=True, domain=[0, 120]))
    y = alt.Y(f"{metric}:Q", title=None, scale=alt.Scale(domain=[0, 1]))
    line = alt.Chart(df).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    pts = alt.Chart(df).mark_circle(size=64, stroke=INK["surface"], strokeWidth=2, opacity=1).encode(
        x=x, y=y, color=color, tooltip=[alt.Tooltip("series:N", title="Model"),
                                        alt.Tooltip("hours_to_landfall:Q", title="T- hours", format=".1f"),
                                        alt.Tooltip(f"{metric}:Q", title=title, format=".2f")])
    layers = [line, pts, alt.Chart(pd.DataFrame({"h": [72]})).mark_rule(color=INK["axis"], strokeDash=[2, 3]).encode(x="h:Q")]
    if target is not None:
        layers.append(alt.Chart(pd.DataFrame({"t": [target]})).mark_rule(color=INK["muted"], strokeDash=[4, 4]).encode(y="t:Q"))
        layers.append(alt.Chart(pd.DataFrame({"t": [target], "h": [118], "lab": [f"target {target:.2f}"]})).mark_text(
            align="left", dy=-7, fontSize=10, color=INK["muted"]).encode(x="h:Q", y="t:Q", text="lab:N"))
    return altair_theme(alt.layer(*layers).properties(height=250, title=title))
