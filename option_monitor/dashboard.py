"""Option market dashboard — visualizes OptionMonitor snapshots.

Reads CSV snapshots written by ``option_monitor/run.py`` from
``output/option_monitor/<date>/`` (legacy flat CSVs in the folder root are
also picked up) and renders four views:

  Live            – real-time prices (websocket stream or 1-minute polling)
  Market Overview – per-symbol sentiment / vol / flow summary for one session
  Symbol Detail   – term structure, skew, put/call flow and key levels
  Trends          – metric history across snapshots (needs >= 2 run dates)

Every view is labelled with the vintage of its data: 🔴 LIVE quotes carry
their timestamp; EOD panels state which completed session they describe.

Run:
    uv run streamlit run option_monitor/dashboard.py

Optional, for lowest-latency live quotes, run the streamer alongside:
    uv run python -m option_monitor.stream

See option_monitor/README.md for the full documentation.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from option_monitor.realtime import (
    STREAM_FILE,
    RealtimeQuoteService,
    read_stream_quotes,
)

BASE_DIR = Path(__file__).parent.parent / "output" / "option_monitor"
TABLES = [
    "spot", "iv_term_structure", "iv_skew", "pc_overall",
    "pc_by_expiry", "pc_by_moneyness", "positioning", "derived",
    "data_quality",
]
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

GREEN, RED, BLUE, ORANGE = "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"

st.set_page_config(page_title="Option Market Monitor", layout="wide")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_meta(folder: Path) -> dict:
    """Read the snapshot's meta.json; {} for older snapshots."""
    p = Path(folder) / "meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


@st.cache_data(ttl=60, show_spinner="Fetching live quotes…")
def poll_live_quotes(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Polling fallback (cached 60 s so reruns don't hammer Yahoo)."""
    return RealtimeQuoteService().fetch(list(symbols))


def get_live_quotes(symbols: list[str]) -> pd.DataFrame:
    """Websocket stream file when fresh, else batched 1-minute polling."""
    stream_path = Path(__file__).parent.parent / STREAM_FILE
    df = read_stream_quotes(stream_path, symbols)
    if not df.empty and bool(df["is_live"].any()):
        return df
    return poll_live_quotes(tuple(symbols))


def discover_snapshots() -> dict[str, Path]:
    """Map snapshot date -> folder. Includes legacy flat CSVs in the root."""
    snaps: dict[str, Path] = {}
    if not BASE_DIR.exists():
        return snaps
    for sub in sorted(BASE_DIR.iterdir()):
        if sub.is_dir() and DATE_RE.match(sub.name):
            snaps[sub.name] = sub
    legacy_spot = BASE_DIR / "spot.csv"
    if legacy_spot.exists():
        try:
            vdate = str(pd.read_csv(legacy_spot)["valuation_date"].iloc[0])
            snaps.setdefault(vdate, BASE_DIR)
        except Exception:
            pass
    return snaps


@st.cache_data(show_spinner=False)
def load_snapshot(folder: str) -> dict[str, pd.DataFrame]:
    out = {}
    for name in TABLES:
        path = Path(folder) / f"{name}.csv"
        out[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return out


def interp_tenor(df: pd.DataFrame, col: str, target: float) -> float | None:
    if df.empty or col not in df.columns:
        return None
    pts = df[["dte", col]].dropna().sort_values("dte")
    if pts.empty:
        return None
    return float(np.interp(target, pts["dte"], pts[col]))


def build_overview(snap: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Use derived.csv when present; otherwise rebuild it from base tables
    (keeps legacy snapshots working)."""
    if not snap["derived"].empty:
        return snap["derived"].copy()

    spot, ts, sk, pco = (
        snap["spot"], snap["iv_term_structure"], snap["iv_skew"], snap["pc_overall"]
    )
    if spot.empty:
        return pd.DataFrame()
    rows = []
    for _, s in spot.iterrows():
        sym = s["symbol"]
        iv30 = interp_tenor(ts[ts["symbol"] == sym], "atm_iv", 30)
        iv90 = interp_tenor(ts[ts["symbol"] == sym], "atm_iv", 90)
        skew30 = interp_tenor(sk[sk["symbol"] == sym], "skew", 30)
        rv20 = s.get("rv_20d")
        pc = pco[pco["symbol"] == sym]
        rows.append({
            "symbol": sym,
            "spot": s["spot"],
            "change_pct": s["change_pct"],
            "rel_volume": s.get("rel_volume"),
            "rv_20d": rv20,
            "atm_iv_30d": iv30,
            "atm_iv_90d": iv90,
            "term_slope": iv30 / iv90 if iv30 and iv90 else None,
            "skew_30d": skew30,
            "iv_rv_spread": (iv30 * 100 - rv20) if iv30 is not None and pd.notna(rv20) else None,
            "pc_vol_ratio": float(pc["pc_vol_ratio"].iloc[0]) if len(pc) else None,
        })
    df = pd.DataFrame(rows)
    # cross-sectional sentiment (higher = more bullish)
    if len(df) >= 3:
        zs = []
        for col in ["pc_vol_ratio", "skew_30d", "iv_rv_spread"]:
            x = pd.to_numeric(df[col], errors="coerce")
            std = x.std(ddof=0)
            zs.append((x - x.mean()) / std if std and std > 0 else x * 0.0)
        df["sentiment"] = (-pd.concat(zs, axis=1).mean(axis=1)).round(2)
    else:
        df["sentiment"] = None
    return df


@st.cache_data(show_spinner=False)
def load_history(folders: dict[str, str]) -> pd.DataFrame:
    """Stack the overview table across all snapshot dates."""
    frames = []
    for vdate, folder in folders.items():
        snap = load_snapshot(folder)
        ov = build_overview(snap)
        if not ov.empty:
            ov["date"] = vdate
            frames.append(ov)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Sidebar ───────────────────────────────────────────────────────────────────

snapshots = discover_snapshots()
if not snapshots:
    st.error(
        "No snapshots found. Run `uv run python -m option_monitor.run` "
        "first — CSVs are expected under `output/option_monitor/<date>/`."
    )
    st.stop()

dates = sorted(snapshots.keys(), reverse=True)
with st.sidebar:
    st.title("Option Market Monitor")
    sel_date = st.selectbox("EOD session", dates, index=0)
    snap = load_snapshot(str(snapshots[sel_date]))
    meta = load_meta(snapshots[sel_date])
    overview = build_overview(snap)
    all_symbols = sorted(overview["symbol"].unique()) if not overview.empty else []
    st.caption(f"{len(all_symbols)} symbols · {len(dates)} snapshot(s) on disk")
    if meta:
        st.caption(
            f"📁 EOD snapshot for session **{meta.get('session_date', sel_date)}**, "
            f"captured {meta.get('run_at_et', '?')}"
        )
        if meta.get("chain_captured_live_during_newer_session"):
            st.caption(
                "⚠️ option chain was captured live during a newer session — "
                "IV/flow are fresher than the EOD prices"
            )
    st.markdown(
        "**How to read the scores**\n\n"
        "- **Sentiment**: options-flow composite vs. the rest of the universe; "
        "above 0 = bullish tilt, below 0 = defensive.\n"
        "- **IV−RV**: implied minus realized vol — how much *fear premium* "
        "options carry beyond actual movement.\n"
        "- **Term slope** > 1: near-term IV above long-term (event stress).\n"
        "- **Skew**: extra cost of downside puts vs upside calls."
    )

# ── Data-quality banner ───────────────────────────────────────────────────────────

dq = snap.get("data_quality", pd.DataFrame())
if not dq.empty:
    stale_price = dq[dq["price_is_stale"] == True]["symbol"].tolist()  # noqa: E712
    stale_chain = dq[dq["chain_is_stale"] == True]["symbol"].tolist()  # noqa: E712
    no_volume = dq[dq["volume_reported"] == False]["symbol"].tolist()  # noqa: E712
    junk_iv = (
        dq[dq["iv_reliable"] == False]["symbol"].tolist()  # noqa: E712
        if "iv_reliable" in dq.columns else []
    )
    msgs = []
    if junk_iv:
        msgs.append(
            f"**Unreliable implied vols** ({len(junk_iv)} symbols): Yahoo served "
            f"placeholder IVs instead of market data (chain captured while the "
            f"market was closed). All IV-based metrics (ATM IV, skew, IV−RV, "
            f"term slope) are shown as missing — re-run the snapshot during or "
            f"just after market hours."
        )
    if stale_price:
        msgs.append(
            f"**Stale prices** ({len(stale_price)} symbols): last price bar predates "
            f"the snapshot date — the market was closed (pre-open, weekend or "
            f"holiday). Spot, change % and relative volume reflect the *prior* "
            f"session."
        )
    if stale_chain:
        msgs.append(
            f"**Stale option chains** ({len(stale_chain)} symbols): last option trade "
            f"predates the snapshot date — IV, skew and flow metrics reflect the "
            f"*prior* session's quotes."
        )
    if no_volume:
        msgs.append(
            f"**No option volume reported** ({len(no_volume)} symbols): "
            f"{', '.join(no_volume)} — volume-based metrics are shown as missing, "
            f"not zero."
        )
    if msgs:
        st.warning("⚠️ Data quality — " + sel_date + "\n\n" + "\n\n".join(msgs))
        with st.expander("Per-symbol data-quality flags"):
            st.dataframe(
                dq.drop(columns=["valuation_date"], errors="ignore"),
                width="stretch", hide_index=True,
            )
elif not overview.empty:
    st.caption(
        "No data-quality table in this snapshot (older run) — freshness of the "
        "data cannot be verified."
    )

tab_live, tab_overview, tab_detail, tab_trends = st.tabs(
    ["🔴 Live", "📊 Market Overview (EOD)", "🔍 Symbol Detail (EOD)", "📈 Trends (EOD)"]
)


def _eod_caption() -> str:
    cap = f"📁 **END-OF-DAY** — completed session **{sel_date}**"
    if meta.get("run_at_et"):
        cap += f", captured {meta['run_at_et']}"
    if meta.get("chain_captured_live_during_newer_session"):
        cap += (" · ⚠️ option chain captured live during a newer session "
                "(IV/flow fresher than the EOD prices)")
    return cap


# ── Live ─────────────────────────────────────────────────────────────────────────

with tab_live:
    st.caption(
        "🔴 **REAL-TIME** — price ticks via Yahoo websocket stream when "
        "`python -m option_monitor.stream` is running, otherwise 1-minute "
        "polling (~1 min delay, cached 60 s). Quotes older than 3 minutes "
        "are marked *not live* (last print of a closed session). "
        "Options data is **not** available in real time — see the EOD tabs."
    )
    if st.button("↻ Refresh quotes"):
        poll_live_quotes.clear()
        st.rerun()

    live_symbols = all_symbols or ["SPY", "QQQ", "IWM", "DIA"]
    try:
        live = get_live_quotes(live_symbols)
    except Exception as exc:  # network failure must not kill the app
        live = pd.DataFrame()
        st.error(f"Live quotes unavailable: {exc}")

    if live.empty:
        st.warning("No live quotes returned — missing, not faked.")
    else:
        n_live = int(live["is_live"].sum())
        src = live["source"].iloc[0]
        asof = max(live["quote_time"])
        st.markdown(
            f"**{n_live}/{len(live)} symbols live** · source: `{src}` · "
            f"latest tick: {asof:%Y-%m-%d %H:%M:%S %Z}"
        )
        if n_live == 0:
            st.info(
                "Market appears closed — these are the last prints of the "
                "previous session, not live prices."
            )

        disp = live.copy()
        disp["quote_time"] = pd.to_datetime(disp["quote_time"], utc=True) \
            .dt.tz_convert("America/New_York").dt.strftime("%H:%M:%S ET")
        disp["status"] = np.where(disp["is_live"], "🔴 live", "⚪ stale")
        show = disp[["symbol", "price", "change_pct", "prev_close",
                     "quote_time", "age_sec", "status"]]
        styled = (
            show.sort_values("change_pct", ascending=False, na_position="last")
            .style.format({"price": "{:,.2f}", "prev_close": "{:,.2f}",
                           "change_pct": "{:+.2f}%"}, na_rep="–")
            .background_gradient(subset=["change_pct"], cmap="RdYlGn",
                                 vmin=-3, vmax=3)
        )
        st.dataframe(styled, width="stretch", height=600, hide_index=True)

        chg = live.dropna(subset=["change_pct"]).sort_values("change_pct")
        if len(chg):
            fig = px.bar(
                chg, x="change_pct", y="symbol", orientation="h",
                color="change_pct", color_continuous_scale=[RED, "#cccccc", GREEN],
                color_continuous_midpoint=0, height=max(420, 22 * len(chg)),
                title="Change vs previous session close (%)",
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title="",
                              xaxis_title="% change")
            st.plotly_chart(fig, width="stretch")

# ── Market Overview ───────────────────────────────────────────────────────────

with tab_overview:
    st.caption(_eod_caption())
    if overview.empty:
        st.warning("Snapshot has no data.")
    else:
        ov = overview.copy()

        # headline KPIs
        c1, c2, c3, c4 = st.columns(4)
        spy = ov[ov["symbol"] == "SPY"]
        if len(spy):
            c1.metric("SPY", f"{spy['spot'].iloc[0]:,.2f}",
                      f"{spy['change_pct'].iloc[0]:+.2f}%")
        med_ivrv = pd.to_numeric(ov["iv_rv_spread"], errors="coerce").median()
        c2.metric("Median IV−RV spread", f"{med_ivrv:+.1f} pts" if pd.notna(med_ivrv) else "–",
                  help="Positive = options pricing in more risk than realized")
        slope = pd.to_numeric(ov["term_slope"], errors="coerce")
        c3.metric("Inverted term structures", f"{int((slope > 1).sum())} / {slope.notna().sum()}",
                  help="Symbols whose 30d IV exceeds 90d IV — near-term stress")
        hot = pd.to_numeric(ov["rel_volume"], errors="coerce")
        c4.metric("Volume > 1.5× average", f"{int((hot > 1.5).sum())}",
                  help="Symbols trading on unusually heavy underlying volume")

        st.divider()

        sent = pd.to_numeric(ov["sentiment"], errors="coerce")
        if sent.notna().any():
            plot_df = ov.assign(sentiment=sent).dropna(subset=["sentiment"]) \
                        .sort_values("sentiment")
            fig = px.bar(
                plot_df, x="sentiment", y="symbol", orientation="h",
                color="sentiment", color_continuous_scale=[RED, "#cccccc", GREEN],
                color_continuous_midpoint=0, height=max(420, 22 * len(plot_df)),
                title=f"Options-flow sentiment — {sel_date} (higher = more bullish)",
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="composite z-score")
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Composite of put/call volume ratio, 30-day skew and IV−RV spread, "
                "each compared across the universe. A defensive print means option "
                "traders are paying up for protection in that name relative to peers."
            )

        st.subheader("All symbols")
        cols = ["symbol", "spot", "change_pct", "rel_volume", "rv_20d",
                "atm_iv_30d", "iv_rv_spread", "term_slope", "skew_30d",
                "pc_vol_ratio", "sentiment"]
        table = ov[[c for c in cols if c in ov.columns]].copy()
        for c in table.columns:
            if c != "symbol":
                table[c] = pd.to_numeric(table[c], errors="coerce")
        styled = (
            table.sort_values("sentiment", ascending=False, na_position="last")
            .style.format(precision=2, na_rep="–")
            .background_gradient(subset=["sentiment"], cmap="RdYlGn", vmin=-2, vmax=2)
            .background_gradient(subset=["change_pct"], cmap="RdYlGn", vmin=-3, vmax=3)
            .background_gradient(subset=["skew_30d"], cmap="OrRd")
        )
        st.dataframe(styled, width="stretch", height=600, hide_index=True)

# ── Symbol Detail ─────────────────────────────────────────────────────────────

with tab_detail:
    st.caption(_eod_caption())
    if not all_symbols:
        st.warning("Snapshot has no data.")
    else:
        default_ix = all_symbols.index("SPY") if "SPY" in all_symbols else 0
        sym = st.selectbox("Symbol", all_symbols, index=default_ix)

        srow = overview[overview["symbol"] == sym].iloc[0]
        pos = snap["positioning"]
        pos = pos[pos["symbol"] == sym] if not pos.empty else pd.DataFrame()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Spot", f"{srow['spot']:,.2f}", f"{srow['change_pct']:+.2f}%")
        k2.metric("30d ATM IV", f"{srow['atm_iv_30d'] * 100:.1f}%"
                  if pd.notna(srow.get("atm_iv_30d")) else "–")
        k3.metric("IV−RV", f"{srow['iv_rv_spread']:+.1f} pts"
                  if pd.notna(srow.get("iv_rv_spread")) else "–",
                  help="Implied vol minus 20-day realized vol")
        k4.metric("P/C volume", f"{srow['pc_vol_ratio']:.2f}"
                  if pd.notna(srow.get("pc_vol_ratio")) else "–",
                  help="> 1 = more puts than calls traded today")
        k5.metric("Rel. volume", f"{srow['rel_volume']:.2f}×"
                  if pd.notna(srow.get("rel_volume")) else "–",
                  help="Underlying volume vs its 20-day average")

        left, right = st.columns(2)

        ts = snap["iv_term_structure"]
        ts = ts[ts["symbol"] == sym].sort_values("dte") if not ts.empty else ts
        with left:
            if not ts.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts["dte"], y=ts["atm_iv"] * 100, mode="lines+markers",
                    line=dict(color=BLUE), name="ATM IV",
                ))
                if pd.notna(srow.get("rv_20d")):
                    fig.add_hline(y=float(srow["rv_20d"]), line_dash="dot",
                                  line_color=ORANGE,
                                  annotation_text="20d realized vol")
                fig.update_layout(title="IV term structure", xaxis_title="days to expiry",
                                  yaxis_title="ATM IV (%)", height=380)
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "Downward slope into the front = event premium / stress; "
                    "IV far above the dotted realized-vol line = expensive options."
                )

        sk = snap["iv_skew"]
        sk = sk[sk["symbol"] == sym].sort_values("dte") if not sk.empty else sk
        with right:
            if not sk.empty:
                fig = go.Figure(go.Scatter(
                    x=sk["dte"], y=sk["skew"] * 100, mode="lines+markers",
                    line=dict(color=RED),
                ))
                fig.add_hline(y=0, line_dash="dot", line_color="grey")
                fig.update_layout(title="Put–call skew by expiry (±5% OTM)",
                                  xaxis_title="days to expiry",
                                  yaxis_title="skew (IV pts)", height=380)
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "Above zero = downside puts cost more than upside calls "
                    "(normal in equities). A spike in the front end = hedging rush."
                )

        left2, right2 = st.columns(2)

        pcm = snap["pc_by_moneyness"]
        pcm = pcm[pcm["symbol"] == sym] if not pcm.empty else pcm
        with left2:
            if not pcm.empty:
                order = ["deep_below", "below", "near_atm", "above", "deep_above"]
                pcm = pcm.set_index("moneyness_bucket").reindex(order).reset_index()
                fig = go.Figure([
                    go.Bar(name="Put OI", x=pcm["moneyness_bucket"], y=pcm["put_oi"], marker_color=RED),
                    go.Bar(name="Call OI", x=pcm["moneyness_bucket"], y=pcm["call_oi"], marker_color=GREEN),
                ])
                fig.update_layout(barmode="group", title="Open interest by strike level",
                                  xaxis_title="strike vs spot", yaxis_title="contracts", height=380)
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "Put OI piled up below spot = hedges/support; call OI above spot "
                    "= upside bets and likely resistance."
                )

        pce = snap["pc_by_expiry"]
        pce = pce[pce["symbol"] == sym] if not pce.empty else pce
        with right2:
            if not pce.empty:
                order = ["0-30d", "31-60d", "61-90d", ">90d"]
                pce = pce.set_index("dte_bucket").reindex(order).dropna(how="all").reset_index()
                fig = go.Figure([
                    go.Bar(name="P/C volume", x=pce["dte_bucket"], y=pce["pc_vol_ratio"], marker_color=ORANGE),
                    go.Bar(name="P/C open interest", x=pce["dte_bucket"], y=pce["pc_oi_ratio"], marker_color=BLUE),
                ])
                fig.add_hline(y=1, line_dash="dot", line_color="grey")
                fig.update_layout(barmode="group", title="Put/call ratios by tenor",
                                  xaxis_title="days to expiry", yaxis_title="ratio", height=380)
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "Volume = today's flow (fast sentiment); open interest = standing "
                    "positions. Both above the line = put-heavy."
                )

        if len(pos):
            p = pos.iloc[0]
            levels = {
                "Put wall": (p.get("put_wall"), RED),
                "Max pain": (p.get("max_pain"), ORANGE),
                "Spot": (p.get("spot"), BLUE),
                "Call wall": (p.get("call_wall"), GREEN),
            }
            fig = go.Figure()
            for name, (val, color) in levels.items():
                if pd.notna(val):
                    fig.add_trace(go.Scatter(
                        x=[val], y=[0], mode="markers+text", text=[f"{name}<br>{val:,.0f}"],
                        textposition="top center", marker=dict(size=16, color=color),
                        showlegend=False,
                    ))
            fig.update_layout(title="Key option levels (near-term chain, DTE ≤ 45)",
                              yaxis=dict(visible=False), xaxis_title="price", height=240)
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Call/put walls are the strikes with the largest open interest — "
                "they often act as magnets or barriers near expiry. Max pain is the "
                "price where option holders lose the most in aggregate."
            )

# ── Trends ────────────────────────────────────────────────────────────────────

with tab_trends:
    st.caption("📁 **END-OF-DAY** — one point per completed session.")
    history = load_history({d: str(p) for d, p in snapshots.items()})
    n_dates = history["date"].nunique() if not history.empty else 0
    if n_dates < 2:
        st.info(
            "Trends need at least two snapshot dates. Run "
            "`uv run python -m option_monitor.run` daily (each run saves into "
            "`output/option_monitor/<date>/`) and history will accumulate here."
        )
    else:
        default = [s for s in ["SPY", "QQQ"] if s in all_symbols] or all_symbols[:2]
        chosen = st.multiselect("Symbols", all_symbols, default=default)
        metric_labels = {
            "sentiment": "Sentiment (composite z-score)",
            "atm_iv_30d": "30d ATM implied vol",
            "iv_rv_spread": "IV − RV spread (pts)",
            "skew_30d": "30d put-call skew",
            "pc_vol_ratio": "Put/call volume ratio",
            "term_slope": "Term-structure slope (30d/90d)",
            "change_pct": "Daily change (%)",
            "rel_volume": "Relative volume (×20d avg)",
        }
        sel_metrics = st.multiselect(
            "Metrics", list(metric_labels), default=["sentiment", "atm_iv_30d", "pc_vol_ratio"],
            format_func=lambda m: metric_labels[m],
        )
        hist = history[history["symbol"].isin(chosen)].sort_values("date")
        for metric in sel_metrics:
            if metric not in hist.columns:
                continue
            fig = px.line(
                hist.assign(**{metric: pd.to_numeric(hist[metric], errors="coerce")}),
                x="date", y=metric, color="symbol", markers=True,
                title=metric_labels[metric], height=340,
            )
            st.plotly_chart(fig, width="stretch")
        st.caption(
            "Watching the *change* in these series matters more than the level: "
            "rising skew + rising P/C volume + falling sentiment is a classic "
            "risk-off rotation even before price confirms it."
        )
