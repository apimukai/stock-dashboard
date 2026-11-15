"""แท็บย้อนหลัง & Backtest (ใช้ใน stock_dashboard.py)
1) Sector Rotation ย้อนหลัง: เลือกวันที่ในอดีต ดูว่าแต่ละกลุ่มอยู่ช่องไหน และหลังจากนั้นกลุ่มทำผลตอบแทนเท่าไร
2) Backtest กลยุทธ์ A (Sector momentum): ทุกสิ้นเดือน ถือ N กลุ่มที่ผลตอบแทนย้อนหลังดีที่สุด เทียบ S&P 500
ข้อมูล: SPDR Select Sector ETF + SPY จาก Yahoo Finance (ราคาปรับปันผลแล้ว = ผลตอบแทนรวมปันผล)
"""
import datetime as dt
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from sector_heatmap import SECTORS

ETF_TH = {etf: th for etf, th in SECTORS.values()}
QUAD = {  # ชื่อช่อง, สี
    "lead": ("🟢 นำตลาด", "#1a9850"), "weak": ("🟠 เริ่มแผ่ว", "#e08214"),
    "lag": ("🔴 แพ้ตลาด", "#d73027"), "imp": ("🔵 เริ่มฟื้น", "#4575b4"),
}
LB_DAYS, MOM_DAYS = 63, 21  # 3 เดือน, 1 เดือน (วันทำการ) ตรงกับแท็บ Heatmap


@st.cache_data(ttl=6 * 3600, show_spinner="กำลังดึงราคา sector ETF ย้อนหลังทั้งหมด ...")
def long_prices() -> pd.DataFrame:
    syms = list(ETF_TH) + ["SPY"]
    d = yf.download(syms, period="max", interval="1d", progress=False, auto_adjust=True)["Close"]
    d.index = pd.DatetimeIndex(d.index.tz_localize(None) if d.index.tz is not None else d.index)
    return d[d.index >= "1999-01-01"].dropna(how="all")


def _quad(x, y):
    return "lead" if x > 0 and y > 0 else "weak" if x > 0 else "lag" if y < 0 else "imp"


def rotation_frame(px: pd.DataFrame) -> dict:
    """ต่อ ETF: DataFrame x (ส่วนเกิน 3 เดือน %), y (โมเมนตัม 1 เดือน) รายวัน"""
    out = {}
    for etf in ETF_TH:
        rs = (px[etf] / px["SPY"]).dropna()
        x = (rs / rs.shift(LB_DAYS) - 1) * 100
        y = x - x.shift(MOM_DAYS)
        out[etf] = pd.DataFrame({"x": x, "y": y}).dropna()
    return out


# ======================= 1) Rotation ย้อนหลัง =======================
def render_rotation_history(px: pd.DataFrame):
    st.markdown("#### 🔄 Sector Rotation ย้อนหลัง")
    rot = rotation_frame(px)
    days = px["SPY"].dropna().index
    first_ok = min(df.index[0] for df in rot.values())
    weeks = days[days >= first_ok][::-5][::-1]  # จุดรายสัปดาห์ นับถอยจากวันล่าสุด

    ss = st.session_state
    lo_d, hi_d = weeks[0].date(), weeks[-1].date()
    # ตัวเลื่อนกับช่องวันที่ใช้ค่าเดียวกัน: เปลี่ยนอันหนึ่ง อีกอันตาม
    for k in ("rh_slider", "rh_jump"):
        if not isinstance(ss.get(k), dt.date) or not (lo_d <= ss[k] <= hi_d):
            ss[k] = hi_d

    def _from_slider():
        ss["rh_jump"] = ss["rh_slider"]

    def _from_jump():
        if isinstance(ss.get("rh_jump"), dt.date):
            ss["rh_slider"] = min(max(ss["rh_jump"], lo_d), hi_d)

    c1, c2, c3 = st.columns([3, 1, 1])
    c1.slider("เลือกวันที่ (เลื่อนทีละสัปดาห์)", min_value=lo_d, max_value=hi_d, step=dt.timedelta(days=7),
              format="DD MMM YYYY", key="rh_slider", on_change=_from_slider)
    c2.date_input("หรือพิมพ์วันที่", min_value=lo_d, max_value=hi_d, key="rh_jump", on_change=_from_jump,
                  format="DD/MM/YYYY")
    tail_n = c3.selectbox("เส้นทางย้อนหลัง", [4, 6, 8, 12], index=1, key="rh_tail",
                          format_func=lambda n: f"{n} สัปดาห์")
    # วันทำการล่าสุดที่ไม่เกินวันที่เลือก
    pick = days[days <= pd.Timestamp(ss["rh_slider"])][-1]

    f = go.Figure()
    rows = []
    for etf, df in rot.items():
        h = df[df.index <= pick]
        if h.empty or (pick - h.index[-1]).days > 10:
            continue  # ETF ยังไม่มีข้อมูล ณ วันนั้น (เช่น XLC ก่อนปี 2018)
        tail = h.iloc[::-5].head(tail_n).iloc[::-1]
        x, y = h["x"].iloc[-1], h["y"].iloc[-1]
        q = _quad(x, y)
        name, color = QUAD[q]
        f.add_trace(go.Scatter(x=tail["x"], y=tail["y"], mode="lines", line=dict(color=color, width=1.5),
                               opacity=0.45, hoverinfo="skip", showlegend=False))
        f.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[ETF_TH[etf]], textposition="top center",
                               marker=dict(size=12, color=color, line=dict(color="white", width=1)),
                               hovertemplate=f"{ETF_TH[etf]} ({etf})<br>ส่วนเกิน 3 เดือน %{{x:+.2f}}%<br>"
                                             "โมเมนตัม %{y:+.2f}<extra></extra>", showlegend=False))
        # หลังจากวันนั้น กลุ่มนี้ชนะ/แพ้ SPY เท่าไร (ดูว่าสัญญาณได้ผลไหม)
        fwd = {}
        for lab, n in [("1 เดือนถัดไป", 21), ("3 เดือนถัดไป", 63)]:
            s, b = px[etf].dropna(), px["SPY"].dropna()
            i = s.index.searchsorted(pick)
            if i + n < len(s):
                fwd[lab] = ((s.iloc[i + n] / s.iloc[i]) - (b.iloc[b.index.searchsorted(pick) + n] /
                                                             b.iloc[b.index.searchsorted(pick)])) * 100
        rows.append({"กลุ่ม": f"{ETF_TH[etf]} ({etf})", "ช่อง": name, "ส่วนเกิน 3 เดือน %": x, "โมเมนตัม": y, **fwd})
    f.add_hline(y=0, line_color="gray")
    f.add_vline(x=0, line_color="gray")
    for tx, ty, k in [(0.98, 0.98, "lead"), (0.98, 0.02, "weak"), (0.02, 0.02, "lag"), (0.02, 0.98, "imp")]:
        f.add_annotation(xref="paper", yref="paper", x=tx, y=ty, text=QUAD[k][0], showarrow=False,
                         xanchor="right" if tx > 0.5 else "left", yanchor="top" if ty > 0.5 else "bottom",
                         font=dict(color=QUAD[k][1], size=12))
    f.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="ผลตอบแทนส่วนเกินเทียบ S&P 500 (3 เดือน, %)", yaxis_title="โมเมนตัม (เปลี่ยนแปลงใน 1 เดือน)")
    g1, g2 = st.columns([1.3, 1])
    with g1:
        st.plotly_chart(f, use_container_width=True)
        st.caption(f"ณ วันที่ {pick:%d %b %Y} · เส้นจาง = เส้นทาง {tail_n} สัปดาห์ก่อนหน้า")
    with g2:
        t = pd.DataFrame(rows).sort_values("ส่วนเกิน 3 เดือน %", ascending=False)
        pc = st.column_config.NumberColumn(format="%+.2f%%")
        st.dataframe(t, hide_index=True, use_container_width=True,
                     column_config={"ส่วนเกิน 3 เดือน %": pc, "โมเมนตัม": st.column_config.NumberColumn(format="%+.2f"),
                                    "1 เดือนถัดไป": pc, "3 เดือนถัดไป": pc})
        st.caption("คอลัมน์ \"เดือนถัดไป\" = หลังจากวันที่เลือก กลุ่มนี้ชนะ (+) หรือแพ้ (−) S&P 500 กี่ % "
                   "ใช้ดูว่าช่องที่อยู่ ณ วันนั้นบอกอนาคตได้แค่ไหน (ว่าง = ยังไม่ถึงเวลานั้น)")

    if st.toggle("▶ แสดงภาพเคลื่อนไหว 52 สัปดาห์ก่อนวันที่เลือก", key="rh_anim"):
        render_rotation_animation(rot, weeks[weeks <= pick][-52:])
    return pick


# ======================= Breadth ย้อนหลัง =======================
US_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files", "us_indices.csv")
SECTOR_TH = {gics: th for gics, (_, th) in SECTORS.items()}


@st.cache_data(ttl=6 * 3600, show_spinner="กำลังดึงราคาหุ้น S&P 500 ย้อนหลัง 3 ปี (ครั้งแรกประมาณ 30 วินาที) ...")
def breadth_history() -> dict:
    """% หุ้นใน S&P 500 ที่ราคาปิดอยู่เหนือ MA50 / MA200 ของแต่ละวัน ทั้งตลาดและรายกลุ่ม"""
    u = pd.read_csv(US_CSV, comment="#", encoding="utf-8-sig")
    u = u[u["sp500"] == 1]
    d = yf.download(list(u["yahoo"]), period="3y", interval="1d", progress=False, auto_adjust=True,
                    threads=True)["Close"]
    d.index = pd.DatetimeIndex(d.index.tz_localize(None) if d.index.tz is not None else d.index)
    sec = dict(zip(u["yahoo"], u["sector"]))
    out = {}
    for n in (50, 200):
        ma = d.rolling(n, min_periods=n).mean()
        above = (d > ma).where(ma.notna())  # NaN = ยังมีข้อมูลไม่พอคำนวณ MA
        ok = above.notna().sum(axis=1) > 0.8 * d.shape[1]  # ใช้เฉพาะวันที่หุ้นส่วนใหญ่มี MA แล้ว
        out[f"all{n}"] = (above.mean(axis=1) * 100)[ok]
        by = above.T.groupby(above.columns.map(sec)).mean().T * 100
        out[f"sec{n}"] = by[ok]
    return out


def render_breadth_history(px: pd.DataFrame, pick: pd.Timestamp):
    st.markdown("#### 📶 Breadth ย้อนหลัง (S&P 500)")
    b = breadth_history()
    a50, a200 = b["all50"], b["all200"]
    first, last = a200.index[0], a200.index[-1]
    if pick < first:
        st.info(f"Breadth ย้อนหลังมีข้อมูลตั้งแต่ {first:%d %b %Y} (ต้องใช้ราคา 200 วันก่อนหน้าเพื่อคำนวณ MA200) "
                f"วันที่เลือก {pick:%d %b %Y} เก่ากว่านั้น จึงแสดงกราฟช่วงทั้งหมดแทน")
    at = a200.index[a200.index <= pick]
    at = at[-1] if len(at) else None

    m = st.columns(4)
    if at is not None:
        m[0].metric(f"เหนือ MA50 ณ {at:%d %b %Y}", f"{a50.get(at, float('nan')):.0f}%",
                    f"วันนี้ {a50.iloc[-1]:.0f}%", delta_color="off")
        m[1].metric(f"เหนือ MA200 ณ {at:%d %b %Y}", f"{a200[at]:.0f}%", f"วันนี้ {a200.iloc[-1]:.0f}%",
                    delta_color="off")
        spy = px["SPY"].dropna()
        i = spy.index.searchsorted(at)
        for col, (lab, n) in zip(m[2:], [("S&P 500 ใน 1 เดือนถัดไป", 21), ("S&P 500 ใน 3 เดือนถัดไป", 63)]):
            col.metric(lab, f"{(spy.iloc[i + n] / spy.iloc[i] - 1) * 100:+.1f}%" if i + n < len(spy) else "-")

    # กราฟเส้น % เหนือ MA + SPY แยกกราฟ (ไม่ใช้ 2 แกน)
    f = go.Figure()
    f.add_trace(go.Scatter(x=a50.index, y=a50, name="% เหนือ MA50", line=dict(color="#4292c6", width=1.5)))
    f.add_trace(go.Scatter(x=a200.index, y=a200, name="% เหนือ MA200", line=dict(color="#08519c", width=2.5)))
    f.add_hline(y=50, line_dash="dot", line_color="gray")
    if at is not None:
        f.add_vline(x=at, line_dash="dash", line_color="#d62728")
    f.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                    legend=dict(orientation="h", y=1.1), yaxis=dict(range=[0, 100], ticksuffix="%"))
    st.plotly_chart(f, use_container_width=True)
    spy = px["SPY"][(px.index >= first) & (px.index <= last)]
    f2 = go.Figure(go.Scatter(x=spy.index, y=spy, name="SPY", line=dict(color="#898781", width=1.5)))
    if at is not None:
        f2.add_vline(x=at, line_dash="dash", line_color="#d62728")
    f2.update_layout(height=170, margin=dict(l=10, r=10, t=25, b=10), title="ราคา S&P 500 (SPY) ช่วงเดียวกัน",
                     hovermode="x unified")
    st.plotly_chart(f2, use_container_width=True)
    st.caption("เส้นประสีแดง = วันที่เลือก (ใช้ตัวเลื่อนเดียวกับ Sector Rotation ด้านบน) · "
               "เหนือ 50% = ขาขึ้นกระจายตัวดี · ถ้าดัชนีทำจุดสูงใหม่แต่ breadth ลดลง = ขึ้นด้วยหุ้นไม่กี่ตัว (divergence)")

    # Heatmap รายกลุ่ม รายสัปดาห์
    s200 = b["sec200"]
    wk = s200.iloc[::-5].iloc[::-1]
    order = s200.iloc[-1].sort_values().index
    f3 = go.Figure(go.Heatmap(
        z=wk[order].T.values, x=wk.index, y=[SECTOR_TH.get(s, s) for s in order], zmin=0, zmax=100,
        colorscale=[[0, "#b2182b"], [0.5, "#f2f2f2"], [1, "#1a7837"]],
        colorbar=dict(title="%", ticksuffix="%", thickness=12),
        hovertemplate="%{y}<br>%{x|%d %b %Y}<br>เหนือ MA200 %{z:.0f}%<extra></extra>"))
    if at is not None:
        f3.add_vline(x=at, line_dash="dash", line_color="#111")
    f3.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), title="% หุ้นเหนือ MA200 รายกลุ่ม (รายสัปดาห์)")
    st.plotly_chart(f3, use_container_width=True)

    if at is not None:
        snap = pd.DataFrame({"กลุ่ม": [SECTOR_TH.get(s, s) for s in s200.columns],
                             "เหนือ MA50 %": b["sec50"].loc[at].values if at in b["sec50"].index else np.nan,
                             "เหนือ MA200 %": s200.loc[at].values,
                             "เหนือ MA200 วันนี้ %": s200.iloc[-1].values}).sort_values("เหนือ MA200 %", ascending=False)
        snap["เปลี่ยนจากวันนั้น"] = snap["เหนือ MA200 วันนี้ %"] - snap["เหนือ MA200 %"]
        pc = st.column_config.NumberColumn(format="%.0f%%")
        st.dataframe(snap, hide_index=True, use_container_width=True, column_config={
            "เหนือ MA50 %": pc, "เหนือ MA200 %": pc, "เหนือ MA200 วันนี้ %": pc,
            "เปลี่ยนจากวันนั้น": st.column_config.NumberColumn(format="%+.0f จุด")})
    st.caption(f"ข้อมูล {first:%d %b %Y} – {last:%d %b %Y} · ราคาปรับปันผลแล้ว (ตัวเลขจึงต่างจากแท็บ Heatmap เล็กน้อย) · "
               "⚠️ ใช้รายชื่อหุ้น S&P 500 ปัจจุบัน ไม่ใช่รายชื่อ ณ วันนั้น (survivorship bias) "
               "หุ้นที่ถูกถอดออกไปแล้วจึงไม่ถูกนับ")


def render_rotation_animation(rot: dict, wk: pd.DatetimeIndex):
    if len(wk) < 2:
        st.caption("ข้อมูลไม่พอ")
        return
    etfs = [e for e, df in rot.items() if not df[df.index <= wk[0]].empty]

    def pts(d):
        xs, ys, cs = [], [], []
        for e in etfs:
            h = rot[e][rot[e].index <= d]
            x, y = h["x"].iloc[-1], h["y"].iloc[-1]
            xs.append(x); ys.append(y); cs.append(QUAD[_quad(x, y)][1])
        return xs, ys, cs

    allx = pd.concat([rot[e]["x"][(rot[e].index >= wk[0]) & (rot[e].index <= wk[-1])] for e in etfs])
    ally = pd.concat([rot[e]["y"][(rot[e].index >= wk[0]) & (rot[e].index <= wk[-1])] for e in etfs])
    lim_x, lim_y = max(5, allx.abs().max() * 1.1), max(5, ally.abs().max() * 1.1)
    x0, y0, c0 = pts(wk[0])
    labels = [ETF_TH[e] for e in etfs]
    frames = []
    for d in wk:
        x, y, c = pts(d)
        frames.append(go.Frame(name=d.strftime("%Y-%m-%d"),
                               data=[go.Scatter(x=x, y=y, mode="markers+text", text=labels, textposition="top center",
                                                marker=dict(size=13, color=c, line=dict(color="white", width=1)))],
                               layout=go.Layout(title_text=f"สัปดาห์ {d:%d %b %Y}")))
    f = go.Figure(data=[go.Scatter(x=x0, y=y0, mode="markers+text", text=labels, textposition="top center",
                                   marker=dict(size=13, color=c0, line=dict(color="white", width=1)),
                                   hovertemplate="%{text}<br>x %{x:+.2f}%<br>y %{y:+.2f}<extra></extra>")],
                  frames=frames)
    f.add_hline(y=0, line_color="gray")
    f.add_vline(x=0, line_color="gray")
    f.update_layout(
        height=520, margin=dict(l=10, r=10, t=50, b=10), title_text=f"สัปดาห์ {wk[0]:%d %b %Y}",
        xaxis=dict(range=[-lim_x, lim_x], title="ส่วนเกิน 3 เดือน (%)"),
        yaxis=dict(range=[-lim_y, lim_y], title="โมเมนตัม"),
        updatemenus=[dict(type="buttons", x=0, y=1.12, xanchor="left", direction="left", buttons=[
            dict(label="▶ เล่น", method="animate",
                 args=[None, {"frame": {"duration": 350, "redraw": True}, "fromcurrent": True,
                              "transition": {"duration": 200}}]),
            dict(label="⏸ หยุด", method="animate",
                 args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])],
        sliders=[dict(active=0, currentvalue=dict(prefix="สัปดาห์: "), pad=dict(t=30), steps=[
            dict(method="animate", label=pd.Timestamp(fr.name).strftime("%d %b %y"), args=[[fr.name], {"frame": {"duration": 0, "redraw": True},
                                                                        "mode": "immediate"}])
            for fr in frames])])
    st.plotly_chart(f, use_container_width=True)


# ======================= 2) Backtest กลยุทธ์ A =======================
def backtest_momentum(px: pd.DataFrame, top_n: int, lb_months: int, start: pd.Timestamp, cost_bps: float):
    """รีบาลานซ์ทุกสิ้นเดือน: เลือก top_n กลุ่มที่ผลตอบแทน lb_months เดือนย้อนหลังดีที่สุด ถือเท่ากันเดือนถัดไป
    ใช้ราคาปิดสิ้นเดือนที่รู้แล้ว ณ วันรีบาลานซ์เท่านั้น (ไม่มองอนาคต)"""
    m = px.resample("ME").last()
    m = m[m.index >= start - pd.DateOffset(months=lb_months + 1)]
    ret = m.pct_change()
    look = m / m.shift(lb_months) - 1
    rows, prev, eq, bench = [], set(), 1.0, 1.0
    dates = [d for d in m.index if d >= start]
    for i in range(len(dates) - 1):
        d, nxt = dates[i], dates[i + 1]
        cand = look.loc[d, list(ETF_TH)].dropna()
        cand = cand[[e for e in cand.index if pd.notna(m.loc[d, e])]]
        if len(cand) < top_n:
            continue
        hold = list(cand.sort_values(ascending=False).index[:top_n])
        turnover = len(set(hold) ^ prev) / (2 * top_n) if prev else 1.0
        r = ret.loc[nxt, hold].mean() - turnover * cost_bps / 1e4
        rb = ret.loc[nxt, "SPY"]
        eq *= 1 + r
        bench *= 1 + rb
        rows.append({"เดือน": nxt, "ถือ": hold, "เปลี่ยน": sorted(set(hold) - prev) if prev else hold,
                     "กลยุทธ์ %": r * 100, "S&P 500 %": rb * 100, "eq": eq, "bench": bench})
        prev = set(hold)
    return pd.DataFrame(rows)


def _stats(eq: pd.Series, mret: pd.Series):
    yrs = len(mret) / 12
    dd = eq / eq.cummax() - 1
    return {
        "ผลตอบแทนรวม": (eq.iloc[-1] - 1) * 100,
        "เฉลี่ยต่อปี (CAGR)": (eq.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else np.nan,
        "ผันผวนต่อปี": mret.std() * np.sqrt(12),
        "ลงหนักสุด (Max DD)": dd.min() * 100,
        "Sharpe (rf=0)": (mret.mean() / mret.std() * np.sqrt(12)) if mret.std() else np.nan,
    }


def render_backtest(px: pd.DataFrame):
    st.markdown("#### 🧪 Backtest กลยุทธ์ A: Sector Momentum")
    st.caption("กติกา: ทุกสิ้นเดือน ดูผลตอบแทนย้อนหลังของ 11 กลุ่ม → ซื้อ N กลุ่มที่ดีที่สุด ลงเงินเท่ากัน → ถือ 1 เดือน → ทำซ้ำ")
    first = px["SPY"].dropna().index[0]
    c = st.columns(4)
    top_n = c[0].selectbox("จำนวนกลุ่มที่ถือ (N)", [1, 2, 3, 4, 5], index=2, key="bt_n")
    lb = c[1].selectbox("ดูผลตอบแทนย้อนหลัง", [1, 3, 6, 12], index=1, key="bt_lb", format_func=lambda v: f"{v} เดือน")
    start = c[2].date_input("เริ่มทดสอบ", value=dt.date(2010, 1, 1), min_value=(first + pd.DateOffset(months=13)).date(),
                            max_value=dt.date.today() - dt.timedelta(days=90), key="bt_start")
    cost = c[3].number_input("ค่าซื้อขาย (bps ต่อรอบ)", 0.0, 100.0, 10.0, 5.0, key="bt_cost",
                             help="10 bps = 0.10% คิดตามสัดส่วนที่เปลี่ยนกลุ่มในแต่ละเดือน")
    bt = backtest_momentum(px, top_n, lb, pd.Timestamp(start), cost)
    if bt.empty:
        st.warning("ข้อมูลไม่พอสำหรับช่วงที่เลือก")
        return
    s = _stats(bt.set_index("เดือน")["eq"], bt["กลยุทธ์ %"] / 100)
    b = _stats(bt.set_index("เดือน")["bench"], bt["S&P 500 %"] / 100)
    win = (bt["กลยุทธ์ %"] > bt["S&P 500 %"]).mean() * 100
    switches = int((bt["เปลี่ยน"].map(len) > 0).sum() - 1)

    k = st.columns(5)
    k[0].metric("ผลตอบแทนรวม", f"{s['ผลตอบแทนรวม']:+,.0f}%", f"S&P 500 {b['ผลตอบแทนรวม']:+,.0f}%", delta_color="off")
    k[1].metric("เฉลี่ยต่อปี (CAGR)", f"{s['เฉลี่ยต่อปี (CAGR)']:+.2f}%",
                f"{s['เฉลี่ยต่อปี (CAGR)'] - b['เฉลี่ยต่อปี (CAGR)']:+.2f}% เทียบ S&P 500")
    k[2].metric("ลงหนักสุด (Max DD)", f"{s['ลงหนักสุด (Max DD)']:.1f}%", f"S&P 500 {b['ลงหนักสุด (Max DD)']:.1f}%",
                delta_color="off")
    k[3].metric("Sharpe", f"{s['Sharpe (rf=0)']:.2f}", f"S&P 500 {b['Sharpe (rf=0)']:.2f}", delta_color="off")
    k[4].metric("เดือนที่ชนะตลาด", f"{win:.0f}%", f"เปลี่ยนกลุ่ม {switches} ครั้ง", delta_color="off")

    log = st.toggle("แกนตั้งแบบ log (เห็นช่วงต้นชัดขึ้น)", value=True, key="bt_log")
    f = go.Figure()
    f.add_trace(go.Scatter(x=bt["เดือน"], y=bt["eq"] * 100, name=f"กลยุทธ์ A (Top {top_n}, {lb} เดือน)",
                           line=dict(color="#2a78d6", width=2)))
    f.add_trace(go.Scatter(x=bt["เดือน"], y=bt["bench"] * 100, name="S&P 500 (SPY)",
                           line=dict(color="#898781", width=2, dash="dash")))
    f.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                    legend=dict(orientation="h", y=1.08), yaxis=dict(title="เงินลงทุน (เริ่ม 100)",
                                                                     type="log" if log else "linear"))
    st.plotly_chart(f, use_container_width=True)

    dd_s = bt["eq"] / bt["eq"].cummax() - 1
    dd_b = bt["bench"] / bt["bench"].cummax() - 1
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=bt["เดือน"], y=dd_s * 100, name="กลยุทธ์ A", fill="tozeroy",
                            line=dict(color="#d73027", width=1.5)))
    f2.add_trace(go.Scatter(x=bt["เดือน"], y=dd_b * 100, name="S&P 500", line=dict(color="#898781", width=1.5, dash="dash")))
    f2.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), title="ระยะห่างจากจุดสูงสุด (Drawdown)",
                     yaxis=dict(ticksuffix="%"), legend=dict(orientation="h", y=1.15), hovermode="x unified")
    st.plotly_chart(f2, use_container_width=True)

    y1, y2 = st.columns([1, 1.3])
    with y1:
        st.markdown("**📅 ผลตอบแทนรายปี**")
        yr = bt.assign(ปี=bt["เดือน"].dt.year).groupby("ปี").apply(
            lambda g: pd.Series({"กลยุทธ์ %": ((1 + g["กลยุทธ์ %"] / 100).prod() - 1) * 100,
                                 "S&P 500 %": ((1 + g["S&P 500 %"] / 100).prod() - 1) * 100}),
            include_groups=False).reset_index()
        yr["ส่วนต่าง"] = yr["กลยุทธ์ %"] - yr["S&P 500 %"]
        pc = st.column_config.NumberColumn(format="%+.1f%%")
        st.dataframe(yr.sort_values("ปี", ascending=False), hide_index=True, use_container_width=True, height=360,
                     column_config={"ปี": st.column_config.NumberColumn(format="%d"),
                                    "กลยุทธ์ %": pc, "S&P 500 %": pc, "ส่วนต่าง": pc})
    with y2:
        st.markdown("**🗂️ กลุ่มที่ถือในแต่ละเดือน (ล่าสุดก่อน)**")
        h = bt[["เดือน", "ถือ", "กลยุทธ์ %", "S&P 500 %"]].copy()
        h["เดือน"] = h["เดือน"].dt.strftime("%b %Y")
        h["ถือ"] = h["ถือ"].map(lambda xs: ", ".join(f"{ETF_TH[x]} ({x})" for x in xs))
        st.dataframe(h.iloc[::-1], hide_index=True, use_container_width=True, height=360,
                     column_config={"กลยุทธ์ %": st.column_config.NumberColumn(format="%+.2f%%"),
                                    "S&P 500 %": st.column_config.NumberColumn(format="%+.2f%%")})
    nxt = look_now(px, top_n, lb)
    st.info(f"ถ้าใช้กติกานี้วันนี้ จะถือ: **{', '.join(f'{ETF_TH[e]} ({e})' for e in nxt)}** "
            f"(ข้อมูลถึง {px.index[-1]:%d %b %Y} · กลยุทธ์จริงรีบาลานซ์สิ้นเดือน)")
    st.caption("⚠️ ผลในอดีตไม่รับประกันอนาคต · ใช้ราคาปรับปันผลแล้ว (รวมปันผล) · ยังไม่รวมภาษีและสเปรด · "
               "กลุ่มสื่อสาร (XLC) เริ่มปี 2018 และอสังหาฯ (XLRE) เริ่มปี 2015 ก่อนหน้านั้นเลือกจาก 9 กลุ่ม · "
               "การปรับตัวเลขจนผลออกมาสวยที่สุดมักทำให้ผลดีเกินจริง (overfitting) · เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน")


def look_now(px: pd.DataFrame, top_n: int, lb_months: int):
    d = px.index[-1]
    past = px[px.index <= d - pd.DateOffset(months=lb_months)].iloc[-1]
    r = (px.iloc[-1][list(ETF_TH)] / past[list(ETF_TH)] - 1).dropna()
    return list(r.sort_values(ascending=False).index[:top_n])


def render_history_tab():
    px = long_prices()
    st.caption(f"ข้อมูล {px.index[0]:%Y}–{px.index[-1]:%d %b %Y} · SPDR Select Sector ETF + SPY · ราคาปรับปันผล (Yahoo Finance)")
    pick = render_rotation_history(px)
    st.divider()
    if st.toggle("📶 แสดง Breadth ย้อนหลัง (S&P 500 · โหลดครั้งแรกประมาณ 30 วินาที)", value=True, key="bh_on"):
        render_breadth_history(px, pick)
    st.divider()
    render_backtest(px)
