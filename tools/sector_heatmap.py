"""หน้า Heatmap กลุ่มอุตสาหกรรมหุ้นสหรัฐฯ (ใช้ใน stock_dashboard.py)
- ตารางผลตอบแทน sector ETF (SPDR) หลายช่วงเวลา
- Treemap S&P 500 แบบ Finviz: sector -> หุ้น, ขนาด = มูลค่าตลาด, สี = % เปลี่ยนแปลง
- Breadth: % หุ้นที่ขึ้น / อยู่เหนือ MA50 / MA200 ในแต่ละ sector
- Sector rotation: ความแข็งแกร่งเทียบ S&P 500 กับโมเมนตัม (คล้าย RRG)
"""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# GICS sector (ตามไฟล์ us_indices.csv) -> (ETF, ชื่อไทย)
SECTORS = {
    "Information Technology": ("XLK", "เทคโนโลยี"),
    "Communication Services": ("XLC", "สื่อสาร"),
    "Consumer Discretionary": ("XLY", "สินค้าฟุ่มเฟือย"),
    "Financials": ("XLF", "การเงิน"),
    "Health Care": ("XLV", "การแพทย์"),
    "Industrials": ("XLI", "อุตสาหกรรม"),
    "Consumer Staples": ("XLP", "สินค้าจำเป็น"),
    "Energy": ("XLE", "พลังงาน"),
    "Materials": ("XLB", "วัตถุดิบ"),
    "Real Estate": ("XLRE", "อสังหาฯ"),
    "Utilities": ("XLU", "สาธารณูปโภค"),
}
PERIODS = {"1D": 1, "5D": 5, "1M": 21, "3M": 63, "6M": 126, "YTD": None, "1Y": 252}
FINVIZ = [[0, "#f63538"], [0.5, "#414554"], [1, "#30cc5a"]]
DIVERGE = [[0, "#b2182b"], [0.5, "#f2f2f2"], [1, "#1a7837"]]


@st.cache_data(ttl=300, show_spinner="กำลังดึงข้อมูล sector ETF ...")
def sector_prices() -> pd.DataFrame:
    syms = [v[0] for v in SECTORS.values()] + ["SPY"]
    d = yf.download(syms, period="2y", interval="1d", progress=False, auto_adjust=True)["Close"]
    d.index = pd.DatetimeIndex(d.index.tz_localize(None) if d.index.tz is not None else d.index)
    return d.dropna(how="all")


def _ret(s: pd.Series, p: str):
    s = s.dropna()
    if p == "YTD":
        b = s[s.index < pd.Timestamp(s.index[-1].year, 1, 1)]
        return (s.iloc[-1] / b.iloc[-1] - 1) * 100 if len(b) else None
    n = PERIODS[p]
    return (s.iloc[-1] / s.iloc[-1 - n] - 1) * 100 if len(s) > n else None


@st.cache_data(ttl=24 * 3600, show_spinner="กำลังดึงมูลค่าตลาดหุ้น S&P 500 (ครั้งแรกประมาณ 20-30 วินาที) ...")
def market_caps(syms: tuple) -> dict:
    def cap(x):
        try:
            return x, float(yf.Ticker(x).fast_info["market_cap"])
        except Exception:
            return x, None
    with ThreadPoolExecutor(16) as ex:
        return dict(ex.map(cap, syms))


def render_sector_heatmap(stocks: pd.DataFrame):
    """stocks: ข้อมูลหุ้น S&P 500 จาก get_bulk + symbol/name/sector"""
    stocks = stocks.copy()
    for c in ("เหนือ MA50", "เหนือ MA200"):  # cache รุ่นเก่าอาจยังไม่มีคอลัมน์นี้
        if c not in stocks:
            stocks[c] = None
    px = sector_prices()
    st.caption(f"ข้อมูลวันที่ {px.index[-1].date()} · ราคาจาก Yahoo Finance · sector ตามมาตรฐาน GICS "
               "· ETF: SPDR Select Sector (XLK, XLF ...)")

    # ---------- 1) ตารางผลตอบแทน sector ----------
    st.markdown("#### 📊 ผลตอบแทนรายกลุ่มอุตสาหกรรม")
    rel = st.toggle("แสดงเป็นผลตอบแทนส่วนเกินเทียบ S&P 500 (ชนะ/แพ้ตลาดกี่ %)", key="sec_rel")
    rows, labels = [], []
    spy = {p: _ret(px["SPY"], p) for p in PERIODS}
    for sec, (etf, th) in SECTORS.items():
        r = [_ret(px[etf], p) for p in PERIODS]
        if rel:
            r = [None if v is None or spy[p] is None else v - spy[p] for v, p in zip(r, PERIODS)]
        rows.append(r)
        labels.append(f"{th} ({etf})")
    if not rel:
        rows.append([spy[p] for p in PERIODS])
        labels.append("S&P 500 (SPY)")
    m = pd.DataFrame(rows, index=labels, columns=list(PERIODS))
    # สีปรับตามแต่ละคอลัมน์ (ช่วงยาวผันผวนกว่าช่วงสั้น) ตัวเลขยังเป็นค่าจริง
    scale = m.abs().max().replace(0, 1)
    f = go.Figure(go.Heatmap(
        z=(m / scale).values, x=m.columns, y=m.index, zmin=-1, zmax=1, zmid=0, colorscale=DIVERGE, showscale=False,
        text=m.map(lambda v: "" if pd.isna(v) else f"{v:+.1f}%").values, texttemplate="%{text}",
        textfont=dict(size=13), customdata=m.values,
        hovertemplate="%{y}<br>%{x}: %{customdata:+.2f}%<extra></extra>", xgap=2, ygap=2))
    f.update_layout(height=40 * len(m) + 60, margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(autorange="reversed"), xaxis=dict(side="top"))
    st.plotly_chart(f, use_container_width=True)
    st.caption("สีเทียบกันภายในคอลัมน์เดียวกัน (เขียวเข้ม = ดีที่สุดในช่วงนั้น แดงเข้ม = แย่ที่สุด)")

    # ---------- 2) Treemap S&P 500 ----------
    st.markdown("#### 🗺️ แผนที่หุ้น S&P 500 (ขนาด = มูลค่าตลาด)")
    c1, c2 = st.columns([1, 2])
    per = c1.radio("สีตามผลตอบแทน", ["1D %", "5D %", "1M %", "YTD %"], horizontal=True, key="tm_per")
    caps = market_caps(tuple(stocks["yahoo"]))
    d = stocks.assign(cap=stocks["yahoo"].map(caps) / 1e9).dropna(subset=["cap", per, "sector"])
    d = d[d["cap"] > 0]
    sec_pick = c2.multiselect("กรองกลุ่ม (ว่าง = ทั้งหมด)", sorted(d["sector"].unique()), key="tm_sec",
                              format_func=lambda s: f"{SECTORS.get(s, ('', s))[1]} · {s}")
    if sec_pick:
        d = d[d["sector"].isin(sec_pick)]
    lim = {"1D %": 3, "5D %": 6, "1M %": 10, "YTD %": 30}[per]
    ids, labs, pars, vals, cols, cd = [], [], [], [], [], []
    for sec, g in d.groupby("sector"):
        ids.append(sec); labs.append(SECTORS.get(sec, ("", sec))[1] + f" · {sec}"); pars.append("")
        vals.append(g["cap"].sum())
        w = (g[per] * g["cap"]).sum() / g["cap"].sum()
        cols.append(w); cd.append([w, g["cap"].sum(), "", len(g)])
    for y_, s_, sec_, cap_, v, nm, pr in zip(d["yahoo"], d["symbol"], d["sector"], d["cap"], d[per],
                                             d["name"], d["ราคา"]):
        ids.append(y_); labs.append(s_); pars.append(sec_); vals.append(cap_)
        cols.append(v); cd.append([v, cap_, nm, pr])
    f = go.Figure(go.Treemap(
        ids=ids, labels=labs, parents=pars, values=vals, branchvalues="total",
        marker=dict(colors=cols, colorscale=FINVIZ, cmin=-lim, cmax=lim, cmid=0, showscale=True,
                    colorbar=dict(title=per, ticksuffix="%", thickness=12), line=dict(width=1, color="#262931")),
        customdata=cd, texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%",
        hovertemplate="<b>%{label}</b> %{customdata[2]}<br>%{customdata[0]:+.2f}%<br>"
                      "มูลค่าตลาด %{customdata[1]:,.0f} พันล้าน $<extra></extra>",
        textfont=dict(color="white"), tiling=dict(pad=2), pathbar=dict(visible=True)))
    f.update_layout(height=720, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="#262931")
    st.plotly_chart(f, use_container_width=True)
    st.caption(f"สีเต็มที่ ±{lim}% · กดที่กลุ่มเพื่อซูมเข้า กดแถบด้านบนเพื่อซูมออก · สีของกลุ่ม = ค่าเฉลี่ยถ่วงน้ำหนักมูลค่าตลาด")

    # ---------- 3) Breadth + 4) Rotation ----------
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("#### 📶 ความกว้างของตลาด (Breadth)")
        br = stocks.dropna(subset=["sector"]).groupby("sector").agg(
            up=("1D %", lambda s: (s > 0).mean() * 100),
            ma50=("เหนือ MA50", lambda s: s.dropna().astype(bool).mean() * 100),
            ma200=("เหนือ MA200", lambda s: s.dropna().astype(bool).mean() * 100),
            n=("symbol", "count")).sort_values("ma50")
        ylab = [SECTORS.get(s, ("", s))[1] for s in br.index]
        f = go.Figure()
        for c, name, color in [("up", "% หุ้นที่ขึ้นวันนี้", "#9ecae1"), ("ma50", "% เหนือ MA50", "#4292c6"),
                               ("ma200", "% เหนือ MA200", "#08519c")]:
            f.add_trace(go.Bar(y=ylab, x=br[c], name=name, orientation="h", marker_color=color,
                               text=br[c].map("{:.0f}%".format), textposition="inside"))
        f.add_vline(x=50, line_dash="dot", line_color="gray")
        f.update_layout(barmode="group", height=560, margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", y=1.06), xaxis=dict(range=[0, 100], ticksuffix="%"))
        st.plotly_chart(f, use_container_width=True)
        allm = stocks["เหนือ MA200"].dropna().astype(bool).mean() * 100
        st.caption(f"ทั้ง S&P 500: หุ้นเหนือ MA200 = {allm:.0f}% · มากกว่า 50% = ขาขึ้นกระจายตัวดี · "
                   "ถ้าดัชนีขึ้นแต่ breadth ต่ำ = ขึ้นเพราะหุ้นใหญ่ไม่กี่ตัว")
    with b2:
        st.markdown("#### 🔄 การหมุนเวียนของกลุ่ม (Sector Rotation)")
        f = go.Figure()
        for sec, (etf, th) in SECTORS.items():
            rs = (px[etf] / px["SPY"]).dropna()
            if len(rs) < 90:
                continue
            # x = ผลตอบแทนส่วนเกิน 3 เดือน, y = x วันนี้ - x เมื่อ 1 เดือนก่อน (เร่งขึ้น/ชะลอลง)
            xr = (rs / rs.shift(63) - 1) * 100
            yr = xr - xr.shift(21)
            # จุดรายสัปดาห์ 6 สัปดาห์ นับถอยหลังจากวันล่าสุด (จุดสุดท้าย = วันนี้)
            tail = pd.DataFrame({"x": xr, "y": yr}).dropna().iloc[::-5].head(6).iloc[::-1]
            if tail.empty:
                continue
            x, y = tail["x"].iloc[-1], tail["y"].iloc[-1]
            color = "#1a9850" if x > 0 and y > 0 else "#fdae61" if x > 0 else "#d73027" if y < 0 else "#4575b4"
            f.add_trace(go.Scatter(x=tail["x"], y=tail["y"], mode="lines", line=dict(color=color, width=1.5),
                                   opacity=0.5, hoverinfo="skip", showlegend=False))
            f.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[th], textposition="top center",
                                   marker=dict(size=12, color=color, line=dict(color="white", width=1)),
                                   hovertemplate=f"{th} ({etf})<br>ส่วนเกิน 3 เดือน %{{x:+.2f}}%<br>"
                                                 "โมเมนตัม 1 เดือน %{y:+.2f}<extra></extra>", showlegend=False))
        f.add_hline(y=0, line_color="gray")
        f.add_vline(x=0, line_color="gray")
        for tx, ty, t, c in [(0.98, 0.98, "🟢 Leading นำตลาด", "#1a9850"), (0.98, 0.02, "🟠 Weakening เริ่มแผ่ว", "#e08214"),
                             (0.02, 0.02, "🔴 Lagging แพ้ตลาด", "#d73027"), (0.02, 0.98, "🔵 Improving เริ่มฟื้น", "#4575b4")]:
            f.add_annotation(xref="paper", yref="paper", x=tx, y=ty, text=t, showarrow=False,
                             xanchor="right" if tx > 0.5 else "left", yanchor="top" if ty > 0.5 else "bottom",
                             font=dict(color=c, size=12))
        f.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="ผลตอบแทนส่วนเกินเทียบ S&P 500 (3 เดือน, %)",
                        yaxis_title="โมเมนตัม (เปลี่ยนแปลงใน 1 เดือน)")
        st.plotly_chart(f, use_container_width=True)
        st.caption("เส้นจาง = เส้นทาง 6 สัปดาห์ล่าสุด · กลุ่มมักหมุนตามเข็มนาฬิกา: "
                   "เริ่มฟื้น → นำตลาด → เริ่มแผ่ว → แพ้ตลาด")

    # ---------- 5) เจาะรายกลุ่ม ----------
    st.markdown("#### 🔎 เจาะดูหุ้นในกลุ่ม")
    k1, k2 = st.columns([2, 1])
    sec = k1.selectbox("เลือกกลุ่ม", list(SECTORS), key="sec_drill",
                       format_func=lambda s: f"{SECTORS[s][1]} · {s} ({SECTORS[s][0]})")
    per2 = k2.radio("เรียงตาม", ["1D %", "5D %", "1M %", "YTD %"], horizontal=True, key="sec_drill_per")
    g = stocks[stocks["sector"] == sec].assign(cap=lambda x: x["yahoo"].map(caps) / 1e9)
    cfg = {"cap": st.column_config.NumberColumn("มูลค่าตลาด (พันล้าน $)", format="%,.0f"),
           "ราคา": st.column_config.NumberColumn(format="%,.2f"),
           **{c: st.column_config.NumberColumn(format="%+.2f%%") for c in ["1D %", "5D %", "1M %", "YTD %"]},
           "name": "บริษัท"}
    cols = ["symbol", "name", "ราคา", "1D %", "5D %", "1M %", "YTD %", "cap"]
    t1, t2 = st.columns(2)
    t1.markdown(f"**🚀 ขึ้นแรงสุด ({per2})**")
    t1.dataframe(g.nlargest(10, per2)[cols], hide_index=True, use_container_width=True, column_config=cfg)
    t2.markdown(f"**🔻 ลงแรงสุด ({per2})**")
    t2.dataframe(g.nsmallest(10, per2)[cols], hide_index=True, use_container_width=True, column_config=cfg)
    st.caption(f"{SECTORS[sec][1]}: {len(g)} หุ้นใน S&P 500 · ดูกราฟ/ข้อมูลพื้นฐานรายตัวได้ที่แท็บ 📈 กราฟราคา และ 🔎 ข้อมูลพื้นฐาน")
