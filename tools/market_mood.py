"""ดัชนีอารมณ์ตลาด (Market Mood Index) – ต้นแบบเฟส 1 · ตลาดสหรัฐฯ
รวมตัวชี้วัด 7 ตัวเป็นคะแนน 0 (กลัวสุดขีด) – 100 (โลภสุดขีด) คล้าย CNN Fear & Greed แต่คำนวณเองจากข้อมูลฟรี

วิธีคำนวณ (ตั้งใจให้เรียบง่าย ไม่จูนตัวเลขกับผลในอดีต เพื่อลด overfitting):
1. แต่ละตัวชี้วัด -> percentile rank เทียบกับ 252 วันทำการก่อนหน้า (1 ปี) -> 0–100 (กลับทิศให้ "สูง = โลภ" ทุกตัว)
2. คะแนนรวม = ค่าเฉลี่ยแบบน้ำหนักเท่ากันของตัวที่มีข้อมูล (ต้องมีอย่างน้อย 5 จาก 7 ตัว)
3. โซน: 0–20 กลัวสุดขีด · 20–40 กลัว · 40–60 กลางๆ · 60–80 โลภ · 80–100 โลภสุดขีด
"""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

SECTOR_ETFS = ["XLK", "XLC", "XLY", "XLF", "XLV", "XLI", "XLP", "XLE", "XLB", "XLRE", "XLU"]
DEFENSIVE, CYCLICAL = ["XLP", "XLU", "XLV"], ["XLY", "XLK", "XLI"]
WINDOW = 252  # percentile เทียบย้อนหลัง 1 ปี

COMPONENTS = {  # key: (ชื่อ, คำอธิบาย)
    "vix": ("ความผันผวน (VIX)", "VIX สูงเมื่อเทียบกับ 1 ปีที่ผ่านมา = นักลงทุนกลัว"),
    "term": ("โครงสร้าง VIX (VIX ÷ VIX3M)", "ความผันผวนระยะสั้นสูงกว่าระยะ 3 เดือน = ตื่นตระหนกเฉียบพลัน"),
    "trend": ("แนวโน้ม (S&P 500 vs MA125)", "ดัชนีอยู่ไกลเหนือเส้นค่าเฉลี่ย 125 วัน = โลภ / ต่ำกว่ามาก = กลัว"),
    "breadth": ("ความกว้างกลุ่ม (% กลุ่มเหนือ MA50)", "กลุ่มอุตสาหกรรมส่วนใหญ่อยู่เหนือเส้น 50 วัน = ตลาดขึ้นกระจายตัว"),
    "haven": ("หนีเข้าที่ปลอดภัย (หุ้น vs พันธบัตร)", "ผลตอบแทน 20 วัน SPY ลบ TLT: พันธบัตรชนะหุ้น = กลัว"),
    "credit": ("ความเสี่ยงสินเชื่อ (HYG vs IEF)", "ผลตอบแทน 20 วัน หุ้นกู้ขยะลบพันธบัตรรัฐ: หุ้นกู้ขยะอ่อน = กลัว"),
    "rotation": ("กลุ่มเสี่ยง vs กลุ่มป้องกัน", "ผลตอบแทน 20 วัน XLY/XLK/XLI ลบ XLP/XLU/XLV: กลุ่มป้องกันชนะ = กลัว"),
}
ZONES = [(0, 20, "😱 กลัวสุดขีด", "#b2182b"), (20, 40, "😟 กลัว", "#ef8a62"), (40, 60, "😐 กลางๆ", "#bdbdbd"),
         (60, 80, "😊 โลภ", "#7fbf7b"), (80, 100.01, "🤑 โลภสุดขีด", "#1b7837")]


def zone_of(v):
    for lo, hi, name, color in ZONES:
        if lo <= v < hi:
            return name, color
    return "-", "#999"


@st.cache_data(ttl=6 * 3600, show_spinner="กำลังดึงข้อมูลสำหรับดัชนีอารมณ์ตลาด ...")
def mood_data() -> pd.DataFrame:
    syms = ["^VIX", "^VIX3M", "SPY", "TLT", "HYG", "IEF"] + SECTOR_ETFS
    d = yf.download(syms, start="2006-01-01", interval="1d", progress=False, auto_adjust=True)["Close"]
    d.index = pd.DatetimeIndex(d.index.tz_localize(None) if d.index.tz is not None else d.index)
    return d.dropna(subset=["SPY"])


def _pct_rank(s: pd.Series) -> pd.Series:
    """percentile ของค่าวันนี้เทียบ 252 วันก่อนหน้า (0–100) ใช้ข้อมูลถึงวันนั้นเท่านั้น ไม่มองอนาคต"""
    return s.rolling(WINDOW, min_periods=WINDOW // 2).rank(pct=True) * 100


def compute_mood(d: pd.DataFrame) -> pd.DataFrame:
    spy = d["SPY"]
    raw = pd.DataFrame(index=d.index)
    raw["vix"] = d["^VIX"]
    raw["term"] = d["^VIX"] / d["^VIX3M"]
    raw["trend"] = spy / spy.rolling(125).mean() - 1
    sec = d[SECTOR_ETFS]
    above = (sec > sec.rolling(50).mean()).where(sec.rolling(50).mean().notna())
    raw["breadth"] = above.mean(axis=1)  # สัดส่วนกลุ่มที่อยู่เหนือ MA50 (เฉพาะกลุ่มที่มีข้อมูล ณ วันนั้น)
    r20 = d.pct_change(20, fill_method=None)
    raw["haven"] = r20["SPY"] - r20["TLT"]
    raw["credit"] = r20["HYG"] - r20["IEF"]
    raw["rotation"] = r20[CYCLICAL].mean(axis=1) - r20[DEFENSIVE].mean(axis=1)

    sc = pd.DataFrame(index=d.index)
    for k in COMPONENTS:
        p = _pct_rank(raw[k])
        sc[k] = 100 - p if k in ("vix", "term") else p  # VIX สูง = กลัว จึงกลับทิศ
    n = sc.notna().sum(axis=1)
    sc["score"] = sc[list(COMPONENTS)].mean(axis=1).where(n >= 5)
    sc["spy"] = spy
    for lab, h in [("fwd1m", 21), ("fwd3m", 63), ("fwd6m", 126)]:
        sc[lab] = (spy.shift(-h) / spy - 1) * 100  # ใช้ประเมินผลเท่านั้น ไม่ได้ใช้คำนวณคะแนน
    return sc.dropna(subset=["score"]), raw


def zone_stats(sc: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    s = sc.copy()
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index <= end]
    rows = []
    for lo, hi, name, _ in ZONES + [(None, None, "ทุกวัน (ฐานเปรียบเทียบ)", None)]:
        g = s if lo is None else s[(s["score"] >= lo) & (s["score"] < hi)]
        r = {"โซน": name, "จำนวนวัน": len(g), "% ของเวลา": len(g) / len(s) * 100 if len(s) else np.nan}
        for lab, col in [("1 เดือน", "fwd1m"), ("3 เดือน", "fwd3m"), ("6 เดือน", "fwd6m")]:
            x = g[col].dropna()
            r[f"เฉลี่ย {lab}"] = x.mean() if len(x) else np.nan
            r[f"ขึ้น {lab} (% ครั้ง)"] = (x > 0).mean() * 100 if len(x) else np.nan
        x = g["fwd3m"].dropna()
        r["แย่สุด 3 เดือน"] = x.min() if len(x) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ======================= สัญญาณเตือนล่วงหน้า: เครดิต + breadth แย่ ขณะดัชนีใกล้จุดสูงสุด =======================
# กติกาตั้งไว้ก่อนทดสอบ (ไม่จูน): ใช้กลุ่ม SPDR 9 กลุ่มดั้งเดิม เพื่อให้ตรงกับ backtest ย้อนหลังถึงปี 1999
WARN_SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
WARN_GAP = 63  # นับเป็นสัญญาณใหม่เมื่อห่างจากครั้งก่อนอย่างน้อย 63 วันทำการ (~3 เดือน)
WARN_TEST = pd.DataFrame([  # ผล backtest S&P 500 ปี 1995–2026 (ใช้ VWEHX/VFITX แทน HYG/IEF ช่วงก่อน 2007)
    {"สัญญาณ": "ทุกวัน (ฐานเปรียบเทียบ)", "จำนวนครั้ง": None, "ผลตอบแทน 3 เดือนถัดไป": 2.5, "โอกาสโดนลง ≥10% ใน 3 เดือน": 15},
    {"สัญญาณ": "🚨 เครดิต + breadth แย่ ใกล้จุดสูงสุด", "จำนวนครั้ง": 6, "ผลตอบแทน 3 เดือนถัดไป": 0.6, "โอกาสโดนลง ≥10% ใน 3 เดือน": 33},
    {"สัญญาณ": "โลภสุดขีด (สไตล์ Bull & Bear > 8)", "จำนวนครั้ง": 45, "ผลตอบแทน 3 เดือนถัดไป": 2.4, "โอกาสโดนลง ≥10% ใน 3 เดือน": 4},
    {"สัญญาณ": "กลัวสุดขีด (< 2)", "จำนวนครั้ง": 40, "ผลตอบแทน 3 เดือนถัดไป": 2.8, "โอกาสโดนลง ≥10% ใน 3 เดือน": 22},
])


def divergence_warning(d: pd.DataFrame) -> pd.DataFrame:
    """เตือนเมื่อ (1) หุ้นกู้ขยะอ่อนกว่าพันธบัตรรัฐ ผิดปกติ (2) กลุ่มอุตสาหกรรมไม่เกินครึ่งอยู่เหนือ MA200
    แต่ (3) S&P 500 ยังห่างจุดสูงสุด 52 สัปดาห์ไม่เกิน 5% = ดัชนีดูดีแต่ข้างในเริ่มผุ"""
    spy = d["SPY"]
    ratio = d["HYG"] / d["IEF"]
    cred_mom = ratio / ratio.rolling(63).mean() - 1
    cred_pct = cred_mom.rolling(504, min_periods=252).rank(pct=True) * 100
    sec = d[WARN_SECTORS]
    ma = sec.rolling(200).mean()
    breadth = (sec > ma).where(ma.notna()).mean(axis=1) * 100
    off_high = (spy / spy.rolling(252).max() - 1) * 100
    w = pd.DataFrame({"credit_pct": cred_pct, "breadth200": breadth, "off_high": off_high})
    w["c_credit"] = w["credit_pct"] < 20
    w["c_breadth"] = w["breadth200"] <= 50
    w["c_high"] = w["off_high"] >= -5
    w["warn"] = w["c_credit"] & w["c_breadth"] & w["c_high"]
    onset, last = [], None
    for i, (t, v) in enumerate(w["warn"].items()):
        new = bool(v) and (last is None or i - last >= WARN_GAP)
        onset.append(new)
        if new:
            last = i
    w["onset"] = onset
    return w.dropna(subset=["credit_pct", "breadth200"])


def render_warning(w: pd.DataFrame, at):
    st.markdown("#### 🚨 สัญญาณเตือนล่วงหน้า: ข้างในตลาดเริ่มผุ ขณะดัชนียังดูดี")
    r = w.loc[w.index[w.index <= at][-1]]
    on = bool(r["warn"])
    recent = w[(w.index <= at) & w["onset"]]
    last_on = recent.index[-1] if len(recent) else None
    cols = st.columns([1.3, 1, 1, 1])
    with cols[0]:
        if on:
            st.error("🚨 **สัญญาณเตือนทำงาน** ครบทั้ง 3 เงื่อนไข")
        else:
            st.success(f"✅ ไม่มีสัญญาณเตือน (ครบ {int(r[['c_credit', 'c_breadth', 'c_high']].sum())}/3 เงื่อนไข)")
        st.caption("ครั้งล่าสุด: " + (f"{last_on:%d %b %Y}" if last_on is not None else "ไม่มีในช่วงข้อมูล"))
    for col, ok, lab, val, rule in [
        (cols[1], r["c_credit"], "เครดิต (HYG vs IEF)", f"{r['credit_pct']:.0f}", "percentile · เตือนเมื่อ < 20"),
        (cols[2], r["c_breadth"], "% กลุ่มเหนือ MA200", f"{r['breadth200']:.0f}%", "เตือนเมื่อ ≤ 50%"),
        (cols[3], r["c_high"], "ห่างจุดสูงสุด 52 สัปดาห์", f"{r['off_high']:.1f}%", "เข้าเงื่อนไขเมื่อไม่เกิน −5%"),
    ]:
        col.metric(("🔴 " if ok else "⚪ ") + lab, val, rule, delta_color="off")
    with st.expander("ผลทดสอบย้อนหลังและข้อควรระวัง"):
        st.dataframe(WARN_TEST, hide_index=True, use_container_width=True, column_config={
            "ผลตอบแทน 3 เดือนถัดไป": st.column_config.NumberColumn(format="%+.1f%%"),
            "โอกาสโดนลง ≥10% ใน 3 เดือน": st.column_config.NumberColumn(format="%.0f%%")})
        st.markdown(
            "- ทดสอบ S&P 500 ปี 1995–2026: เตือนก่อนยอดปี **2000 (11 วัน)** และ **2018 (62 วัน)** "
            "แต่ **ไม่เตือน** รอบ 2007, 2020, 2022, 2025\n"
            "- เกิดแค่ **6 ครั้งใน 30 ปี** จำนวนน้อยเกินจะเชื่อได้เต็มที่ ใช้เป็นธงเตือนให้ระวัง ไม่ใช่สัญญาณขายทันที\n"
            "- ถ้าขายออกตามสัญญาณนี้อย่างเดียว การขาดทุนสูงสุดแทบไม่ลด (−55%) ที่ลดได้จริงคือกฎ MA200 (−23%) "
            "เมื่อใช้คู่กัน Sharpe ดีขึ้นเล็กน้อย 0.83 → 0.87\n"
            "- ตัวชี้วัดจากผลสำรวจผู้จัดการกองทุน (FMS Cash) และกระแสเงินกองทุน (Fund Flow) เป็นข้อมูลเสียเงิน จึงไม่ได้รวม")
    return w[w["onset"]].index


# ======================= UI =======================
def render_mood_tab():
    d = mood_data()
    sc, raw = compute_mood(d)
    warn = divergence_warning(d)
    last = sc.index[-1]
    st.caption(f"ต้นแบบเฟส 1 · ตลาดสหรัฐฯ · ข้อมูล {sc.index[0]:%b %Y} – {last:%d %b %Y} · "
               "คะแนน 0 = กลัวสุดขีด, 100 = โลภสุดขีด · ตัวชี้วัด 7 ตัว น้ำหนักเท่ากัน")

    ss = st.session_state
    lo_d, hi_d = sc.index[0].date(), last.date()
    if not isinstance(ss.get("mood_date"), dt.date) or not (lo_d <= ss["mood_date"] <= hi_d):
        ss["mood_date"] = hi_d
    c1, c2 = st.columns([1, 3])
    c1.date_input("ดูคะแนน ณ วันที่", min_value=lo_d, max_value=hi_d, key="mood_date", format="DD/MM/YYYY")
    at = sc.index[sc.index <= pd.Timestamp(ss["mood_date"])][-1]
    row = sc.loc[at]
    zname, zcolor = zone_of(row["score"])

    def ago(n):
        i = sc.index.get_loc(at)
        return sc["score"].iloc[i - n] if i - n >= 0 else np.nan

    # ----- เกจ + ส่วนประกอบ -----
    g1, g2 = st.columns([1, 1.3])
    with g1:
        f = go.Figure(go.Indicator(
            mode="gauge+number", value=row["score"], number=dict(valueformat=".0f", font=dict(size=56)),
            title=dict(text=f"{zname}<br><span style='font-size:13px'>ณ {at:%d %b %Y}</span>"),
            gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 20, 40, 60, 80, 100]),
                       bar=dict(color="#222", thickness=0.25),
                       steps=[dict(range=[lo, min(hi, 100)], color=c) for lo, hi, _, c in ZONES])))
        f.update_layout(height=300, margin=dict(l=30, r=30, t=70, b=10))
        st.plotly_chart(f, use_container_width=True)
        m = st.columns(3)
        for col, (lab, n) in zip(m, [("1 สัปดาห์ก่อน", 5), ("1 เดือนก่อน", 21), ("3 เดือนก่อน", 63)]):
            v = ago(n)
            col.metric(lab, "-" if pd.isna(v) else f"{v:.0f}",
                       None if pd.isna(v) else f"{row['score'] - v:+.0f} จุดถึงวันที่เลือก", delta_color="off")
    with g2:
        st.markdown("**ส่วนประกอบของคะแนน**")
        comp = [(COMPONENTS[k][0], row[k], COMPONENTS[k][1]) for k in COMPONENTS if pd.notna(row[k])]
        comp.sort(key=lambda x: x[1])
        f = go.Figure(go.Bar(x=[c[1] for c in comp], y=[c[0] for c in comp], orientation="h",
                             marker_color=[zone_of(c[1])[1] for c in comp],
                             text=[f"{c[1]:.0f}" for c in comp], textposition="outside",
                             customdata=[c[2] for c in comp],
                             hovertemplate="%{y}<br>คะแนน %{x:.0f}<br>%{customdata}<extra></extra>"))
        f.add_vline(x=50, line_dash="dot", line_color="gray")
        f.update_layout(height=340, margin=dict(l=10, r=30, t=10, b=10), xaxis=dict(range=[0, 105]))
        st.plotly_chart(f, use_container_width=True)
        st.caption("แต่ละตัวคือ percentile เทียบ 1 ปีย้อนหลัง (50 = ปกติ) · ชี้เมาส์ที่แท่งเพื่อดูความหมาย")

    warn_dates = render_warning(warn, at)

    # ----- กราฟย้อนหลัง -----
    st.markdown("#### 📈 คะแนนย้อนหลังเทียบ S&P 500")
    rng = st.radio("ช่วงที่แสดง", ["1 ปี", "3 ปี", "5 ปี", "ทั้งหมด"], index=1, horizontal=True, key="mood_rng")
    yrs = {"1 ปี": 1, "3 ปี": 3, "5 ปี": 5, "ทั้งหมด": 30}[rng]
    h = sc[sc.index >= last - pd.DateOffset(years=yrs)]
    smooth = h["score"].rolling(5, min_periods=1).mean()
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06)
    for lo, hi, name, color in ZONES:
        f.add_hrect(y0=lo, y1=min(hi, 100), fillcolor=color, opacity=0.15, line_width=0, row=1, col=1)
    f.add_trace(go.Scatter(x=h.index, y=h["score"], name="คะแนนรายวัน", line=dict(color="#bbb", width=1)),
                row=1, col=1)
    f.add_trace(go.Scatter(x=h.index, y=smooth, name="เฉลี่ย 5 วัน", line=dict(color="#222", width=2)), row=1, col=1)
    f.add_trace(go.Scatter(x=h.index, y=h["spy"], name="S&P 500 (SPY)", line=dict(color="#2a78d6", width=1.5)),
                row=2, col=1)
    wd = [t for t in warn_dates if t >= h.index[0]]
    if wd:
        f.add_trace(go.Scatter(x=wd, y=h["spy"].reindex(wd), mode="markers", name="🚨 สัญญาณเตือน",
                               marker=dict(symbol="triangle-down", size=12, color="#d62728")), row=2, col=1)
    if h.index[0] <= at <= h.index[-1]:
        f.add_vline(x=at, line_dash="dash", line_color="#d62728")
    f.update_yaxes(range=[0, 100], title_text="คะแนน", row=1, col=1)
    f.update_yaxes(title_text="SPY", type="log" if yrs >= 5 else "linear", row=2, col=1)
    f.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                    legend=dict(orientation="h", y=1.05))
    st.plotly_chart(f, use_container_width=True)

    # ----- สถิติรายโซน -----
    st.markdown("#### 🧮 ในอดีต เมื่ออยู่แต่ละโซน ตลาดหลังจากนั้นเป็นอย่างไร")
    period = st.radio("ช่วงข้อมูลที่ใช้คิดสถิติ",
                      ["ทั้งหมด", "2007–2018 (ช่วงแรก)", "2019–ปัจจุบัน (ช่วงหลัง)"], horizontal=True, key="mood_per",
                      help="เทียบสองช่วงเพื่อดูว่าผลคงเส้นคงวาไหม ถ้าได้ผลแค่ช่วงเดียว อาจเป็นความบังเอิญ")
    se = {"ทั้งหมด": (None, None), "2007–2018 (ช่วงแรก)": (None, pd.Timestamp("2018-12-31")),
          "2019–ปัจจุบัน (ช่วงหลัง)": (pd.Timestamp("2019-01-01"), None)}[period]
    zs = zone_stats(sc, *se)
    pc = st.column_config.NumberColumn(format="%+.2f%%")
    pp = st.column_config.NumberColumn(format="%.0f%%")
    st.dataframe(zs, hide_index=True, use_container_width=True, column_config={
        "% ของเวลา": pp, "เฉลี่ย 1 เดือน": pc, "เฉลี่ย 3 เดือน": pc, "เฉลี่ย 6 เดือน": pc,
        "ขึ้น 1 เดือน (% ครั้ง)": pp, "ขึ้น 3 เดือน (% ครั้ง)": pp, "ขึ้น 6 เดือน (% ครั้ง)": pp,
        "แย่สุด 3 เดือน": pc})
    zz = zs[zs["โซน"] != "ทุกวัน (ฐานเปรียบเทียบ)"]
    base = zs.loc[zs["โซน"] == "ทุกวัน (ฐานเปรียบเทียบ)", "เฉลี่ย 3 เดือน"].iloc[0]
    f = go.Figure(go.Bar(x=zz["โซน"], y=zz["เฉลี่ย 3 เดือน"], marker_color=[z[3] for z in ZONES],
                         text=[f"{v:+.1f}%" for v in zz["เฉลี่ย 3 เดือน"]], textposition="outside"))
    f.add_hline(y=base, line_dash="dot", line_color="gray", annotation_text=f"เฉลี่ยทุกวัน {base:+.1f}%")
    f.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="ผลตอบแทน S&P 500 เฉลี่ย 3 เดือนถัดไป แยกตามโซน",
                    yaxis=dict(ticksuffix="%"))
    st.plotly_chart(f, use_container_width=True)
    st.caption("ผลตอบแทนข้างหน้าใช้ประเมินดัชนีเท่านั้น ไม่ได้ใช้คำนวณคะแนน · วันที่ติดกันในโซนเดียวกันซ้อนทับกัน "
               "จึงดูจำนวนวันประกอบ (ช่วงวิกฤตไม่กี่ครั้งอาจทำให้ค่าเฉลี่ยของโซนสุดขีดเอียง)")

    render_phase2(sc)

    with st.expander("📘 ตัวชี้วัดทั้ง 7 ตัวและวิธีคำนวณ"):
        for k, (name, desc) in COMPONENTS.items():
            st.markdown(f"- **{name}** — {desc}")
        st.markdown(
            "- แต่ละตัวแปลงเป็น **percentile เทียบ 252 วันทำการก่อนหน้า** (ใช้ข้อมูลถึงวันนั้นเท่านั้น ไม่มองอนาคต)\n"
            "- คะแนนรวม = **ค่าเฉลี่ยน้ำหนักเท่ากัน** (ต้องมีข้อมูลอย่างน้อย 5 ตัว) · ไม่ได้ปรับน้ำหนักให้ผลในอดีตสวย\n"
            "- ข้อมูลฟรีจาก Yahoo Finance: ^VIX, ^VIX3M, SPY, TLT, HYG, IEF และ SPDR sector ETF 11 กลุ่ม")
    st.caption("⚠️ ต้นแบบเพื่อการศึกษา ใช้ดูสภาวะตลาดโดยรวม ไม่ใช่คำแนะนำซื้อขาย · "
               "ช่วงวิกฤตจริงความกลัวอยู่ระดับสูงได้นานหลายเดือน")


# ======================= เฟส 2: สัญญาณ + Backtest =======================
STATES = {  # key: (ชื่อ, สี, สัดส่วนที่ถือ)
    "buy": ("🟢 ซื้อ / ทยอยสะสม", "#1b7837", 1.0),
    "lock": ("🟢 ถือหลังสัญญาณซื้อ (ขั้นต่ำ 3 เดือน)", "#5aae61", 1.0),
    "buy_half": ("🟢 ทยอยซื้อครึ่งเดียว (ขาลง แต่กลัวสุดขีด)", "#a6d96a", 0.5),
    "hold": ("🔵 ถือ", "#2a78d6", 1.0),
    "reduce": ("🟠 ลดความเสี่ยง (ถือครึ่ง)", "#e08214", 0.5),
    "caution": ("🔴 ระวัง (ถือเงินสด)", "#b2182b", 0.0),
}
FEAR, GREED, LOCK_DAYS = 25, 80, 63  # กฎตายตัว ไม่ได้จูนกับผลในอดีต
PRESETS = {"S&P 500 (SPY)": "SPY", "Nasdaq-100 (QQQ)": "QQQ", "Dow Jones (DIA)": "DIA", "Russell 2000 (IWM)": "IWM"}
RULES = {  # key: (ชื่อ, คำอธิบาย)
    "B": ("กฎ B (แนะนำ)",
          "ใช้ MA200 เป็นหลัก แล้วให้อารมณ์ตลาดปรับน้ำหนัก: ขาขึ้น → ถือเต็ม แต่ลดเหลือครึ่งเมื่อโลภสุดขีด (≥ 80) · "
          "ขาลง → ถือเงินสด ยกเว้นกลัวสุดขีดและเริ่มดีขึ้น (< 25 และสูงกว่า 5 วันก่อน) → ซื้อครึ่งเดียว ไม่บังคับถือขั้นต่ำ"),
    "O": ("กฎ O (ต้นฉบับเฟส 2)",
          "กลัวสุดขีดและเริ่มดีขึ้น → ซื้อเต็มและถือขั้นต่ำ 3 เดือน (ไม่สนแนวโน้ม) · ขาลง → เงินสด · "
          "ขาขึ้นและโลภสุดขีด → ถือครึ่ง · อื่นๆ → ถือเต็ม"),
    "MA200": ("MA200 อย่างเดียว", "ราคาเหนือเส้น 200 วัน → ถือเต็ม · ใต้เส้น → ถือเงินสด (ไม่ใช้อารมณ์ตลาด)"),
}
# ผลทดสอบที่ประกาศกฎไว้ก่อนทดสอบ (เฉลี่ย SPY, QQQ, DIA, IWM · ค่าซื้อขาย 5 bps · ทดสอบ 16 พ.ย. 2026)
VARIANT_TEST = pd.DataFrame([
    ["ซื้อแล้วถือ", 8.5, -51.0, 0.49, 15.5, -36.7, 0.78],
    ["กฎ O (ต้นฉบับ)", 5.8, -48.3, 0.39, 15.4, -33.0, 0.85],
    ["MA200 อย่างเดียว", 8.7, -21.4, 0.74, 10.1, -25.2, 0.74],
    ["กฎ B", 8.0, -19.6, 0.67, 12.7, -23.5, 0.87],
], columns=["วิธี", "2008–18 CAGR %", "2008–18 Max DD %", "2008–18 Sharpe",
            "2019–ปัจจุบัน CAGR %", "2019–ปัจจุบัน Max DD %", "2019–ปัจจุบัน Sharpe"])


@st.cache_data(ttl=6 * 3600, show_spinner="กำลังดึงราคาสินทรัพย์สำหรับ backtest ...")
def asset_prices(syms: tuple) -> pd.DataFrame:
    d = yf.download(list(syms) + ["SPY"], start="2006-01-01", interval="1d", progress=False, auto_adjust=True)["Close"]
    if isinstance(d, pd.Series):
        d = d.to_frame(syms[0])
    d.index = pd.DatetimeIndex(d.index.tz_localize(None) if d.index.tz is not None else d.index)
    return d


def signal_states(score: pd.Series, price: pd.Series, rule: str = "B") -> pd.DataFrame:
    """สถานะและสัดส่วนที่ถือรายวัน บนวันทำการของ price (score ของตลาดสหรัฐฯ forward-fill มาที่วันนั้น)
    ใช้ข้อมูลถึงวันนั้นเท่านั้น · สัดส่วนไปมีผลวันทำการถัดไป (ดู run_backtest)"""
    idx = price.dropna().index
    s5 = score.rolling(5, min_periods=1).mean().reindex(idx, method="ffill")
    ma200 = price.reindex(idx).rolling(200, min_periods=200).mean()
    up = price.reindex(idx) > ma200
    fear_rising = (s5 < FEAR) & (s5 > s5.shift(5))
    greed = s5 >= GREED
    na = ma200.isna() | s5.isna()

    if rule == "MA200":
        state = pd.Series(np.where(up, "hold", "caution"), index=idx)
    elif rule == "B":
        state = pd.Series(np.select(
            [up & fear_rising, up & greed, up, fear_rising],
            ["buy", "reduce", "hold", "buy_half"], "caution"), index=idx)
    else:  # O: ต้นฉบับ (ซื้อเมื่อกลัว + ถือขั้นต่ำ 3 เดือน)
        state = pd.Series(np.select([fear_rising, ~up, greed & up], ["buy", "caution", "reduce"], "hold"), index=idx)
    state[na] = "na"

    pos, shown, lock = [], [], 0
    for st_ in state.values:
        if rule == "O" and st_ == "buy":
            lock = LOCK_DAYS
        if st_ == "na":
            pos.append(np.nan)
            shown.append("na")
        elif lock > 0:
            pos.append(1.0)
            shown.append("buy" if st_ == "buy" else "lock")  # แสดงตามสิ่งที่กลยุทธ์ถือจริง
        else:
            pos.append(STATES[st_][2])
            shown.append(st_)
        lock = max(0, lock - 1)
    return pd.DataFrame({"state": shown, "pos": pos, "s5": s5, "up": up}, index=idx)


def run_backtest(price: pd.Series, pos: pd.Series, start, cost_bps: float):
    """ซื้อขายที่ราคาปิดวันทำการถัดไปหลังเกิดสัญญาณ · เงินสดไม่ได้ดอกเบี้ย · หักค่าซื้อขายตามสัดส่วนที่เปลี่ยน"""
    p = price.dropna()
    p = p[p.index >= pd.Timestamp(start)]
    r = p.pct_change().fillna(0)
    w = pos.reindex(p.index).shift(1).fillna(0)
    strat = w * r - w.diff().abs().fillna(0) * cost_bps / 1e4
    return (1 + strat).cumprod(), strat, w


def perf(eq: pd.Series, ret: pd.Series, w=None) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    dd = eq / eq.cummax() - 1
    out = {"CAGR %": (eq.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else np.nan,
           "ผลตอบแทนรวม %": (eq.iloc[-1] - 1) * 100,
           "Max DD %": dd.min() * 100,
           "Sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() else np.nan}
    if w is not None:
        out["% เวลาที่ลงทุน"] = w.mean() * 100
        out["จำนวนครั้งที่ปรับพอร์ต"] = int((w.diff().abs() > 0).sum())
    return out


def compare_asset(price: pd.Series, spy: pd.Series, score: pd.Series, start, cost_bps: float, trend_src: str):
    """ผลของซื้อแล้วถือ + ทุกชุดกฎ บนสินทรัพย์เดียว"""
    trend_px = spy if trend_src == "market" else price
    p = price.dropna()
    out, sigs = {}, {}
    out["bh"] = run_backtest(p, pd.Series(1.0, index=p.index), start, 0)
    for rule in RULES:
        sig = signal_states(score, trend_px, rule).reindex(p.index, method="ffill")
        out[rule] = run_backtest(p, sig["pos"], start, cost_bps)
        sigs[rule] = sig
    return out, sigs


def render_phase2(sc: pd.DataFrame):
    st.divider()
    st.markdown("#### 🧭 สัญญาณ ซื้อ / ถือ / ระวัง (เฟส 2)")
    rule = st.radio("ชุดกฎ", list(RULES), format_func=lambda k: RULES[k][0], horizontal=True, key="m2_rule",
                    help="กฎ B ผ่านการทดสอบแบบแบ่งสองช่วงเวลาได้สมดุลที่สุด (ดูตารางผลทดสอบด้านล่าง)")
    st.caption(f"**{RULES[rule][0]}:** {RULES[rule][1]} · ใช้คะแนนเฉลี่ย 5 วัน · ซื้อขายที่ราคาปิดวันทำการถัดไปหลังเกิดสัญญาณ")

    spy = mood_data()["SPY"]
    sig_mkt = signal_states(sc["score"], spy, rule)
    last_state = sig_mkt["state"].iloc[-1]
    name, color, _ = STATES.get(last_state, ("-", "#999", 0))
    w = sig_mkt["pos"].iloc[-1]
    k = st.columns(3)
    k[0].markdown(f'<div style="border-radius:12px;padding:14px 18px;background:{color}22;border:1px solid {color}">'
                  f'<div style="font-size:.85rem;opacity:.7">สัญญาณตลาดสหรัฐฯ ({RULES[rule][0]}) ณ {sig_mkt.index[-1]:%d %b %Y}</div>'
                  f'<div style="font-size:1.5rem;font-weight:700;color:{color}">{name}</div>'
                  f'<div style="font-size:.85rem">สัดส่วนหุ้นตามกฎ {w * 100:.0f}%</div></div>', unsafe_allow_html=True)
    k[1].metric("คะแนนเฉลี่ย 5 วัน", f"{sig_mkt['s5'].iloc[-1]:.0f}",
                f"{sig_mkt['s5'].iloc[-1] - sig_mkt['s5'].iloc[-6]:+.0f} จาก 5 วันก่อน", delta_color="off")
    k[2].metric("S&P 500 เทียบ MA200", "เหนือเส้น (ขาขึ้น)" if sig_mkt["up"].iloc[-1] else "ใต้เส้น (ขาลง)")

    t = sig_mkt[sig_mkt.index >= sig_mkt.index[-1] - pd.DateOffset(years=3)]
    f = go.Figure()
    for key, (lab, col, _) in STATES.items():
        m = t["state"] == key
        if m.any():
            f.add_trace(go.Scatter(x=t.index[m], y=spy.reindex(t.index)[m], mode="markers", name=lab,
                                   marker=dict(color=col, size=5)))
    f.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                    title=f"S&P 500 ระบายสีตามสัญญาณ {RULES[rule][0]} (3 ปีล่าสุด)",
                    legend=dict(orientation="h", y=1.14), hovermode="closest")
    st.plotly_chart(f, use_container_width=True)

    # ----- Backtest หลายสินทรัพย์ -----
    st.markdown("#### 🧪 Backtest: ทำตามสัญญาณ เทียบซื้อแล้วถือ")
    c = st.columns([2, 2, 1, 1, 1.4])
    picks = c[0].multiselect("ดัชนี / ETF", list(PRESETS), default=list(PRESETS)[:2], key="m2_presets")
    extra = c[1].text_input("หุ้นหรือสินทรัพย์อื่น (คั่นด้วย ,)", placeholder="เช่น NVDA, AAPL, PTT.BK", key="m2_extra")
    start = c[2].date_input("เริ่ม", value=dt.date(2008, 1, 1), min_value=dt.date(2007, 12, 1),
                            max_value=dt.date.today() - dt.timedelta(days=200), key="m2_start")
    cost = c[3].number_input("ค่าซื้อขาย (bps)", 0.0, 100.0, 5.0, 5.0, key="m2_cost")
    trend_src = c[4].radio("ตัวกรองแนวโน้มใช้", ["ตลาด (S&P 500)", "ตัวสินทรัพย์เอง"], key="m2_trend",
                           help="ตลาด = ดูว่า S&P 500 อยู่เหนือ MA200 ไหม · ตัวสินทรัพย์เอง = ดู MA200 ของตัวที่ทดสอบ")
    syms = {PRESETS[p]: p for p in picks}
    for x in [s.strip().upper() for s in extra.split(",") if s.strip()]:
        syms.setdefault(x, x)
    if not syms:
        st.info("เลือกอย่างน้อย 1 สินทรัพย์")
        return
    px = asset_prices(tuple(sorted(syms)))
    methods = [("bh", "ซื้อแล้วถือ")] + [(r, RULES[r][0]) for r in RULES]
    rows, results = [], {}
    for sym, label in syms.items():
        if sym not in px or px[sym].dropna().empty:
            st.warning(f"ไม่พบข้อมูล {sym}")
            continue
        p = px[sym].dropna()
        st_ = max(pd.Timestamp(start), p.index[0] + pd.Timedelta(days=300))  # ต้องมี MA200 ก่อน
        res, _ = compare_asset(p, px["SPY"], sc["score"], st_, cost,
                               "market" if trend_src.startswith("ตลาด") else "self")
        results[label] = res
        for key, nm in methods:
            eq, r, ww = res[key]
            rows.append({"สินทรัพย์": label, "เริ่ม": f"{st_:%b %Y}", "วิธี": nm,
                         **perf(eq, r, None if key == "bh" else ww)})
    if not rows:
        return
    tbl = pd.DataFrame(rows)
    pc = st.column_config.NumberColumn(format="%+.2f%%")
    st.dataframe(tbl, hide_index=True, use_container_width=True, column_config={
        "CAGR %": pc, "ผลตอบแทนรวม %": st.column_config.NumberColumn(format="%+,.0f%%"),
        "Max DD %": st.column_config.NumberColumn(format="%.1f%%"), "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "% เวลาที่ลงทุน": st.column_config.NumberColumn(format="%.0f%%")})
    st.caption("Sharpe สูง = ผลตอบแทนต่อความเสี่ยงดี · Max DD = ลงหนักสุดจากจุดสูงสุด · "
               "\"MA200 อย่างเดียว\" เป็นตัวควบคุม ใช้ดูว่าอารมณ์ตลาดช่วยเพิ่มอะไรจากการดูแนวโน้มอย่างเดียว")

    one = st.selectbox("ดูกราฟของ", list(results), key="m2_one")
    res = results[one]
    log = st.toggle("แกน log", value=True, key="m2_log")
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)
    styles = {"bh": ("#898781", "dash"), "B": ("#1b7837", None), "O": ("#9970ab", "dot"), "MA200": ("#e08214", "dot")}
    for key, nm in methods:
        col, dash = styles[key]
        f.add_trace(go.Scatter(x=res[key][0].index, y=res[key][0] * 100, name=nm,
                               line=dict(color=col, width=2.5 if key == rule else 1.5, dash=dash)), row=1, col=1)
    ww = res[rule][2]
    f.add_trace(go.Scatter(x=ww.index, y=ww * 100, name=f"% ที่ถือ ({RULES[rule][0]})", fill="tozeroy",
                           line=dict(color=styles[rule][0], width=1)), row=2, col=1)
    f.update_yaxes(type="log" if log else "linear", title_text="เงินลงทุน (เริ่ม 100)", row=1, col=1)
    f.update_yaxes(range=[0, 105], ticksuffix="%", title_text="ถือหุ้น", row=2, col=1)
    f.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                    legend=dict(orientation="h", y=1.06))
    st.plotly_chart(f, use_container_width=True)

    with st.expander("📋 ผลทดสอบเปรียบเทียบชุดกฎ (ประกาศกฎไว้ก่อนทดสอบ แบ่งสองช่วงเวลา)", expanded=False):
        pct = st.column_config.NumberColumn(format="%+.1f%%")
        st.dataframe(VARIANT_TEST, hide_index=True, use_container_width=True, column_config={
            "2008–18 CAGR %": pct, "2008–18 Max DD %": st.column_config.NumberColumn(format="%.1f%%"),
            "2019–ปัจจุบัน CAGR %": pct, "2019–ปัจจุบัน Max DD %": st.column_config.NumberColumn(format="%.1f%%")})
        st.markdown(
            "- ค่าเฉลี่ยของ SPY, QQQ, DIA, IWM · ค่าซื้อขาย 5 bps · ตัวกรองแนวโน้มใช้ S&P 500 · ทดสอบ 16 พ.ย. 2026\n"
            "- **กฎ B** ลด Max DD ได้เกินครึ่งเทียบซื้อแล้วถือ **ทั้งสองช่วง** และ Sharpe สูงกว่าซื้อแล้วถือทั้งสองช่วง\n"
            "- **กฎ O** ช่วงวิกฤต 2008 แทบไม่ช่วย เพราะซื้อเต็มตอนกลัวทั้งที่ราคายังลง (รับมีด)\n"
            "- ไม่มีกฎไหนชนะซื้อแล้วถือด้าน**ผลตอบแทน**ในตลาดกระทิงแรงช่วง 2019–ปัจจุบัน สิ่งที่ได้คือความเสี่ยงที่ต่ำลง\n"
            "- กฎทางเลือกที่ทดสอบ: A (ซื้อเมื่อกลัวเฉพาะขาขึ้น) กลายเป็นตัวเดียวกับ MA200 อย่างเดียว · "
            "C (ขาลงซื้อครึ่งไม่ล็อก) ให้ผลเหมือน B ทุกตัว")
    st.caption("⚠️ ผลในอดีตไม่รับประกันอนาคต · เงินสดไม่ได้ดอกเบี้ย (ช่วงดอกเบี้ยสูงกลยุทธ์ที่ถือเงินสดจะดูแย่กว่าจริงเล็กน้อย) · "
               "ยังไม่รวมภาษี · หุ้นต่างตลาด (เช่น .BK) ใช้สัญญาณของตลาดสหรัฐฯ และซื้อขายวันทำการถัดไปของตลาดนั้น · "
               "เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน")
