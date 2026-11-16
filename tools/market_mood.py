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


# ======================= UI =======================
def render_mood_tab():
    d = mood_data()
    sc, raw = compute_mood(d)
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

    with st.expander("📘 ตัวชี้วัดทั้ง 7 ตัวและวิธีคำนวณ"):
        for k, (name, desc) in COMPONENTS.items():
            st.markdown(f"- **{name}** — {desc}")
        st.markdown(
            "- แต่ละตัวแปลงเป็น **percentile เทียบ 252 วันทำการก่อนหน้า** (ใช้ข้อมูลถึงวันนั้นเท่านั้น ไม่มองอนาคต)\n"
            "- คะแนนรวม = **ค่าเฉลี่ยน้ำหนักเท่ากัน** (ต้องมีข้อมูลอย่างน้อย 5 ตัว) · ไม่ได้ปรับน้ำหนักให้ผลในอดีตสวย\n"
            "- ข้อมูลฟรีจาก Yahoo Finance: ^VIX, ^VIX3M, SPY, TLT, HYG, IEF และ SPDR sector ETF 11 กลุ่ม")
    st.caption("⚠️ ต้นแบบเพื่อการศึกษา ใช้ดูสภาวะตลาดโดยรวม ไม่ใช่คำแนะนำซื้อขาย · "
               "ช่วงวิกฤตจริงความกลัวอยู่ระดับสูงได้นานหลายเดือน")
