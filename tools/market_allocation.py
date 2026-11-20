"""แท็บ "สัดส่วนหุ้น-พันธบัตร" – สูตรรวม 3 ชั้น (ต้นแบบ) · ตลาดสหรัฐฯ
กฎประกาศไว้ก่อนทดสอบ (16 พ.ย. 2026):
  ชั้น 1 Valuation: Earnings Yield Gap (100/PE − ผลตอบแทนพันธบัตร 10 ปี) z-score เทียบ 10 ปี
         z > +1 → หุ้น 70% · −1..+1 → 55% · z < −1 → 40%
  ชั้น 2 แนวโน้ม: S&P 500 ต่ำกว่าค่าเฉลี่ย 10 เดือน → ลดหุ้นเหลือ 50% ของชั้น 1
  ชั้น 3 กลัวสุดขีด (เฉพาะตอนหลุดแนวโน้ม): VIX อยู่ใน 10% สูงสุดของ 5 ปี และต่ำกว่าเดือนก่อน → 75% ของชั้น 1
ส่วนที่ไม่ใช่หุ้นถือพันธบัตรรัฐบาลสหรัฐฯ 10 ปี · คำนวณเมื่อสิ้นเดือน มีผลเดือนถัดไป

ข้อมูลฟรีทั้งหมด: P/E และอัตราปันผล S&P 500 จาก multpl.com (สำรอง: files/sp500_valuation_monthly.csv)
ราคา S&P 500, ^TNX, ^VIX จาก Yahoo Finance · ผลตอบแทนพันธบัตร 10 ปีประมาณจาก yield (duration 7.5)
"""
import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_CSV = os.path.join(HERE, "..", "files", "sp500_valuation_monthly.csv")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}
DURATION = 7.5
EYG_W = {"cheap": 0.70, "fair": 0.55, "rich": 0.40}


# ---------------- ข้อมูล ----------------
def _multpl(slug: str) -> pd.Series:
    html = requests.get(f"https://www.multpl.com/{slug}/table/by-month", headers=UA, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    t.columns = ["date", "v"]
    t["date"] = pd.to_datetime(t["date"])
    t["v"] = pd.to_numeric(t["v"].astype(str).str.replace("†", "").str.replace("%", "").str.strip(), errors="coerce")
    return t.dropna().set_index("date")["v"].sort_index()


@st.cache_data(ttl=24 * 3600, show_spinner="กำลังดึง P/E และอัตราปันผล S&P 500 (multpl.com) ...")
def valuation() -> tuple:
    """คืน (pe, div_yield %, แหล่งที่มา) · ถ้าดึงสดไม่ได้ ใช้ไฟล์สำรองใน repo"""
    try:
        pe, dy = _multpl("s-p-500-pe-ratio"), _multpl("s-p-500-dividend-yield")
        if len(pe) > 1000 and len(dy) > 1000:
            return pe, dy, "multpl.com (สด)"
    except Exception:
        pass
    v = pd.read_csv(VAL_CSV, parse_dates=["date"], index_col="date")
    return v["pe"].dropna(), v["div_yield_pct"].dropna(), "ไฟล์สำรอง files/sp500_valuation_monthly.csv"


@st.cache_data(ttl=3600, show_spinner="กำลังดึง S&P 500, ผลตอบแทนพันธบัตร 10 ปี และ VIX ...")
def market_daily() -> pd.DataFrame:
    d = yf.download(["^GSPC", "^TNX", "^VIX", "SPY"], period="max", interval="1d", progress=False, auto_adjust=False)["Close"]
    d.index = pd.DatetimeIndex(d.index.tz_localize(None) if d.index.tz is not None else d.index)
    return d


def build():
    pe, dy, src = valuation()
    d = market_daily()
    m = pd.DataFrame({"px": d["^GSPC"].dropna().resample("ME").last(),
                      "tnx": d["^TNX"].dropna().resample("ME").last(),
                      "vix": d["^VIX"].dropna().resample("ME").last()})
    m["pe"] = pe.resample("ME").last().reindex(m.index).ffill(limit=2)
    m["dy"] = dy.resample("ME").last().reindex(m.index).ffill(limit=2)
    m = m[m.index >= "1962-01-31"]
    m["eyg"] = 100 / m["pe"] - m["tnx"]
    mu, sd = m["eyg"].rolling(120, min_periods=60).mean(), m["eyg"].rolling(120, min_periods=60).std()
    m["z"] = (m["eyg"] - mu) / sd
    m["strategic"] = np.where(m["z"] > 1, EYG_W["cheap"], np.where(m["z"] < -1, EYG_W["rich"], EYG_W["fair"]))
    m.loc[m["z"].isna(), "strategic"] = np.nan
    ma10 = m["px"].rolling(10).mean()
    m["down"] = (m["px"] < ma10).astype(float).where(ma10.notna())
    vp = m["vix"].rolling(60, min_periods=36).rank(pct=True) * 100
    m["vix_pct"] = vp
    m["fear_easing"] = ((vp >= 90) & (m["vix"] < m["vix"].shift(1))).where(vp.notna())
    l12 = m["strategic"] * np.where(m["down"] == 1, 0.5, 1.0)
    l123 = m["strategic"] * np.where(m["down"] == 1, np.where(m["fear_easing"] == 1, 0.75, 0.5), 1.0)
    m["w12"] = l12
    m["w"] = pd.Series(l123, index=m.index).where(m["vix_pct"].notna(), l12)
    m["w_eyg"] = m["strategic"]
    m["w_trend"] = pd.Series(np.where(m["down"] == 1, 0.275, 0.55), index=m.index).where(m["down"].notna())
    m["eq_r"] = m["px"].pct_change() + m["dy"].shift(1) / 100 / 12
    m["bd_r"] = m["tnx"].shift(1) / 100 / 12 - DURATION * (m["tnx"] - m["tnx"].shift(1)) / 100
    return m, d, src


def run(m: pd.DataFrame, w: pd.Series, start, end=None) -> dict:
    x = m[(m.index >= pd.Timestamp(start)) & ((m.index <= pd.Timestamp(end)) if end else True)]
    ww = w.reindex(x.index).shift(1)  # ตัดสินใจสิ้นเดือน t (ราคาปิดสิ้นเดือน) → มีผลเดือน t+1
    r = (ww * x["eq_r"] + (1 - ww) * x["bd_r"])[ww.notna()].dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    dd = eq / eq.cummax() - 1
    return {"CAGR %": (eq.iloc[-1] ** (1 / yrs) - 1) * 100, "Sharpe": r.mean() / r.std() * np.sqrt(12),
            "Max DD %": dd.min() * 100, "Vol %": r.std() * np.sqrt(12) * 100, "หุ้นเฉลี่ย %": ww.reindex(r.index).mean() * 100}


# ---------------- UI ----------------
def _card(title, value, sub, color):
    return (f'<div style="border-radius:12px;padding:12px 16px;background:{color}1f;border:1px solid {color};height:100%">'
            f'<div style="font-size:.85rem;opacity:.75">{title}</div>'
            f'<div style="font-size:1.45rem;font-weight:700;color:{color}">{value}</div>'
            f'<div style="font-size:.85rem">{sub}</div></div>')


def allocation_now() -> dict:
    """สัดส่วนหุ้นที่สูตรแนะนำ ประมาณการ ณ ราคาล่าสุด (ใช้ในรายงานประจำวัน) · ตรรกะเดียวกับการ์ดในแท็บนี้"""
    m_all, d, src = build()
    today = d["^GSPC"].dropna().index[-1]
    m = m_all[m_all.index < today + pd.offsets.MonthEnd(0)]
    last_m = m.dropna(subset=["w"]).index[-1]
    pe, _, _ = valuation()
    px_now, tnx_now, vix_now = d["^GSPC"].dropna().iloc[-1], d["^TNX"].dropna().iloc[-1], d["^VIX"].dropna().iloc[-1]
    cut = today.to_period("M").to_timestamp("M")
    eyg_now = 100 / pe.iloc[-1] - tnx_now
    hist = pd.concat([m["eyg"].dropna()[lambda s: s.index < cut].iloc[-119:], pd.Series([eyg_now])])
    z_now = (eyg_now - hist.mean()) / hist.std()
    strat = EYG_W["cheap"] if z_now > 1 else EYG_W["rich"] if z_now < -1 else EYG_W["fair"]
    me = m["px"].dropna()[lambda s: s.index < cut].iloc[-9:]
    down = px_now < pd.concat([me, pd.Series([px_now])]).mean()
    vh = m["vix"].dropna()[lambda s: s.index < cut]
    fear = (vh.iloc[-59:] < vix_now).mean() * 100 >= 90 and vix_now < vh.iloc[-1]
    w = strat * ((0.75 if fear else 0.5) if down else 1.0)
    return {"w": w, "z": z_now, "strategic": strat, "down": bool(down), "fear": bool(fear), "date": today,
            "w_month": m.loc[last_m, "w"], "month": last_m}


def render_allocation_tab():
    m_all, d, src = build()
    today = d["^GSPC"].dropna().index[-1]
    m = m_all[m_all.index < today + pd.offsets.MonthEnd(0)]  # ใช้เฉพาะเดือนที่ปิดแล้ว (เดือนปัจจุบันยังไม่จบ)
    last_m = m.dropna(subset=["w"]).index[-1]
    pe, _, _ = valuation()

    # ----- สถานะวันนี้ (ประมาณการ ก่อนปิดเดือน) -----
    px_now, tnx_now, vix_now = d["^GSPC"].dropna().iloc[-1], d["^TNX"].dropna().iloc[-1], d["^VIX"].dropna().iloc[-1]
    eyg_now = 100 / pe.iloc[-1] - tnx_now
    hist = m["eyg"].dropna()
    hist = hist[hist.index < today.to_period("M").to_timestamp("M")].iloc[-119:]  # 119 เดือนก่อน + เดือนนี้ = 10 ปี
    z_now = (eyg_now - pd.concat([hist, pd.Series([eyg_now])]).mean()) / pd.concat([hist, pd.Series([eyg_now])]).std()
    strat_now = EYG_W["cheap"] if z_now > 1 else EYG_W["rich"] if z_now < -1 else EYG_W["fair"]
    month_ends = m["px"].dropna()
    month_ends = month_ends[month_ends.index < today.to_period("M").to_timestamp("M")].iloc[-9:]
    ma10_now = pd.concat([month_ends, pd.Series([px_now])]).mean()
    down_now = px_now < ma10_now
    vix_hist = m["vix"].dropna()
    vix_hist = vix_hist[vix_hist.index < today.to_period("M").to_timestamp("M")]
    vix_pct_now = (vix_hist.iloc[-59:] < vix_now).mean() * 100
    fear_now = vix_pct_now >= 90 and vix_now < vix_hist.iloc[-1]
    w_now = strat_now * ((0.75 if fear_now else 0.5) if down_now else 1.0)
    spy = d["SPY"].dropna()
    ma200 = spy.rolling(200).mean().iloc[-1]

    st.caption(f"ต้นแบบ · ตลาดสหรัฐฯ (S&P 500 + พันธบัตรรัฐบาล 10 ปี) · ข้อมูลถึง {today:%d %b %Y} · P/E จาก {src}")
    c = st.columns([1.2, 1, 1, 1])
    c[0].markdown(_card("สัดส่วนที่สูตรแนะนำ (ประมาณการวันนี้)", f"หุ้น {w_now * 100:.0f}% · พันธบัตร {100 - w_now * 100:.0f}%",
                        f"ค่าทางการเมื่อสิ้นเดือน {last_m:%b %Y}: หุ้น {m.loc[last_m, 'w'] * 100:.0f}%", "#2a78d6"),
                  unsafe_allow_html=True)
    band = "ถูก (z > +1)" if z_now > 1 else "แพง (z < −1)" if z_now < -1 else "ปกติ"
    c[1].markdown(_card("ชั้น 1 · Valuation (EYG)", f"z = {z_now:+.2f} · {band}",
                        f"P/E {pe.iloc[-1]:.1f} → EY {100 / pe.iloc[-1]:.2f}% − 10Y {tnx_now:.2f}% = {eyg_now:+.2f}% → หุ้นตั้งต้น {strat_now * 100:.0f}%",
                        "#1b7837" if z_now > 1 else "#b2182b" if z_now < -1 else "#888"), unsafe_allow_html=True)
    c[2].markdown(_card("ชั้น 2 · แนวโน้ม (เฉลี่ย 10 เดือน)", "หลุดแนวโน้ม → ลดครึ่ง" if down_now else "ขาขึ้น → ไม่ลด",
                        f"S&P 500 {px_now:,.0f} เทียบเส้น {ma10_now:,.0f} · (เส้น 200 วันของ SPY: {'เหนือ' if spy.iloc[-1] > ma200 else 'ใต้'}เส้น)",
                        "#b2182b" if down_now else "#1b7837"), unsafe_allow_html=True)
    c[3].markdown(_card("ชั้น 3 · กลัวสุดขีด (สวนกระแส)", "กลัวสุดขีดและเริ่มคลาย" if fear_now else "ไม่เข้าเงื่อนไข",
                        f"VIX {vix_now:.1f} · สูงกว่า {vix_pct_now:.0f}% ของ 5 ปี · ใช้เฉพาะตอนหลุดแนวโน้ม",
                        "#e08214" if fear_now else "#888"), unsafe_allow_html=True)
    st.caption("ค่า \"วันนี้\" เป็นการประมาณ สูตรตัดสินใจจริงที่ราคาปิดสิ้นเดือน · P/E เดือนปัจจุบันของ multpl เป็นค่าประมาณ (†)")

    # ----- กราฟย้อนหลัง -----
    st.markdown("#### 📈 สัดส่วนหุ้นที่สูตรแนะนำย้อนหลัง")
    rng = st.radio("ช่วง", ["10 ปี", "20 ปี", "ทั้งหมด"], index=1, horizontal=True, key="al_rng")
    yrs = {"10 ปี": 10, "20 ปี": 20, "ทั้งหมด": 80}[rng]
    h = m[m.index >= m.index[-1] - pd.DateOffset(years=yrs)].dropna(subset=["w"])
    f = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.05)
    f.add_trace(go.Scatter(x=h.index, y=h["px"], name="S&P 500", line=dict(color="#2a78d6")), row=1, col=1)
    f.add_trace(go.Scatter(x=h.index, y=h["w"] * 100, name="% หุ้นที่แนะนำ", line=dict(color="#1b7837", shape="hv"),
                           fill="tozeroy"), row=2, col=1)
    f.add_trace(go.Scatter(x=h.index, y=h["z"], name="EYG z-score", line=dict(color="#7f3b08")), row=3, col=1)
    for y0 in (1, -1):
        f.add_hline(y=y0, line_dash="dot", line_color="gray", row=3, col=1)
    f.update_yaxes(type="log", title_text="S&P 500", row=1, col=1)
    f.update_yaxes(range=[0, 80], ticksuffix="%", title_text="หุ้น", row=2, col=1)
    f.update_yaxes(title_text="EYG z", row=3, col=1)
    f.update_layout(height=640, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", legend=dict(orientation="h", y=1.04))
    st.plotly_chart(f, use_container_width=True)
    st.caption("EYG z > +1 = หุ้นถูกเทียบพันธบัตร (ถือมาก) · z < −1 = หุ้นแพง (ถือน้อย) · เส้นสีเขียวต่ำ = หลุดแนวโน้ม/หุ้นแพง")

    # ----- Backtest -----
    st.markdown("#### 🧪 ผลทดสอบย้อนหลัง (ข้อมูลฟรีชุดเดียวกับที่ใช้งานจริง)")
    first = m.dropna(subset=["w"]).index[0] + pd.DateOffset(months=1)
    periods = {f"ช่วงแรก ({first:%Y}–1994)": (first, "1994-12-31"), "ช่วงหลัง (1995–ปัจจุบัน)": ("1995-01-01", None),
               f"ทั้งหมด ({first:%Y}–ปัจจุบัน)": (first, None)}
    pname = st.radio("ช่วงเวลา", list(periods), index=2, horizontal=True, key="al_per")
    a, b = periods[pname]
    avg = m["w"].shift(1)[(m.index >= pd.Timestamp(a)) & ((m.index <= pd.Timestamp(b)) if b else True)].mean()
    rows = []
    for lab, w in [("สูตรรวม 3 ชั้น", m["w"]), ("ชั้น 1+2 (EYG + แนวโน้ม)", m["w12"]), ("EYG อย่างเดียว", m["w_eyg"]),
                   ("แนวโน้มอย่างเดียว", m["w_trend"]), (f"คงที่ หุ้น {avg * 100:.0f}% (เท่าค่าเฉลี่ยสูตร)", pd.Series(avg, index=m.index)),
                   ("คงที่ 60/40", pd.Series(0.6, index=m.index)), ("หุ้น 100%", pd.Series(1.0, index=m.index))]:
        rows.append({"วิธี": lab, **run(m, w, a, b)})
    pc = st.column_config.NumberColumn(format="%+.2f%%")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
        "CAGR %": pc, "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Max DD %": st.column_config.NumberColumn(format="%.1f%%"), "Vol %": st.column_config.NumberColumn(format="%.1f%%"),
        "หุ้นเฉลี่ย %": st.column_config.NumberColumn(format="%.0f%%")})

    bears = [("1973–74 วิกฤตน้ำมัน", "1973-01-01", "1974-12-31"), ("1987 Black Monday", "1987-08-01", "1987-12-31"),
             ("2000–02 ดอทคอม", "2000-09-01", "2002-10-31"), ("2007–09 วิกฤตการเงิน", "2007-10-01", "2009-03-31"),
             ("2020 COVID", "2020-02-01", "2020-04-30"), ("2022 ดอกเบี้ยขึ้น", "2022-01-01", "2022-10-31"),
             ("2025 ภาษีนำเข้า", "2025-02-01", "2025-04-30")]
    avg_all = m["w"].shift(1).mean()
    br = []
    for lab, s0, s1 in bears:
        x = m[(m.index >= s0) & (m.index <= s1)]
        row = {"ตลาดหมี": lab}
        for nm, w in [("สูตรรวม", m["w"]), (f"คงที่ {avg_all * 100:.0f}%", pd.Series(avg_all, index=m.index)),
                      ("หุ้น 100%", pd.Series(1.0, index=m.index))]:
            ww = w.shift(1).reindex(x.index)
            row[nm] = ((1 + ww * x["eq_r"] + (1 - ww) * x["bd_r"]).prod() - 1) * 100
        br.append(row)
    st.markdown("**ผลตอบแทนสะสมในช่วงตลาดหมีแต่ละรอบ**")
    st.dataframe(pd.DataFrame(br), hide_index=True, use_container_width=True,
                 column_config={k: pc for k in br[0] if k != "ตลาดหมี"})

    with st.expander("📘 กฎของสูตรและข้อควรระวัง"):
        st.markdown(
            "**กฎ (ประกาศไว้ก่อนทดสอบ)**\n"
            "1. **Valuation:** EYG = 100/PE − ผลตอบแทนพันธบัตร 10 ปี · z-score เทียบ 10 ปีย้อนหลัง · "
            "z > +1 → หุ้น 70% · −1..+1 → 55% · < −1 → 40%\n"
            "2. **แนวโน้ม:** S&P 500 ต่ำกว่าค่าเฉลี่ย 10 เดือน → ลดหุ้นเหลือครึ่งหนึ่ง\n"
            "3. **กลัวสุดขีด:** เฉพาะตอนหลุดแนวโน้ม ถ้า VIX อยู่ใน 10% สูงสุดของ 5 ปีและลดลงจากเดือนก่อน → ถือ 75% แทน 50%\n"
            "- ตัดสินใจเมื่อสิ้นเดือน มีผลเดือนถัดไป · ส่วนที่ไม่ใช่หุ้นถือพันธบัตร 10 ปี\n\n"
            "**ข้อควรระวัง**\n"
            "- ชั้นต่างๆ ถูกเลือกจากผลทดสอบก่อนหน้าบนข้อมูลช่วงเดียวกัน ผลจริงในอนาคตอาจด้อยกว่านี้\n"
            "- ช่วงปี 1981–2020 พันธบัตรเป็นขาขึ้นยาว กลยุทธ์ที่ถือพันธบัตรมากจึงได้เปรียบ\n"
            "- ผลตอบแทนพันธบัตรประมาณจาก yield 10 ปี (duration 7.5) · หุ้นรวมปันผลจากอัตราปันผลรายเดือน\n"
            "- ตลาดลงเร็วแล้วเด้งเร็ว (เช่น COVID 2020) ตัวกรองรายเดือนตอบสนองไม่ทัน\n"
            "- ยังไม่รวมค่าซื้อขายและภาษี · เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน")
