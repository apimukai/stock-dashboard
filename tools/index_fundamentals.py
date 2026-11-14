"""ข้อมูลพื้นฐานของดัชนี (ใช้ใน stock_dashboard.py)

Yahoo ไม่มีงบ/นักวิเคราะห์ของ "ดัชนี" โดยตรง จึงใช้:
- ETF ที่อิงดัชนีนั้น (คอลัมน์ etf ใน files/world_indices.csv): P/E, P/BV, P/S, ปันผล, น้ำหนัก sector, หุ้น 10 อันดับแรก
- หุ้น 10 อันดับแรก: เอามุมมอง/ราคาเป้าหมายของนักวิเคราะห์แต่ละตัว มาเฉลี่ยถ่วงน้ำหนักตามสัดส่วนในดัชนี
  (bottom-up) -> ได้ "ดัชนีเป้าหมายโดยนัย"
"""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from stock_profile import CSS, metric_html, num, pct, _num

HERE = os.path.dirname(os.path.abspath(__file__))
SET100_CSV = os.path.join(HERE, "..", "files", "set100.csv")
UPSIDE_CAP = (-0.5, 1.0)  # ตัด upside รายตัวที่ผิดปกติ (ข้อมูลเป้าหมายบางตัวใน Yahoo ผิด)
SECTOR_TH = {"technology": "เทคโนโลยี", "financial_services": "การเงิน", "consumer_cyclical": "สินค้าฟุ่มเฟือย",
             "communication_services": "สื่อสาร", "healthcare": "การแพทย์", "industrials": "อุตสาหกรรม",
             "consumer_defensive": "สินค้าจำเป็น", "energy": "พลังงาน", "basic_materials": "วัตถุดิบ",
             "realestate": "อสังหาฯ", "utilities": "สาธารณูปโภค"}
KEYS = ["shortName", "longName", "currency", "currentPrice", "regularMarketPrice", "targetMeanPrice",
        "recommendationMean", "numberOfAnalystOpinions", "trailingPE", "forwardPE", "earningsGrowth",
        "trailingEps", "forwardEps", "marketCap", "sector"]


def _fix_symbol(s: str, etf: str) -> str:
    s = str(s).strip()
    if s.isdigit():  # Yahoo ให้รหัสหุ้นฮ่องกงบางตัวแบบ "00939"
        return f"{int(s):04d}.HK"
    if s.endswith(".KQ") and etf == "EWY":  # หุ้นใน KOSPI ต้องเป็น .KS
        return s[:-3] + ".KS"
    return s


def _info(sym: str) -> dict:
    try:
        i = yf.Ticker(sym).info or {}
        return {k: i.get(k) for k in KEYS}
    except Exception:
        return {}


@st.cache_data(ttl=6 * 3600, show_spinner="กำลังดึงข้อมูล ETF ของดัชนี ...")
def etf_data(etf: str) -> dict:
    t = yf.Ticker(etf)
    out = {"info": {}, "top": pd.DataFrame(), "eq": pd.DataFrame(), "sectors": {}, "desc": ""}
    try:
        i = t.info or {}
        out["info"] = {k: i.get(k) for k in ["longName", "trailingPE", "yield", "totalAssets", "ytdReturn",
                                             "netExpenseRatio", "currency"]}
    except Exception:
        pass
    try:
        fd = t.funds_data
        out["top"], out["eq"] = fd.top_holdings, fd.equity_holdings
        out["sectors"], out["desc"] = dict(fd.sector_weightings or {}), fd.description or ""
    except Exception:
        pass
    return out


@st.cache_data(ttl=6 * 3600, show_spinner="กำลังดึงมุมมองนักวิเคราะห์ของหุ้นในดัชนี ...")
def holdings_table(etf: str) -> pd.DataFrame:
    """หุ้น 10 อันดับแรก + ข้อมูลนักวิเคราะห์รายตัว"""
    if etf == "SET50":  # THD ไม่มีรายชื่อ holdings -> ใช้ SET50 เรียงตามมูลค่าตลาด
        s = pd.read_csv(SET100_CSV, comment="#", encoding="utf-8-sig")
        syms = list(s.loc[s["set50"] == 1, "yahoo"])

        def cap(x):
            try:
                return x, float(yf.Ticker(x).fast_info["market_cap"])
            except Exception:
                return x, None
        with ThreadPoolExecutor(10) as ex:
            caps = dict(ex.map(cap, syms))
        cs = pd.Series(caps, dtype=float).dropna()
        w = (cs / cs.sum()).sort_values(ascending=False).head(10)
        top = pd.DataFrame({"symbol": w.index, "weight": w.values, "name": ""})
    else:
        th = etf_data(etf)["top"]
        if th is None or th.empty:
            return pd.DataFrame()
        top = pd.DataFrame({"symbol": [_fix_symbol(x, etf) for x in th.index],
                            "weight": th["Holding Percent"].values, "name": th["Name"].values})
    with ThreadPoolExecutor(10) as ex:
        infos = list(ex.map(_info, top["symbol"]))
    df = pd.concat([top.reset_index(drop=True), pd.DataFrame(infos)], axis=1)
    df["name"] = [n or i.get("longName") or i.get("shortName") or s
                  for n, i, s in zip(df["name"], infos, df["symbol"])]
    df["price"] = df["currentPrice"].fillna(df["regularMarketPrice"])
    df["upside"] = df["targetMeanPrice"] / df["price"] - 1
    return df


def _rec_label(v):
    """คะแนนนักวิเคราะห์ 1-5 -> คำไทย"""
    if v is None or pd.isna(v):
        return "-"
    return "ซื้ออย่างยิ่ง" if v < 1.5 else "ซื้อ" if v < 2.5 else "ถือ" if v < 3.5 else "ขาย"


def _wavg(v: pd.Series, w: pd.Series):
    m = v.notna() & w.notna()
    return (v[m] * w[m]).sum() / w[m].sum() if m.any() and w[m].sum() > 0 else None


def render_index_fundamentals(row, level=None):
    """row: แถวจาก world_indices.csv · level: ค่าดัชนีล่าสุด"""
    etf = row["etf"]
    st.markdown(CSS, unsafe_allow_html=True)
    proxy = "THD (iShares MSCI Thailand) + หุ้น SET50" if etf == "SET50" else etf
    e = etf_data("THD" if etf == "SET50" else etf)
    h = holdings_table(etf)

    # ----- ตัวชี้วัดระดับดัชนี -----
    eq = e["eq"]

    def inv(label):  # Yahoo เก็บเป็นส่วนกลับ (E/P, B/P ...) -> กลับเป็น P/E
        try:
            v = _num(eq.loc[label].iloc[0])
            return 1 / v if v else None
        except Exception:
            return None
    pe = _num(e["info"].get("trailingPE")) or inv("Price/Earnings")

    agg = {}
    if not h.empty:
        w = h["weight"]
        up = h["upside"].clip(*UPSIDE_CAP)
        agg["upside"] = _wavg(up, w)
        agg["rec"] = _wavg(h["recommendationMean"], w)
        fpe = h["forwardPE"].where(h["forwardPE"] > 0)
        m = fpe.notna()
        agg["fpe"] = w[m].sum() / (w[m] / fpe[m]).sum() if m.any() else None  # harmonic mean ถ่วงน้ำหนัก
        agg["growth"] = _wavg(h["earningsGrowth"].clip(-0.9, 3), w)
        fg = (h["forwardEps"] / h["trailingEps"] - 1).where(h["trailingEps"] > 0).clip(-0.9, 3)
        agg["fgrowth"] = _wavg(fg, w)
        agg["cov"] = w[h["targetMeanPrice"].notna()].sum()
        agg["capped"] = int(((h["upside"] < UPSIDE_CAP[0]) | (h["upside"] > UPSIDE_CAP[1])).sum())

    target = level * (1 + agg["upside"]) if level and agg.get("upside") is not None else None
    rec = None
    if agg.get("rec"):
        lab = _rec_label(agg["rec"])
        rec = (lab, {"ซื้ออย่างยิ่ง": "#1b8a3a", "ซื้อ": "#2ca02c", "ถือ": "#c99a06"}.get(lab, "#d62728"))

    st.markdown(f"### {row['country']} · {row['index_name']}")
    st.caption(f"ข้อมูลพื้นฐานจาก ETF ที่อิงดัชนี: **{proxy}** · มุมมองนักวิเคราะห์ = ค่าเฉลี่ยถ่วงน้ำหนักของหุ้น 10 อันดับแรก")

    cells = [
        metric_html("ระดับดัชนีล่าสุด", num(level, 2), "ค่าดัชนีล่าสุด"),
        metric_html("P/E ของดัชนี", num(pe, 2, " เท่า"), "ราคา ÷ กำไร ของหุ้นทั้งดัชนี (จาก ETF)"),
        metric_html("Forward P/E (Top 10)", num(agg.get("fpe"), 2, " เท่า"),
                    "P/E จากกำไรคาดการณ์ปีหน้า ของหุ้น 10 อันดับแรก (ถ่วงน้ำหนัก)"),
        metric_html("P/BV", num(inv("Price/Book"), 2, " เท่า"), "ราคา ÷ มูลค่าทางบัญชี ของทั้งดัชนี"),
        metric_html("P/S", num(inv("Price/Sales"), 2, " เท่า"), "ราคา ÷ ยอดขาย ของทั้งดัชนี"),
        metric_html("อัตราปันผล", pct(e["info"].get("yield"), 2), "อัตราปันผลของ ETF ที่อิงดัชนี"),
        metric_html("กำไรเติบโตล่าสุด (Top 10)", pct(agg.get("growth")),
                    "การเติบโตกำไรไตรมาสล่าสุด YoY ถ่วงน้ำหนัก (ตัดค่าสุดโต่ง)", "",
                    "pf-good" if (agg.get("growth") or 0) > 0 else "pf-bad" if agg.get("growth") else ""),
        metric_html("EPS growth คาดการณ์ (Top 10)", pct(agg.get("fgrowth")),
                    "EPS คาดการณ์ปีหน้า ÷ EPS 12 เดือนล่าสุด - 1 ถ่วงน้ำหนัก", "",
                    "pf-good" if (agg.get("fgrowth") or 0) > 0 else "pf-bad" if agg.get("fgrowth") else ""),
    ]
    st.markdown('<div class="pf-sec">📊 มูลค่าและการเติบโต</div><div class="pf-grid">' + "".join(cells) + "</div>",
                unsafe_allow_html=True)

    a_cells = [
        metric_html("ดัชนีเป้าหมายโดยนัย", num(target, 2),
                    "ระดับดัชนี × (1 + upside เฉลี่ยถ่วงน้ำหนักของราคาเป้าหมายหุ้น Top 10)",
                    "" if agg.get("upside") is None else f"{agg['upside'] * 100:+.1f}% จากปัจจุบัน",
                    "pf-good" if (agg.get("upside") or 0) > 0 else "pf-bad" if agg.get("upside") else ""),
        (f'<div class="pf-m" title="ค่าเฉลี่ยถ่วงน้ำหนักของคะแนนนักวิเคราะห์ 1 = ซื้ออย่างยิ่ง ... 5 = ขาย">'
         f'<div class="pf-l">มุมมองนักวิเคราะห์</div><div class="pf-v" style="color:{rec[1]}">{rec[0]}</div>'
         f'<div class="pf-n">คะแนน {agg["rec"]:.2f} / 5</div></div>') if rec else "",
        metric_html("ครอบคลุม", pct(agg.get("cov")), "สัดส่วนน้ำหนักในดัชนีของหุ้นที่มีราคาเป้าหมาย",
                    "ของน้ำหนักทั้งดัชนี"),
    ]
    st.markdown('<div class="pf-sec">🎯 มุมมองนักวิเคราะห์ (bottom-up)</div><div class="pf-grid">'
                + "".join(a_cells) + "</div>", unsafe_allow_html=True)
    if agg.get("capped"):
        st.warning(f"หุ้น {agg['capped']} ตัวใน Top 10 มีราคาเป้าหมายห่างจากราคาปัจจุบันผิดปกติ "
                   f"(เกิน {UPSIDE_CAP[1]:+.0%} หรือต่ำกว่า {UPSIDE_CAP[0]:+.0%}) จึงถูกจำกัดค่าก่อนเฉลี่ย "
                   "ข้อมูลนักวิเคราะห์ของตลาดนี้ใน Yahoo อาจไม่อัปเดต ควรใช้ตัวเลขด้วยความระวัง")
    if agg.get("cov") is not None and agg["cov"] < 0.2:
        st.warning(f"หุ้น 10 อันดับแรกมีน้ำหนักรวมแค่ {agg['cov'] * 100:.1f}% ของดัชนี "
                   "(ดัชนีที่มีหุ้นเล็กจำนวนมาก เช่น Russell 2000) ดัชนีเป้าหมายโดยนัยจึงไม่ค่อยเป็นตัวแทนทั้งดัชนี")

    # ----- holdings + sector -----
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown("**🏢 หุ้น 10 อันดับแรกในดัชนี (Holdings)**")
        if h.empty:
            st.caption("ไม่มีข้อมูลรายชื่อหุ้น")
        else:
            t = pd.DataFrame({
                "หุ้น": h["name"], "Symbol": h["symbol"], "น้ำหนัก": h["weight"] * 100,
                "ราคา": h["price"], "เป้าหมาย": h["targetMeanPrice"], "Upside": h["upside"] * 100,
                "มุมมอง": [_rec_label(v) for v in h["recommendationMean"]],
                "นักวิเคราะห์": h["numberOfAnalystOpinions"], "P/E": h["trailingPE"], "Fwd P/E": h["forwardPE"],
                "กำไรโต": h["earningsGrowth"] * 100,
            })
            st.dataframe(t, hide_index=True, use_container_width=True, column_config={
                "น้ำหนัก": st.column_config.ProgressColumn(format="%.1f%%", min_value=0,
                                                         max_value=float(t["น้ำหนัก"].max())),
                "ราคา": st.column_config.NumberColumn(format="%,.2f"),
                "เป้าหมาย": st.column_config.NumberColumn(format="%,.2f"),
                "Upside": st.column_config.NumberColumn(format="%+.1f%%"),
                "P/E": st.column_config.NumberColumn(format="%.1f"),
                "Fwd P/E": st.column_config.NumberColumn(format="%.1f"),
                "กำไรโต": st.column_config.NumberColumn(format="%+.0f%%")})
            st.caption(f"น้ำหนักรวม Top 10 = {h['weight'].sum() * 100:.1f}% ของดัชนี · ราคาเป็นสกุลเงินของแต่ละตลาด · "
                       f"upside รายตัวถูกจำกัดที่ {UPSIDE_CAP[0]:+.0%} ถึง {UPSIDE_CAP[1]:+.0%} ก่อนเฉลี่ย "
                       "(กันข้อมูลผิดปกติ)" + (" · น้ำหนักคำนวณจากมูลค่าตลาดใน SET50" if etf == "SET50" else ""))
    with c2:
        st.markdown("**🧩 สัดส่วนตามกลุ่มอุตสาหกรรม**")
        sec = {SECTOR_TH.get(k, k): v * 100 for k, v in (e["sectors"] or {}).items() if v and v > 0.001}
        if not sec:
            st.caption("ไม่มีข้อมูล")
        else:
            s = pd.Series(sec).sort_values()
            f = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color="#4c78a8",
                                 text=[f"{v:.1f}%" for v in s.values], textposition="outside"))
            f.update_layout(height=360, margin=dict(l=10, r=30, t=10, b=10), xaxis_title="%")
            st.plotly_chart(f, use_container_width=True)
    st.caption("⚠️ ดัชนีเป้าหมายโดยนัยไม่ใช่ตัวเลขที่นักวิเคราะห์ประกาศสำหรับดัชนีโดยตรง แต่คำนวณจากราคาเป้าหมายของหุ้นใหญ่ "
               "ใช้เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน")
