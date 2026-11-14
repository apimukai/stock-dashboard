"""หน้าข้อมูลพื้นฐานของหุ้น (ใช้ใน stock_dashboard.py)
ข้อมูลจาก Yahoo Finance ผ่าน yfinance: Ticker.info, income_stmt, dividends
"""
import datetime as dt
import html

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

SECTOR_TH = {
    "Technology": "เทคโนโลยี", "Financial Services": "การเงิน", "Energy": "พลังงาน",
    "Healthcare": "การแพทย์และสุขภาพ", "Consumer Cyclical": "สินค้าฟุ่มเฟือย/ค้าปลีก",
    "Consumer Defensive": "สินค้าอุปโภคบริโภคจำเป็น", "Industrials": "อุตสาหกรรม",
    "Basic Materials": "วัตถุดิบพื้นฐาน", "Real Estate": "อสังหาริมทรัพย์", "Utilities": "สาธารณูปโภค",
    "Communication Services": "สื่อสารและบริการ",
}
REC_TH = {"strong_buy": ("ซื้ออย่างยิ่ง", "#1b8a3a"), "buy": ("ซื้อ", "#2ca02c"), "hold": ("ถือ", "#c99a06"),
          "underperform": ("ต่ำกว่าตลาด", "#e07b00"), "sell": ("ขาย", "#d62728"), "strong_sell": ("ขาย", "#d62728")}

CSS = """
<style>
.pf-head{display:flex;flex-wrap:wrap;gap:6px;margin:2px 0 10px}
.pf-chip{background:rgba(128,128,128,.12);border-radius:999px;padding:3px 10px;font-size:.85rem}
.pf-sec{font-weight:700;font-size:1.05rem;margin:14px 0 6px}
.pf-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(165px,1fr));gap:8px}
.pf-m{border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:8px 12px;cursor:help}
.pf-l{font-size:.8rem;opacity:.7}
.pf-v{font-size:1.3rem;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.35}
.pf-n{font-size:.75rem;opacity:.65}
.pf-good .pf-v{color:#2ca02c}.pf-bad .pf-v{color:#d62728}
.pf-bar{position:relative;height:8px;border-radius:4px;margin:10px 0 4px;
  background:linear-gradient(90deg,#d62728,#c99a06,#2ca02c)}
.pf-dot{position:absolute;top:-4px;width:4px;height:16px;background:currentColor;border-radius:2px}
.pf-rng{display:flex;justify-content:space-between;font-size:.8rem;opacity:.75}
.pf-sum{line-height:1.6;font-size:.95rem}
</style>
"""


@st.cache_data(ttl=3600, show_spinner="กำลังดึงข้อมูลพื้นฐาน ...")
def get_info(sym: str) -> dict:
    try:
        return yf.Ticker(sym).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_financials(sym: str):
    t = yf.Ticker(sym)
    try:
        inc = t.income_stmt
    except Exception:
        inc = pd.DataFrame()
    try:
        div = t.dividends
    except Exception:
        div = pd.Series(dtype=float)
    return inc, div


# ---------- formatting ----------
def _num(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) or isinstance(v, str) else float(v)


def big(v, cur=""):
    v = _num(v)
    if v is None:
        return None
    a = abs(v)
    s = (f"{v / 1e12:,.2f} ล้านล้าน" if a >= 1e12 else f"{v / 1e9:,.2f} พันล้าน" if a >= 1e9
         else f"{v / 1e6:,.2f} ล้าน" if a >= 1e6 else f"{v:,.0f}")
    return f"{s} {cur}".strip()


def num(v, nd=2, suf=""):
    v = _num(v)
    return None if v is None else f"{v:,.{nd}f}{suf}"


def pct(v, nd=1, already=False):
    v = _num(v)
    return None if v is None else f"{v if already else v * 100:,.{nd}f}%"


def date_from_ts(v):
    v = _num(v)
    return None if v is None else dt.datetime.fromtimestamp(v, dt.timezone.utc).strftime("%d %b %Y")


def metric_html(label, value, hint, note="", tone=""):
    if value is None:
        return ""
    return (f'<div class="pf-m {tone}" title="{html.escape(hint)}"><div class="pf-l">{label}</div>'
            f'<div class="pf-v">{value}</div><div class="pf-n">{note}</div></div>')


def section(title, items):
    cells = "".join(metric_html(*it) for it in items)
    if not cells:
        return ""
    return f'<div class="pf-sec">{title}</div><div class="pf-grid">{cells}</div>'


def tone(v, good, bad):
    """good/bad: ฟังก์ชันตรวจค่า -> คลาสสีเขียว/แดง"""
    v = _num(v)
    if v is None:
        return ""
    return "pf-good" if good(v) else "pf-bad" if bad(v) else ""


# ---------- main renderers ----------
def render_header(sym: str, display_name: str = "", chips_on: bool = True) -> bool:
    """ชื่อบริษัท ราคา และป้ายข้อมูล คืน True ถ้าเป็นหุ้น (มีงบการเงิน)"""
    i = get_info(sym)
    cur = i.get("currency") or ""
    price = _num(i.get("regularMarketPrice") or i.get("currentPrice"))
    prev = _num(i.get("regularMarketPreviousClose") or i.get("previousClose"))
    name = i.get("longName") or i.get("shortName") or display_name or sym
    is_stock = i.get("quoteType") == "EQUITY"

    st.markdown(CSS, unsafe_allow_html=True)
    h1, h2 = st.columns([3, 1])
    h1.markdown(f"### {name}  \n`{sym}`")
    if price is not None:
        h2.metric(f"ราคาล่าสุด ({cur})", f"{price:,.2f}",
                  None if not prev else f"{price - prev:+,.2f} ({(price / prev - 1) * 100:+.2f}%)")

    chips = []
    if i.get("sector"):
        chips.append(f"🏷️ {SECTOR_TH.get(i['sector'], i['sector'])} ({i['sector']})")
    if i.get("industry"):
        chips.append(f"🏭 {i['industry']}")
    if i.get("country"):
        chips.append(f"🌏 {i['country']}")
    if i.get("fullTimeEmployees"):
        chips.append(f"👥 พนักงาน {i['fullTimeEmployees']:,} คน")
    if i.get("website"):
        w = html.escape(i["website"])
        chips.append(f'🔗 <a href="{w}" target="_blank">{w.replace("https://", "").replace("http://", "")}</a>')
    if chips and chips_on:
        st.markdown('<div class="pf-head">' + "".join(f'<span class="pf-chip">{c}</span>' for c in chips)
                    + "</div>", unsafe_allow_html=True)
    return is_stock


def render_metrics(sym: str):
    """ตัวชี้วัดพื้นฐาน + ช่วงราคา 52 สัปดาห์ + ลักษณะธุรกิจ"""
    i = get_info(sym)
    cur = i.get("currency") or ""
    price = _num(i.get("regularMarketPrice") or i.get("currentPrice"))
    st.markdown(CSS, unsafe_allow_html=True)

    # ปันผล 12 เดือนล่าสุด คำนวณจากประวัติจริง (ค่า dividendRate ของ Yahoo บางตัวไม่ตรง)
    _, div = get_financials(sym)
    ttm_div = None
    if div is not None and len(div):
        d = div.copy()
        d.index = d.index.tz_localize(None) if d.index.tz is not None else d.index
        ttm_div = float(d[d.index >= pd.Timestamp.today() - pd.Timedelta(days=365)].sum())
    ttm_yield = ttm_div / price if ttm_div and price else None

    html_parts = [
        section("💰 มูลค่าหุ้น (Valuation)", [
            ("มูลค่าตลาด (Market Cap)", big(i.get("marketCap"), cur), "ราคาหุ้น × จำนวนหุ้นทั้งหมด = ขนาดของบริษัท"),
            ("P/E (12 เดือนล่าสุด)", num(i.get("trailingPE"), 2, " เท่า"),
             "ราคา ÷ กำไรต่อหุ้น 12 เดือนล่าสุด: จ่ายกี่บาทต่อกำไร 1 บาท ควรเทียบกับหุ้นกลุ่มเดียวกัน"),
            ("Forward P/E", num(i.get("forwardPE"), 2, " เท่า"), "ราคา ÷ กำไรต่อหุ้นที่นักวิเคราะห์คาดปีหน้า"),
            ("P/BV", num(i.get("priceToBook"), 2, " เท่า"),
             "ราคา ÷ มูลค่าทางบัญชีต่อหุ้น: ต่ำกว่า 1 = ราคาต่ำกว่าส่วนของผู้ถือหุ้นตามบัญชี"),
            ("PEG", num(i.get("trailingPegRatio") or i.get("pegRatio"), 2),
             "P/E ÷ อัตราเติบโตของกำไร: ราว 1 ถือว่าสมเหตุสมผล ต่ำกว่า 1 อาจถูกเมื่อเทียบกับการเติบโต"),
            ("EV/EBITDA", num(i.get("enterpriseToEbitda"), 2, " เท่า"),
             "มูลค่ากิจการ ÷ กำไรก่อนดอกเบี้ย ภาษี ค่าเสื่อม: ใช้เทียบบริษัทที่มีหนี้ต่างกัน"),
        ]),
        section("📊 กำไรและประสิทธิภาพ", [
            ("EPS (12 เดือนล่าสุด)", num(i.get("trailingEps"), 2, f" {cur}"), "กำไรสุทธิต่อหุ้น 12 เดือนล่าสุด"),
            ("Forward EPS", num(i.get("forwardEps"), 2, f" {cur}"), "กำไรต่อหุ้นที่นักวิเคราะห์คาดการณ์"),
            ("อัตรากำไรสุทธิ", pct(i.get("profitMargins")), "กำไรสุทธิ ÷ รายได้: ขายได้ 100 บาท เหลือกำไรกี่บาท",
             "", tone(i.get("profitMargins"), lambda v: v >= .1, lambda v: v < 0)),
            ("ROE", pct(i.get("returnOnEquity")), "กำไร ÷ ส่วนของผู้ถือหุ้น: ใช้เงินผู้ถือหุ้นทำกำไรได้ดีแค่ไหน (>15% ถือว่าดี)",
             "", tone(i.get("returnOnEquity"), lambda v: v >= .15, lambda v: v < 0)),
            ("ROA", pct(i.get("returnOnAssets")), "กำไร ÷ สินทรัพย์รวม: ใช้สินทรัพย์ทำกำไรได้ดีแค่ไหน",
             "", tone(i.get("returnOnAssets"), lambda v: v >= .05, lambda v: v < 0)),
            ("รายได้เติบโต (YoY)", pct(i.get("revenueGrowth")), "รายได้ไตรมาสล่าสุดเทียบกับไตรมาสเดียวกันปีก่อน",
             "", tone(i.get("revenueGrowth"), lambda v: v > 0, lambda v: v < 0)),
            ("กำไรเติบโต (YoY)", pct(i.get("earningsGrowth")), "กำไรไตรมาสล่าสุดเทียบกับไตรมาสเดียวกันปีก่อน",
             "", tone(i.get("earningsGrowth"), lambda v: v > 0, lambda v: v < 0)),
            ("รายได้ 12 เดือน", big(i.get("totalRevenue"), cur), "รายได้รวม 12 เดือนล่าสุด"),
        ]),
        section("💵 เงินปันผล", [
            ("อัตราปันผล (12 เดือน)", pct(ttm_yield, 2), "ปันผลที่จ่ายจริงใน 12 เดือนล่าสุด ÷ ราคาปัจจุบัน",
             f"{ttm_div:,.2f} {cur}/หุ้น" if ttm_div else ""),
            ("ปันผลเฉลี่ย 5 ปี", pct(i.get("fiveYearAvgDividendYield"), 2, already=True),
             "อัตราปันผลเฉลี่ยย้อนหลัง 5 ปี"),
            ("Payout Ratio", pct(i.get("payoutRatio")),
             "สัดส่วนกำไรที่จ่ายเป็นปันผล: สูงเกิน 100% = จ่ายมากกว่ากำไร อาจไม่ยั่งยืน",
             "", tone(i.get("payoutRatio"), lambda v: 0 < v <= .8, lambda v: v > 1)),
            ("ปันผลครั้งล่าสุด", num(i.get("lastDividendValue"), 2, f" {cur}"), "เงินปันผลต่อหุ้นงวดล่าสุด"),
            ("วัน XD ล่าสุด", date_from_ts(i.get("exDividendDate")),
             "วันที่ซื้อหุ้นแล้วจะไม่ได้รับปันผลงวดนั้น (ต้องถือก่อนวันนี้)"),
        ]),
        section("⚖️ ฐานะการเงินและความเสี่ยง", [
            ("หนี้สิน/ทุน (D/E)", num((_num(i.get("debtToEquity")) or 0) / 100 if i.get("debtToEquity") else None, 2, " เท่า"),
             "หนี้สินที่มีดอกเบี้ย ÷ ส่วนของผู้ถือหุ้น: ยิ่งสูงยิ่งพึ่งหนี้มาก (ธนาคารมักไม่มีค่านี้)",
             "", tone(i.get("debtToEquity"), lambda v: v < 100, lambda v: v > 200)),
            ("Current Ratio", num(i.get("currentRatio"), 2, " เท่า"),
             "สินทรัพย์หมุนเวียน ÷ หนี้สินหมุนเวียน: >1 = มีสภาพคล่องพอจ่ายหนี้ระยะสั้น",
             "", tone(i.get("currentRatio"), lambda v: v >= 1.2, lambda v: v < 1)),
            ("Beta", num(i.get("beta"), 2),
             "ความผันผวนเทียบตลาด: 1 = เท่าตลาด, >1 = แกว่งแรงกว่าตลาด, <1 = แกว่งน้อยกว่า"),
            ("กระแสเงินสดอิสระ", big(i.get("freeCashflow"), cur),
             "เงินสดจากการดำเนินงานหลังหักลงทุน: เงินที่เหลือไว้จ่ายปันผล/ลดหนี้"),
            ("ผู้บริหาร/ผู้ถือหุ้นใหญ่ถือ", pct(i.get("heldPercentInsiders")), "สัดส่วนหุ้นที่ผู้บริหาร/ผู้ถือหุ้นใหญ่ถือ"),
            ("สถาบันถือ", pct(i.get("heldPercentInstitutions")), "สัดส่วนหุ้นที่กองทุน/สถาบันถือ"),
        ]),
    ]

    # 52W range
    lo, hi = _num(i.get("fiftyTwoWeekLow")), _num(i.get("fiftyTwoWeekHigh"))
    rng = ""
    if lo is not None and hi and price is not None and hi > lo:
        pos = max(0, min(100, (price - lo) / (hi - lo) * 100))
        rng = (f'<div class="pf-m" title="ราคาปัจจุบันอยู่ตรงไหนของช่วงต่ำสุด-สูงสุด 52 สัปดาห์">'
               f'<div class="pf-l">ช่วงราคา 52 สัปดาห์</div><div class="pf-bar"><div class="pf-dot" '
               f'style="left:calc({pos:.1f}% - 2px)"></div></div><div class="pf-rng"><span>{lo:,.2f}</span>'
               f'<span>ปัจจุบันที่ {pos:.0f}% ของช่วง</span><span>{hi:,.2f}</span></div></div>')
    if rng:
        html_parts.append('<div class="pf-sec">📍 ราคาเทียบช่วง 52 สัปดาห์</div>'
                          f'<div class="pf-grid" style="grid-template-columns:minmax(260px,520px)">{rng}</div>')

    st.markdown("".join(html_parts), unsafe_allow_html=True)
    st.caption("💡 ชี้เมาส์ที่แต่ละช่องเพื่อดูความหมาย · สีเขียว/แดงเป็นเกณฑ์ทั่วไปเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน")

    if i.get("longBusinessSummary"):
        with st.expander("🏢 บริษัทนี้ทำธุรกิจอะไร (คำอธิบายจาก Yahoo Finance ภาษาอังกฤษ)", expanded=True):
            st.markdown(f'<div class="pf-sum">{html.escape(i["longBusinessSummary"])}</div>', unsafe_allow_html=True)


def render_financials(sym: str):
    """กราฟงบการเงินรายปี + ประวัติเงินปันผล"""
    i = get_info(sym)
    cur = i.get("currency") or ""
    inc, div = get_financials(sym)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**📑 รายได้และกำไรสุทธิรายปี**")
        if inc is None or inc.empty or "Total Revenue" not in inc.index:
            st.caption("ไม่มีข้อมูลงบการเงิน")
        else:
            cols = sorted(inc.columns)
            years = [c.strftime("%Y") for c in cols]
            rev = inc.loc["Total Revenue", cols] / 1e9
            ni = inc.loc["Net Income", cols] / 1e9 if "Net Income" in inc.index else None
            eps_row = "Diluted EPS" if "Diluted EPS" in inc.index else "Basic EPS" if "Basic EPS" in inc.index else None
            f = make_subplots(specs=[[{"secondary_y": True}]])
            f.add_trace(go.Bar(x=years, y=rev, name="รายได้", marker_color="#4c78a8"))
            if ni is not None:
                f.add_trace(go.Bar(x=years, y=ni, name="กำไรสุทธิ",
                                   marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in ni.fillna(0)]))
            if eps_row:
                f.add_trace(go.Scatter(x=years, y=inc.loc[eps_row, cols], name="EPS", mode="lines+markers",
                                       line=dict(color="#e45756", width=2)), secondary_y=True)
            f.update_layout(barmode="group", height=360, margin=dict(l=10, r=10, t=10, b=10),
                            legend=dict(orientation="h", y=1.12), hovermode="x unified")
            f.update_yaxes(title_text=f"พันล้าน {cur}", secondary_y=False)
            f.update_yaxes(title_text=f"EPS ({cur})", secondary_y=True, showgrid=False)
            st.plotly_chart(f, use_container_width=True)
            if ni is not None:
                margin = (ni / rev * 100).round(1)
                st.caption("อัตรากำไรสุทธิ: " + " · ".join(f"{y} {m:.1f}%" for y, m in zip(years, margin)
                                                           if pd.notna(m)))

    with c2:
        st.markdown("**💵 เงินปันผลต่อหุ้นรายปี**")
        if div is None or not len(div):
            st.caption("ไม่มีประวัติการจ่ายเงินปันผล")
        else:
            d = div.copy()
            d.index = d.index.tz_localize(None) if d.index.tz is not None else d.index
            yearly = d.groupby(d.index.year).sum().tail(10)
            this_year = dt.date.today().year
            f = go.Figure(go.Bar(x=[str(y) + (" (ยังไม่ครบปี)" if y == this_year else "") for y in yearly.index],
                                 y=yearly.values, marker_color="#59a14f",
                                 text=[f"{v:,.2f}" for v in yearly.values], textposition="outside"))
            f.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=f"{cur}/หุ้น")
            st.plotly_chart(f, use_container_width=True)
            st.caption("งวดล่าสุด: " + " · ".join(f"{ix:%d %b %Y} = {v:,.2f}" for ix, v in d.tail(4)[::-1].items()))


# ---------- analyst targets & forecasts ----------
@st.cache_data(ttl=6 * 3600, show_spinner="กำลังดึงข้อมูลนักวิเคราะห์ ...")
def get_estimates(sym: str) -> dict:
    t = yf.Ticker(sym)
    out = {}
    for a in ["analyst_price_targets", "recommendations", "revenue_estimate", "earnings_estimate",
              "eps_trend", "earnings_history"]:
        try:
            out[a] = getattr(t, a)
        except Exception:
            out[a] = None
    return out


def _df(x):
    return x if isinstance(x, pd.DataFrame) and not x.empty else None


def _fy_labels(sym):
    """ปีบัญชีของประมาณการ 0y/+1y = ปีงบล่าสุดที่ประกาศแล้ว +1/+2 (NVDA ปีงบจบ ม.ค.)"""
    inc, _ = get_financials(sym)
    if inc is not None and not inc.empty:
        last = max(inc.columns).year
    else:
        last = dt.date.today().year - 1
    return {"0q": "ไตรมาสนี้", "+1q": "ไตรมาสหน้า", "0y": f"ปีนี้ (FY{last + 1})", "+1y": f"ปีหน้า (FY{last + 2})"}, last


def render_analyst(sym: str):
    i = get_info(sym)
    cur = i.get("currency") or ""
    est = get_estimates(sym)
    price = _num(i.get("regularMarketPrice") or i.get("currentPrice"))
    st.markdown(CSS, unsafe_allow_html=True)

    # ----- ราคาเป้าหมาย -----
    pt = est.get("analyst_price_targets") or {}
    mean = _num(pt.get("mean"))
    if not mean:
        st.info("ยังไม่มีนักวิเคราะห์ให้ราคาเป้าหมายหุ้นตัวนี้ใน Yahoo Finance")
    else:
        price = _num(pt.get("current")) or price
        n_an = i.get("numberOfAnalystOpinions")
        st.markdown(f"##### 🎯 ราคาเป้าหมายของนักวิเคราะห์ ({n_an or '-'} ราย)")

        def cell(label, v, hint):
            up = (v / price - 1) if v and price else None
            return metric_html(label, num(v, 2, f" {cur}"), hint,
                               "" if up is None else f"{up * 100:+.1f}% จากราคาปัจจุบัน",
                               "pf-good" if up and up > 0 else "pf-bad" if up and up < 0 else "")

        rec = REC_TH.get(i.get("recommendationKey") or "")
        rec_cell = ""
        if rec:
            score = _num(i.get("recommendationMean"))
            rec_cell = (f'<div class="pf-m" title="ความเห็นรวมของนักวิเคราะห์ (คะแนน 1 = ซื้ออย่างยิ่ง ... 5 = ขาย)">'
                        f'<div class="pf-l">ความเห็นรวม</div><div class="pf-v" style="color:{rec[1]}">{rec[0]}</div>'
                        f'<div class="pf-n">คะแนน {score:.2f} / 5</div></div>' if score else
                        f'<div class="pf-m"><div class="pf-l">ความเห็นรวม</div>'
                        f'<div class="pf-v" style="color:{rec[1]}">{rec[0]}</div></div>')
        st.markdown('<div class="pf-grid">'
                    + metric_html("ราคาปัจจุบัน", num(price, 2, f" {cur}"), "ราคาล่าสุด")
                    + cell("เป้าต่ำสุด", _num(pt.get("low")), "ราคาเป้าหมายต่ำสุดในบรรดานักวิเคราะห์")
                    + cell("เป้าเฉลี่ย", mean, "ค่าเฉลี่ยราคาเป้าหมายของนักวิเคราะห์ทุกราย")
                    + cell("เป้ามัธยฐาน", _num(pt.get("median")), "ค่ากลางของราคาเป้าหมาย (ไม่ถูกดึงด้วยค่าสุดโต่ง)")
                    + cell("เป้าสูงสุด", _num(pt.get("high")), "ราคาเป้าหมายสูงสุดในบรรดานักวิเคราะห์")
                    + rec_cell + "</div>", unsafe_allow_html=True)

        # แถบช่วงราคาเป้าหมาย
        lo, hi = _num(pt.get("low")), _num(pt.get("high"))
        f = go.Figure()
        f.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode="lines", line=dict(color="rgba(128,128,128,.5)", width=10),
                               hoverinfo="skip", showlegend=False))
        pts = [("ต่ำสุด", lo, "#d62728"), ("มัธยฐาน", _num(pt.get("median")), "#c99a06"),
               ("เฉลี่ย", mean, "#2ca02c"), ("สูงสุด", hi, "#1b8a3a")]
        for k, (lab, v, c) in enumerate(pts):
            if v:
                f.add_trace(go.Scatter(x=[v], y=[0], mode="markers+text", marker=dict(size=14, color=c),
                                       text=[f"{lab}<br>{v:,.2f}"], textposition="top center" if k % 2 else "bottom center",
                                       name=lab, hovertemplate=f"{lab}: %{{x:,.2f}}<extra></extra>", showlegend=False))
        if price:
            f.add_trace(go.Scatter(x=[price], y=[0], mode="markers+text", marker=dict(size=18, symbol="diamond",
                                   color="#4c78a8", line=dict(width=2, color="white")),
                                   text=[f"ราคาปัจจุบัน<br>{price:,.2f}"], textposition="top center",
                                   hovertemplate="ราคาปัจจุบัน: %{x:,.2f}<extra></extra>", showlegend=False))
        f.update_layout(height=190, margin=dict(l=20, r=20, t=10, b=10), yaxis=dict(visible=False, range=[-1, 1]),
                        xaxis=dict(title=cur))
        st.plotly_chart(f, use_container_width=True)

    # ----- คำแนะนำซื้อ/ถือ/ขาย -----
    rc = _df(est.get("recommendations"))
    if rc is not None and {"strongBuy", "buy", "hold", "sell", "strongSell"} <= set(rc.columns):
        rc = rc.head(4)
        ylab = {"0m": "เดือนนี้", "-1m": "1 เดือนก่อน", "-2m": "2 เดือนก่อน", "-3m": "3 เดือนก่อน"}
        y = [ylab.get(p, p) for p in rc["period"]]
        f = go.Figure()
        for colname, lab, c in [("strongBuy", "ซื้ออย่างยิ่ง", "#1b8a3a"), ("buy", "ซื้อ", "#7cc47f"),
                                ("hold", "ถือ", "#f2c94c"), ("sell", "ขาย", "#f2994a"), ("strongSell", "ขายอย่างยิ่ง", "#d62728")]:
            f.add_trace(go.Bar(y=y, x=rc[colname], name=lab, orientation="h", marker_color=c,
                               text=[v if v else "" for v in rc[colname]], textposition="inside"))
        f.update_layout(barmode="stack", height=240, margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", y=1.15), yaxis=dict(autorange="reversed"), xaxis_title="จำนวนนักวิเคราะห์")
        st.markdown("##### 🗳️ คำแนะนำของนักวิเคราะห์ (ย้อนหลัง 4 เดือน)")
        st.plotly_chart(f, use_container_width=True)

    # ----- ประมาณการรายได้และ EPS -----
    rev, eps = _df(est.get("revenue_estimate")), _df(est.get("earnings_estimate"))
    labels, last_fy = _fy_labels(sym)
    if rev is None and eps is None:
        st.caption("ไม่มีข้อมูลประมาณการรายได้/กำไรจากนักวิเคราะห์")
        return
    st.markdown("##### 🔮 ประมาณการรายได้และ EPS (เทียบกับผลจริงย้อนหลัง)")
    inc, _ = get_financials(sym)
    cols = sorted(inc.columns)[-4:] if inc is not None and not inc.empty else []
    c1, c2 = st.columns(2)

    def forecast_chart(target, title, actual_row, est_df, scale, unit):
        f = go.Figure()
        if actual_row is not None and cols:
            act = inc.loc[actual_row, cols] / scale
            f.add_trace(go.Bar(x=[f"FY{c.year}" for c in cols], y=act, name="ผลจริง", marker_color="#4c78a8",
                               text=[f"{v:,.2f}" for v in act], textposition="outside"))
        if est_df is not None:
            e = est_df.loc[[p for p in ("0y", "+1y") if p in est_df.index]]
            e = e[e["avg"].fillna(0) != 0]
            if len(e):
                avg, lo, hi = e["avg"] / scale, e["low"] / scale, e["high"] / scale
                f.add_trace(go.Bar(x=[f"FY{last_fy + (1 if p == '0y' else 2)} (คาด)" for p in e.index], y=avg,
                                   name="ประมาณการ (เฉลี่ย)", marker_color="#f58518",
                                   error_y=dict(type="data", symmetric=False, array=hi - avg, arrayminus=avg - lo,
                                                color="rgba(90,90,90,.8)"),
                                   text=[f"{v:,.2f}<br>({g * 100:+.0f}%)" if pd.notna(g) else f"{v:,.2f}"
                                         for v, g in zip(avg, e["growth"])],
                                   textposition="outside",
                                   customdata=list(zip(lo, hi, e["numberOfAnalysts"])),
                                   hovertemplate="เฉลี่ย %{y:,.2f}<br>ต่ำสุด %{customdata[0]:,.2f}<br>"
                                                 "สูงสุด %{customdata[1]:,.2f}<br>นักวิเคราะห์ %{customdata[2]} ราย<extra></extra>"))
        f.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10), yaxis_title=unit,
                        legend=dict(orientation="h", y=-0.15), showlegend=True)
        target.plotly_chart(f, use_container_width=True)

    eps_row = "Diluted EPS" if inc is not None and "Diluted EPS" in inc.index else \
        "Basic EPS" if inc is not None and "Basic EPS" in inc.index else None
    rev_row = "Total Revenue" if inc is not None and "Total Revenue" in inc.index else None
    forecast_chart(c1, "รายได้", rev_row, rev, 1e9, f"พันล้าน {cur}")
    forecast_chart(c2, "กำไรต่อหุ้น (EPS)", eps_row, eps, 1, cur)
    st.caption("แท่งสีส้ม = ค่าเฉลี่ยประมาณการ เส้นขีด = ช่วงต่ำสุด-สูงสุดของนักวิเคราะห์ · ตัวเลขในวงเล็บ = อัตราเติบโตที่คาด · "
               "EPS ประมาณการเป็นแบบ normalized (ไม่รวมรายการพิเศษ) จึงอาจต่างจาก EPS ในงบ")

    # ตารางประมาณการ
    rows = []
    for p in ("0q", "+1q", "0y", "+1y"):
        r = {"ช่วงเวลา": labels[p]}
        ok = False
        for d, pre, sc in [(rev, "รายได้", 1e9), (eps, "EPS", 1)]:
            if d is not None and p in d.index and pd.notna(d.loc[p, "avg"]) and d.loc[p, "avg"] != 0:
                ok = True
                r[f"{pre} คาด"] = d.loc[p, "avg"] / sc
                r[f"{pre} ต่ำ-สูง"] = f"{d.loc[p, 'low'] / sc:,.2f} – {d.loc[p, 'high'] / sc:,.2f}"
                r[f"{pre} โต %"] = d.loc[p, "growth"] * 100 if pd.notna(d.loc[p, "growth"]) else None
                r[f"{pre} นักวิเคราะห์"] = int(d.loc[p, "numberOfAnalysts"]) if pd.notna(d.loc[p, "numberOfAnalysts"]) else None
        if ok:
            rows.append(r)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     column_config={"รายได้ คาด": st.column_config.NumberColumn(f"รายได้คาด (พันล้าน {cur})", format="%,.2f"),
                                    "EPS คาด": st.column_config.NumberColumn(f"EPS คาด ({cur})", format="%,.2f"),
                                    "รายได้ โต %": st.column_config.NumberColumn(format="%+.1f%%"),
                                    "EPS โต %": st.column_config.NumberColumn(format="%+.1f%%")})

    c3, c4 = st.columns(2)
    # การปรับประมาณการ EPS
    tr = _df(est.get("eps_trend"))
    if tr is not None:
        tr = tr.loc[[p for p in ("0y", "+1y") if p in tr.index]]
        tr = tr[tr["current"].fillna(0) != 0]
        if len(tr):
            with c3:
                st.markdown("**📝 นักวิเคราะห์ปรับประมาณการ EPS อย่างไร**")
                t = pd.DataFrame({"ช่วงเวลา": [labels[p] for p in tr.index],
                                  "90 วันก่อน": tr["90daysAgo"], "30 วันก่อน": tr["30daysAgo"],
                                  "7 วันก่อน": tr["7daysAgo"], "ตอนนี้": tr["current"]})
                t["เปลี่ยน vs 90 วัน"] = (tr["current"] / tr["90daysAgo"].replace(0, pd.NA) - 1) * 100
                fmt2 = st.column_config.NumberColumn(format="%,.2f")
                st.dataframe(t, hide_index=True, use_container_width=True,
                             column_config={"90 วันก่อน": fmt2, "30 วันก่อน": fmt2, "7 วันก่อน": fmt2, "ตอนนี้": fmt2,
                                            "เปลี่ยน vs 90 วัน": st.column_config.NumberColumn(format="%+.1f%%")})
                st.caption("ปรับขึ้นต่อเนื่อง = นักวิเคราะห์มองกำไรดีขึ้น · ปรับลง = มองแย่ลง")

    # ผลประกอบการจริงเทียบประมาณการ
    eh = _df(est.get("earnings_history"))
    if eh is not None and {"epsActual", "epsEstimate"} <= set(eh.columns):
        eh = eh.dropna(subset=["epsActual"])
        with c4:
            st.markdown("**📣 EPS จริงเทียบประมาณการ (4 ไตรมาสล่าสุด)**")
            q = [pd.Timestamp(x).strftime("%b %Y") for x in eh.index]
            f = go.Figure()
            f.add_trace(go.Bar(x=q, y=eh["epsEstimate"], name="ประมาณการ", marker_color="#bab0ac"))
            f.add_trace(go.Bar(x=q, y=eh["epsActual"], name="จริง",
                               marker_color=["#2ca02c" if a >= e else "#d62728"
                                             for a, e in zip(eh["epsActual"], eh["epsEstimate"].fillna(0))],
                               text=[f"{s * 100:+.1f}%" if pd.notna(s) else "" for s in eh.get("surprisePercent", [])],
                               textposition="outside"))
            f.update_layout(barmode="group", height=300, margin=dict(l=10, r=10, t=10, b=10),
                            legend=dict(orientation="h", y=1.15), yaxis_title=cur)
            st.plotly_chart(f, use_container_width=True)
            st.caption("เขียว = ดีกว่าคาด (beat) · แดง = แย่กว่าคาด (miss) · % = ส่วนต่างจากประมาณการ")
