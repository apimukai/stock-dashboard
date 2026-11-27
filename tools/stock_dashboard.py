"""Streamlit dashboard: ราคาหุ้น/สินทรัพย์แบบเกือบ Real-time จาก Yahoo Finance (ผ่าน yfinance)

ติดตั้ง:  pip install -r requirements-dashboard.txt
รัน:      streamlit run stock_dashboard.py
(หรือดับเบิลคลิก run_dashboard.bat)

หมายเหตุ: ข้อมูล Yahoo Finance ฟรีมักดีเลย์ ~0-20 นาทีแล้วแต่ตลาด (หุ้นสหรัฐฯ ส่วนใหญ่เกือบ real-time,
SET ดีเลย์ ~15 นาที) ใช้เพื่อการศึกษา/ติดตาม ไม่ใช่สัญญาณซื้อขาย
"""
import os
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

from stock_profile import render_header, render_metrics, render_analyst, render_financials
from world_markets import load_world, get_world, render_world, flag_url
from index_fundamentals import render_index_fundamentals
from sector_heatmap import render_sector_heatmap, clickable_table
from history_backtest import render_history_tab
from set_signals import render_signals_hub
from market_mood import render_mood_tab
from market_allocation import render_allocation_tab
from mobile_ui import apply_mobile_ui
from daily_report import render_daily_report
from tv_chart import lw_chart, tv_widget, tv_symbol, signal_events, x7_panel, MARKERS, DEFAULT_MARKERS

st.set_page_config(page_title="Stock Dashboard", page_icon="📈", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
META_CSV = os.path.join(HERE, "..", "files", "assets_meta.csv")
PRICES_CSV = os.path.join(HERE, "..", "files", "assets_prices.csv")
SET100_CSV = os.path.join(HERE, "..", "files", "set100.csv")  # รายชื่อจาก set.or.th อัปเดตทุก ม.ค./ก.ค.
US_CSV = os.path.join(HERE, "..", "files", "us_indices.csv")  # สร้างด้วย tools/fetch_us_lists.py

DEFAULT_WATCHLIST = ["^GSPC", "^IXIC", "^SET.BK", "NVDA", "AAPL", "MSFT", "TSLA", "NBIS", "CRWV", "DRAM", "RAM", "SVIX", "^VIX", "SMH", "QQQ", "SQQQ", "JEPQ", "GC=F", "BTC-USD", "THB=X"]

# period -> interval ที่ Yahoo อนุญาต (intraday ย้อนหลังได้จำกัด)
PERIODS = {
    "1 วัน": ("1d", "1m"),
    "5 วัน": ("5d", "5m"),
    "1 เดือน": ("1mo", "30m"),
    "3 เดือน": ("3mo", "1d"),
    "6 เดือน": ("6mo", "1d"),
    "YTD": ("ytd", "1d"),
    "1 ปี": ("1y", "1d"),
    "2 ปี": ("2y", "1d"),
    "5 ปี": ("5y", "1wk"),
    "สูงสุด": ("max", "1mo"),
}


@st.cache_data(ttl=3600)
def load_names() -> dict:
    """ชื่อไทยจาก assets_meta.csv ของโปรเจกต์ (ถ้ามี)"""
    try:
        df = pd.read_csv(META_CSV, encoding="utf-8-sig")
        return dict(zip(df["symbol"], df["name"]))
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def get_quote(sym: str) -> dict:
    """ราคาล่าสุด + ราคาปิดก่อนหน้า (cache 60 วินาที)"""
    t = yf.Ticker(sym)
    fi = t.fast_info
    last = prev = None
    try:
        last = float(fi["last_price"])
        prev = float(fi["previous_close"])
    except Exception:
        pass
    if last is None or pd.isna(last):
        h = t.history(period="5d", interval="1d")
        if not h.empty:
            last = float(h["Close"].iloc[-1])
            prev = float(h["Close"].iloc[-2]) if len(h) > 1 else None
    cur = None
    try:
        cur = fi["currency"]
    except Exception:
        pass
    return {"last": last, "prev": prev, "currency": cur}


@st.cache_data(ttl=3600, show_spinner=False)
def local_history(sym: str) -> pd.DataFrame:
    """ราคาปิดรายวันจาก files/assets_prices.csv (ใช้กับสัญลักษณ์ที่ Yahoo ไม่มีย้อนหลัง เช่น ^SET.BK)"""
    try:
        p = pd.read_csv(PRICES_CSV, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()
    p = p[p["symbol"] == sym]
    if p.empty:
        return pd.DataFrame()
    s = p.set_index(pd.to_datetime(p["date"]))["close"].astype(float).sort_index()
    return pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s, "Volume": 0})


PERIOD_DAYS = {"3mo": 92, "6mo": 183, "1y": 366, "2y": 731, "5y": 1827}
NO_DAILY_HISTORY = {"^SET.BK"}  # Yahoo ไม่มีแท่งรายวันย้อนหลัง (มีแค่วันล่าสุด) และแท่งรายเดือนขาดช่วง


@st.cache_data(ttl=1800, show_spinner=False)
def hourly_daily(sym: str) -> pd.DataFrame:
    """แท่งรายวัน (เปิด/สูง/ต่ำ/ปิด) สร้างจากแท่งรายชั่วโมงของ Yahoo (ย้อนหลังได้ ~2 ปี) สำหรับดัชนีที่ Yahoo ไม่มีแท่งรายวันย้อนหลัง
    เช่น ^SET.BK · ราคาปิดแทนด้วยราคาปิดทางการจากไฟล์ในโปรเจกต์ (แท่งชั่วโมงไม่รวมช่วงจับคู่ปิดตลาด คลาด ~0.1%)"""
    try:
        h = yf.Ticker(sym).history(period="730d", interval="1h", auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    if h.empty:
        return pd.DataFrame()
    h.index = h.index.tz_convert("Asia/Bangkok").tz_localize(None) if h.index.tz is not None else h.index
    d = h.resample("D").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    d = d.dropna(subset=["Close"])
    off = local_history(sym)
    if not off.empty:
        c = off["Close"].reindex(d.index)
        d["Close"] = c.fillna(d["Close"])
    try:  # แท่งวันล่าสุดแบบทางการ (Yahoo มีให้เฉพาะวันล่าสุด)
        t = yf.Ticker(sym).history(period="5d", interval="1d", auto_adjust=False).dropna(subset=["Close"])
        if len(t):
            t.index = t.index.tz_convert("Asia/Bangkok").tz_localize(None).normalize() if t.index.tz is not None else t.index
            for day, r in t.iterrows():
                d.loc[day, ["Open", "High", "Low", "Close"]] = [r["Open"], r["High"], r["Low"], r["Close"]]
    except Exception:
        pass
    d = d.sort_index()
    d["High"] = d[["High", "Open", "Close"]].max(axis=1)
    d["Low"] = d[["Low", "Open", "Close"]].min(axis=1)
    return d


@st.cache_data(ttl=60, show_spinner=False)
def get_history(sym: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False)
    if not df.empty:
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    # Yahoo ให้แท่งรายวันของ ^SET.BK แค่วันล่าสุด -> สร้างแท่งเทียนจากแท่งรายชั่วโมง (ย้อนหลัง ~2 ปี)
    # แท่งรายเดือนของ Yahoo ก็ขาดช่วง -> เดือนที่มีแท่งรายวันใช้แท่งที่สร้างเอง เดือนก่อนหน้านั้นใช้ของ Yahoo
    if interval in ("1d", "1wk") and len(df) < 5 or interval == "1mo" and sym in NO_DAILY_HISTORY:
        d = hourly_daily(sym)
        if len(d) >= 5:
            agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
            if interval == "1wk":
                d = d.resample("W-FRI").agg(agg).dropna(subset=["Close"])
            elif interval == "1mo":
                d = d.resample("MS").agg(agg).dropna(subset=["Close"])
                d = d[d.index > d.index[0]]  # เดือนแรกอาจไม่ครบเดือน
                if len(df):
                    old = df.copy()
                    old.index = old.index.tz_convert("Asia/Bangkok").tz_localize(None) if old.index.tz is not None else old.index
                    d = pd.concat([old[old.index < d.index[0]], d])
            if period == "ytd":
                d = d[d.index >= pd.Timestamp(dt.date.today().year, 1, 1)]
            elif period in PERIOD_DAYS:
                d = d[d.index >= pd.Timestamp.today() - pd.Timedelta(days=PERIOD_DAYS[period])]
            return d
    # สำรอง: ราคาปิดอย่างเดียวจากไฟล์ในโปรเจกต์ (กราฟเป็นเส้น)
    if len(df) < 5 and interval in ("1d", "1wk", "1mo"):
        loc = local_history(sym)
        if not loc.empty:
            if period == "ytd":
                loc = loc[loc.index >= pd.Timestamp(dt.date.today().year, 1, 1)]
            elif period in PERIOD_DAYS:
                loc = loc[loc.index >= pd.Timestamp.today() - pd.Timedelta(days=PERIOD_DAYS[period])]
            return loc
    return df


@st.cache_data(ttl=3600)
def load_set100() -> pd.DataFrame:
    try:
        return pd.read_csv(SET100_CSV, comment="#", encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=["symbol", "yahoo", "set50"])


@st.cache_data(ttl=3600)
def load_us() -> pd.DataFrame:
    try:
        return pd.read_csv(US_CSV, comment="#", encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=["symbol", "yahoo", "name", "sector", "dow30", "ndx100", "sp500"])


# ชุดหุ้นสำหรับแท็บการ์ด/ภาพรวม: ชื่อ -> (ตลาด, คอลัมน์กรอง หรือ None = ทั้งไฟล์)
UNIVERSES = {
    "🇹🇭 SET100": ("TH", None),
    "🇹🇭 SET50": ("TH", "set50"),
    "🇺🇸 Dow 30": ("US", "dow30"),
    "🇺🇸 Nasdaq-100": ("US", "ndx100"),
    "🇺🇸 S&P 500": ("US", "sp500"),
    "🌍 ดัชนีทั่วโลก": ("WORLD", None),
}


def load_universe(name: str) -> pd.DataFrame:
    """คืนตาราง symbol, yahoo, name, sector, star (★ = SET50 ในชุด SET100)"""
    market, flag = UNIVERSES[name]
    if market == "TH":
        df = load_set100().copy()
        df["name"], df["sector"] = df["symbol"], None
        df["star"] = (df["set50"] == 1) & (flag is None)
    else:
        df = load_us().copy()
        df["star"] = False
    if flag:
        df = df[df[flag] == 1]
    return df[["symbol", "yahoo", "name", "sector", "star"]]


def get_bulk(yahoo_syms: tuple) -> pd.DataFrame:
    # ชุดใหญ่ (S&P 500 ใช้ ~10 วินาที) เก็บ cache นานขึ้น ไม่ให้ดึงใหม่ทุกนาที
    return (_bulk_large if len(yahoo_syms) > 200 else _bulk_small)(yahoo_syms)


@st.cache_data(ttl=60, show_spinner="กำลังดึงราคาหุ้น ...")
def _bulk_small(yahoo_syms: tuple) -> pd.DataFrame:
    return _bulk(yahoo_syms)


@st.cache_data(ttl=300, show_spinner="กำลังดึงราคาหุ้นชุดใหญ่ (~10 วินาที) ...")
def _bulk_large(yahoo_syms: tuple) -> pd.DataFrame:
    return _bulk(yahoo_syms)


def _bulk(yahoo_syms: tuple) -> pd.DataFrame:
    """ดึงราคาปิดรายวัน ~1 ปี ของหุ้นทั้งชุดในคำขอเดียว แล้วคำนวณผลตอบแทนหลายช่วง"""
    d = yf.download(list(yahoo_syms), period="1y", interval="1d", progress=False,
                    threads=True, auto_adjust=False, group_by="column")
    close, vol = d["Close"], d["Volume"]
    ytd0 = pd.Timestamp(dt.date.today().year, 1, 1)
    rows = []
    for s in yahoo_syms:
        c = close[s].dropna() if s in close else pd.Series(dtype=float)
        if len(c) < 2:
            rows.append({"yahoo": s})
            continue
        last = c.iloc[-1]
        idx = c.index.tz_localize(None) if c.index.tz is not None else c.index

        def ret(n):
            return (last / c.iloc[-1 - n] - 1) * 100 if len(c) > n else None

        ytd_base = c[idx < ytd0]
        v = vol[s].iloc[-1] if s in vol else 0
        rows.append({
            "yahoo": s, "ราคา": last, "เปลี่ยนแปลง": last - c.iloc[-2],
            "1D %": ret(1), "5D %": ret(5), "1M %": ret(21),
            "YTD %": (last / ytd_base.iloc[-1] - 1) * 100 if len(ytd_base) else None,
            "ห่างจุดสูงสุด 52W %": (last / c.max() - 1) * 100,
            "มูลค่าซื้อขาย (ล้าน)": (v or 0) * last / 1e6,
            "เหนือ MA50": bool(last > c.tail(50).mean()) if len(c) >= 50 else None,
            "เหนือ MA200": bool(last > c.tail(200).mean()) if len(c) >= 200 else None,
            "วันที่": idx[-1].date(),
        })
    return pd.DataFrame(rows)


def now_th():
    """เวลาไทย (เซิร์ฟเวอร์ Streamlit Cloud ใช้ UTC)"""
    return dt.datetime.now(ZoneInfo("Asia/Bangkok"))


def fmt(x, nd=2):
    return "-" if x is None or pd.isna(x) else f"{x:,.{nd}f}"


# ---------------- Sidebar ----------------
names = load_names()
st.sidebar.header("⚙️ ตั้งค่า")
raw = st.sidebar.text_area(
    "Watchlist (คั่นด้วย , หรือขึ้นบรรทัดใหม่)",
    value=", ".join(DEFAULT_WATCHLIST),
    help="ใช้สัญลักษณ์แบบ Yahoo เช่น AAPL, PTT.BK, ^SET.BK, GC=F, BTC-USD, THB=X",
)
watchlist = list(dict.fromkeys(s.strip().upper() for s in raw.replace("\n", ",").split(",") if s.strip()))
if names:
    with st.sidebar.expander("สัญลักษณ์ใน assets_meta.csv"):
        st.dataframe(pd.DataFrame(names.items(), columns=["symbol", "ชื่อ"]), hide_index=True, height=250)

period_label = st.sidebar.selectbox("ช่วงเวลา", list(PERIODS), index=6)
period, interval = PERIODS[period_label]
show_ma = st.sidebar.multiselect("เส้นค่าเฉลี่ย (MA)", [10, 20, 50, 100, 200], default=[10, 50, 100, 200])
refresh = st.sidebar.select_slider("รีเฟรชอัตโนมัติ (วินาที)", options=[0, 30, 60, 120, 300], value=60,
                                   help="0 = ปิด")
if st.sidebar.button("🔄 ดึงข้อมูลใหม่ตอนนี้"):
    st.cache_data.clear()

st.title("📈 Stock Dashboard (Yahoo Finance)")
apply_mobile_ui()  # จอมือถือ: แท็บเป็นปุ่มหลายแถว, ราคา 2 คอลัมน์, ปุ่มลอย ↑ เมนู
if not watchlist:
    st.info("ใส่สัญลักษณ์อย่างน้อย 1 ตัวใน Watchlist")
    st.stop()


# ---------------- Quote tiles (auto refresh) ----------------
@st.fragment(run_every=refresh if refresh else None)
def quote_board():
    st.caption(f"อัปเดตล่าสุด: {now_th():%Y-%m-%d %H:%M:%S}"
               + (f" · รีเฟรชทุก {refresh} วินาที" if refresh else ""))
    cols = st.container(key="quoteboard").columns(min(5, len(watchlist)))  # key ใช้จัดตาราง 2 ช่องบนมือถือ
    rows = []
    for i, sym in enumerate(watchlist):
        try:
            q = get_quote(sym)
        except Exception as e:
            q = {"last": None, "prev": None, "currency": None}
            st.toast(f"{sym}: ดึงข้อมูลไม่ได้ ({e})")
        chg = pct = None
        if q["last"] is not None and q["prev"]:
            chg = q["last"] - q["prev"]
            pct = chg / q["prev"] * 100
        label = names.get(sym, sym)
        cols[i % len(cols)].metric(
            label=f"{label} ({sym})" if label != sym else sym,
            value=fmt(q["last"]),
            delta=None if chg is None else f"{chg:+,.2f} ({pct:+.2f}%)",
        )
        rows.append({"Symbol": sym, "ชื่อ": label, "ราคาล่าสุด": q["last"], "ปิดก่อนหน้า": q["prev"],
                     "เปลี่ยนแปลง": chg, "%": pct, "สกุลเงิน": q["currency"]})
    with st.expander("ตารางสรุป Watchlist"):
        df = pd.DataFrame(rows).sort_values("%", ascending=False, na_position="last")
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={"%": st.column_config.NumberColumn(format="%+.2f%%"),
                                    "ราคาล่าสุด": st.column_config.NumberColumn(format="%,.2f"),
                                    "ปิดก่อนหน้า": st.column_config.NumberColumn(format="%,.2f"),
                                    "เปลี่ยนแปลง": st.column_config.NumberColumn(format="%+,.2f")})


quote_board()
st.divider()

set100 = load_set100()
us = load_us()
# ตัวเลือกกราฟ/เปรียบเทียบ = Watchlist + หุ้น SET100 + หุ้นสหรัฐฯ ในดัชนี
world = load_world()
all_syms = list(dict.fromkeys(watchlist + list(world["yahoo"]) + list(set100["yahoo"]) + list(us["yahoo"])))
names.update({r.yahoo: r.name for r in us.itertuples()})
names.update({r.yahoo: f"{r.country} · {r.index_name}" for r in world.itertuples()})

TAB_CARDS, TAB_REPORT, TAB_OVERVIEW, TAB_SIG, TAB_SECTOR, TAB_HIST, TAB_MOOD, TAB_ALLOC, TAB_FUND, TAB_CHART, TAB_COMPARE = (
    "💳 การ์ดราคาหุ้น", "📰 รายงานประจำวัน", "🗺️ ภาพรวมตลาด", "🧭 สัญญาณเทคนิค (SET100 · S&P 500)", "🔥 US Sector Heatmap",
    "⏪ ย้อนหลัง & Backtest", "🌡️ ดัชนีอารมณ์ตลาด", "⚖️ สัดส่วนหุ้น-พันธบัตร", "🔎 ข้อมูลพื้นฐาน", "📈 กราฟราคา",
    "📊 เปรียบเทียบผลตอบแทน")

# หุ้นที่เลือก (sel_sym) ใช้ร่วมกันระหว่างแท็บข้อมูลพื้นฐานกับแท็บกราฟ
ss = st.session_state
if ss.get("sel_sym") not in all_syms:
    ss["sel_sym"] = watchlist[0] if watchlist[0] in all_syms else all_syms[0]
# กดการ์ด -> จำหุ้นไว้ แล้วรันใหม่ทั้งหน้า: เลือกหุ้นนั้น + สลับไปแท็บกราฟราคา (ต้องทำก่อนสร้าง widget)
# แท็บข้อมูลพื้นฐานใช้ sel_sym ตัวเดียวกัน จึงแสดงหุ้นตัวเดียวกันเมื่อสลับไป
if "goto_sym" in ss:
    ss["sel_sym"] = ss.pop("goto_sym")
    ss["main_tabs"] = TAB_CHART
for _k in ("fund_sym", "chart_sym"):
    ss[_k] = ss["sel_sym"]


def _pick_sym(key):
    ss["sel_sym"] = ss[key]


def sym_selector(key):
    return st.selectbox("เลือกหุ้น/สินทรัพย์ (พิมพ์ค้นหาได้ เช่น PTT.BK, NVDA)", all_syms, key=key,
                        on_change=_pick_sym, args=(key,),
                        format_func=lambda s: f"{s} — {names[s]}" if s in names else s)

universe = st.radio("ชุดหุ้น (ใช้กับแท็บการ์ดและภาพรวมตลาด)", list(UNIVERSES), horizontal=True, key="universe")
market = UNIVERSES[universe][0]
unit = "ลบ." if market == "TH" else "ล้าน $"
VALUE_COL = "มูลค่าซื้อขาย (ล้าน)"

tab4, tab_rep, tab3, tab_sig, tab_sec, tab_hist, tab_mood, tab_alloc, tab_fund, tab1, tab2 = st.tabs(
    [TAB_CARDS, TAB_REPORT, TAB_OVERVIEW, TAB_SIG, TAB_SECTOR, TAB_HIST, TAB_MOOD, TAB_ALLOC, TAB_FUND, TAB_CHART, TAB_COMPARE],
                                           key="main_tabs",
                                 on_change="rerun")  # เปิดการจำแท็บ เพื่อให้สลับแท็บจากโค้ดได้

CARD_CSS = """
<style>
.st-key-cardgrid button{width:170px;min-height:96px;border-radius:10px;padding:8px 12px;
  justify-content:flex-start;text-align:left;transition:transform .08s}
.st-key-cardgrid button:hover{transform:translateY(-2px);box-shadow:0 3px 10px rgba(0,0,0,.15)}
.st-key-cardgrid button p{font-size:.95rem;line-height:1.45;text-align:left;font-variant-numeric:tabular-nums}
.st-key-cardgrid button p strong:nth-of-type(2){font-size:1.55rem}
.st-key-cardgrid button p img{height:22px !important;max-height:22px !important;width:auto;border-radius:3px;
  box-shadow:0 0 0 1px rgba(0,0,0,.15);vertical-align:-5px;margin-right:4px}
.st-key-cardgrid [class*="st-key-c_up_"] button{background:rgba(44,160,44,.12);border-color:rgba(44,160,44,.5)}
.st-key-cardgrid [class*="st-key-c_down_"] button{background:rgba(214,39,40,.12);border-color:rgba(214,39,40,.5)}
</style>
"""


def universe_data() -> pd.DataFrame:
    if market == "WORLD":
        w = get_world().dropna(subset=["ราคา"]).copy()
        w["symbol"] = w["country"]
        w["flag_img"] = w["flag"].map(flag_url)
        w["name"], w["sector"], w["star"] = w["index_name"], w["region"], False
        w[VALUE_COL] = 0.0
        return w
    base = load_universe(universe)
    if base.empty:
        return base
    return get_bulk(tuple(base["yahoo"])).merge(base, on="yahoo", how="left").dropna(subset=["ราคา"])


@st.fragment(run_every=refresh if refresh else None)
def cards_tab():
    b = universe_data()
    if b.empty:
        st.warning("ไม่มีข้อมูลชุดหุ้นนี้ (ตรวจไฟล์ files/set100.csv หรือ files/us_indices.csv)")
        return
    c1, c2 = st.columns([1, 1])
    card_sort = c1.selectbox("เรียงตาม", ["ตามตัวอักษร", "ขึ้นมากสุด", "ลงมากสุด", "มูลค่าซื้อขาย"], key="card_sort")
    q = c2.text_input("ค้นหาหุ้น", placeholder="เช่น PTT, KBANK, NVDA", key="card_q").strip().upper()
    if q:
        b = b[b["symbol"].str.upper().str.contains(q, regex=False)
              | b["name"].astype(str).str.upper().str.contains(q, regex=False)]
    col, asc = {"ตามตัวอักษร": ("symbol", True), "ขึ้นมากสุด": ("1D %", False),
                "ลงมากสุด": ("1D %", True), "มูลค่าซื้อขาย": (VALUE_COL, False)}[card_sort]
    b = b.sort_values(col, ascending=asc)
    st.caption(f"{universe} · {len(b)} หุ้น · ข้อมูลวันที่ {b['วันที่'].max() if len(b) else '-'} · "
               f"ดึงล่าสุด {now_th():%H:%M:%S} · % เทียบราคาปิดวันก่อน"
               + (" · ★ = อยู่ใน SET50" if b["star"].any() else "")
               + " · 👆 กดการ์ดเพื่อดูกราฟราคาของหุ้นตัวนั้น")
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    clicked = None
    with st.container(key="cardgrid", horizontal=True, gap="small"):
        for _, r in b.iterrows():
            p, ch = r["1D %"], r["เปลี่ยนแปลง"]
            cls = "flat" if pd.isna(p) or p == 0 else ("up" if p > 0 else "down")
            color = {"up": "green", "down": "red", "flat": "gray"}[cls]
            chg = "-" if pd.isna(p) else f"{ch:+,.2f} ({p:+.2f}%)"
            star = " :orange[★]" if r["star"] else ""
            flag = f"![]({r['flag_img']}) " if isinstance(r.get("flag_img"), str) else ""
            label = f"{flag}**{r['symbol']}**{star}  \n**{r['ราคา']:,.2f}**  \n:{color}[{chg}]"
            tip = f"{r['name']} — กดเพื่อดูกราฟ" if r["name"] != r["symbol"] else f"กดเพื่อดูกราฟ {r['symbol']}"
            if st.button(label, key=f"c_{cls}_{r['yahoo']}", help=tip):
                clicked = r["yahoo"]
    if clicked:
        st.session_state["goto_sym"] = clicked
        st.rerun()  # รันทั้งหน้า (ไม่ใช่แค่ fragment) เพื่อสลับแท็บ


with tab4:
    cards_tab()


# ---------------- Market overview ----------------
@st.fragment(run_every=refresh if refresh else None)
def overview_tab():
    df = universe_data()
    if df.empty:
        st.error("ดึงข้อมูลไม่ได้ ลองกด 'ดึงข้อมูลใหม่ตอนนี้'")
        return
    if market == "WORLD":
        render_world(df)
        return
    rcol = st.radio("วัดผลตอบแทนช่วง", ["1D %", "5D %", "1M %", "YTD %"], index=3, horizontal=True,
                    key="ov_rcol")  # ค่าเริ่มต้น: YTD

    up, down = int((df[rcol] > 0).sum()), int((df[rcol] < 0).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("หุ้นขึ้น", up)
    m2.metric("หุ้นลง", down)
    m3.metric("ไม่เปลี่ยนแปลง", len(df) - up - down)
    m4.metric(f"ค่าเฉลี่ย {rcol}", f"{df[rcol].mean():+.2f}%")
    st.caption(f"{universe} · ข้อมูลวันที่ {df['วันที่'].max()} · {len(df)} หุ้น · ราคาจาก Yahoo Finance"
               + (" (ดีเลย์ ~15 นาที)" if market == "TH" else ""))

    # Heatmap: ขนาด = มูลค่าซื้อขาย, สี = % เปลี่ยนแปลง (หุ้นสหรัฐฯ จัดกลุ่มตาม sector)
    hm = df.assign(size=df[VALUE_COL].clip(lower=0.01))
    lim = max(1.0, float(hm[rcol].abs().quantile(0.95)))
    ids, labels, parents, values, colors, custom = [], [], [], [], [], []
    has_sector = hm["sector"].notna().any()
    if has_sector:
        hm["sector"] = hm["sector"].fillna("Other")
        for sec, g in hm.groupby("sector"):
            ids.append(f"sec:{sec}"); labels.append(sec); parents.append("")
            values.append(g["size"].sum())
            colors.append((g[rcol] * g["size"]).sum() / g["size"].sum())
            custom.append([None, g["size"].sum()])
    for _, r in hm.iterrows():
        ids.append(r["yahoo"]); labels.append(f"{r['symbol']}<br>{r[rcol]:+.2f}%")
        parents.append(f"sec:{r['sector']}" if has_sector else "")
        values.append(r["size"]); colors.append(r[rcol]); custom.append([r["ราคา"], r[VALUE_COL]])
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
        marker=dict(colors=colors, colorscale=[[0, "#d62728"], [0.5, "#3a3a3a"], [1, "#2ca02c"]],
                    cmin=-lim, cmax=lim, showscale=True, colorbar=dict(title="%")),
        customdata=custom,
        hovertemplate="%{label}<br>ราคา %{customdata[0]:,.2f}<br>มูลค่า %{customdata[1]:,.0f} " + unit
                      + "<extra></extra>",
        textfont=dict(color="white"), tiling=dict(pad=1)))
    fig.update_layout(height=620 if has_sector else 520, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    g1, g2 = st.columns(2)
    for col, title, part in [(g1, "🚀 Top 10 ขึ้นแรง", df.nlargest(10, rcol)),
                             (g2, "🔻 Top 10 ลงแรง", df.nsmallest(10, rcol))]:
        f = go.Figure(go.Bar(x=part[rcol], y=part["symbol"], orientation="h",
                             marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in part[rcol]],
                             text=part[rcol].map("{:+.2f}%".format), textposition="auto"))
        f.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"))
        col.plotly_chart(f, use_container_width=True)

    st.subheader("ตารางหุ้นทั้งหมด")
    cols = ["symbol"] + (["name", "sector"] if market == "US" else ["star"]) + [
        "ราคา", "เปลี่ยนแปลง", "1D %", "5D %", "1M %", "YTD %", "ห่างจุดสูงสุด 52W %", VALUE_COL, "yahoo"]
    tbl = df[cols].sort_values(rcol, ascending=False).rename(columns={VALUE_COL: f"มูลค่าซื้อขาย ({unit})"})
    if "star" in tbl:
        tbl["star"] = tbl["star"].map({True: "✓", False: ""})
    pct = st.column_config.NumberColumn(format="%+.2f%%")
    st.caption("👆 คลิกแถวหุ้นเพื่อไปดูกราฟราคา")
    clickable_table(st, tbl, "ov_tbl", use_container_width=True, height=500,
                    column_config={"star": st.column_config.TextColumn("SET50", width="small"),
                                "name": "ชื่อบริษัท",
                                "ราคา": st.column_config.NumberColumn(format="%,.2f"),
                                "เปลี่ยนแปลง": st.column_config.NumberColumn(format="%+,.2f"),
                                "1D %": pct, "5D %": pct, "1M %": pct, "YTD %": pct, "ห่างจุดสูงสุด 52W %": pct,
                                f"มูลค่าซื้อขาย ({unit})": st.column_config.NumberColumn(format="%,.0f")})
    fname = universe.split(" ", 1)[1].replace(" ", "_").replace("&", "")
    st.download_button("⬇️ ดาวน์โหลดตาราง CSV", tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{fname}_{dt.date.today()}.csv", mime="text/csv")


with tab3:
    overview_tab()

# ---------------- สัญญาณเทคนิค (SET100 · S&P 500 ในแท็บเดียว เลือกตลาดด้านใน) ----------------
with tab_sig:
    if getattr(tab_sig, "open", True):
        render_signals_hub(set100, us)

# ---------------- US sector heatmap ----------------
with tab_sec:
    if getattr(tab_sec, "open", True):  # ดึงข้อมูลเฉพาะตอนเปิดแท็บนี้ (ครั้งแรกใช้เวลา ~30 วินาที)
        _sp = load_universe("🇺🇸 S&P 500")
        render_sector_heatmap(get_bulk(tuple(_sp["yahoo"])).merge(_sp, on="yahoo", how="left")
                              .dropna(subset=["ราคา"]))

# ---------------- History & backtest ----------------
with tab_hist:
    if getattr(tab_hist, "open", True):  # ดึงข้อมูลเฉพาะตอนเปิดแท็บนี้
        render_history_tab()

# ---------------- Market mood index ----------------
with tab_mood:
    if getattr(tab_mood, "open", True):  # ดึงข้อมูลเฉพาะตอนเปิดแท็บนี้
        render_mood_tab()

# ---------------- Stock/bond allocation model ----------------
with tab_rep:
    if getattr(tab_rep, "open", True):  # ดึงข้อมูลเฉพาะตอนเปิดแท็บนี้
        render_daily_report()

with tab_alloc:
    if getattr(tab_alloc, "open", True):  # ดึงข้อมูลเฉพาะตอนเปิดแท็บนี้
        render_allocation_tab()

# ---------------- Fundamentals ----------------
with tab_fund:
    sym = sym_selector("fund_sym")
    wrow = world[world["yahoo"] == sym]
    if len(wrow) and isinstance(wrow.iloc[0]["etf"], str):
        # ดัชนี (หรือ ETF ประเทศ) ในชุดดัชนีทั่วโลก -> ข้อมูลพื้นฐานระดับดัชนี
        wl = get_world()
        lvl = wl.loc[wl["yahoo"] == sym, "ราคา"]
        render_index_fundamentals(wrow.iloc[0], float(lvl.iloc[0]) if len(lvl) and pd.notna(lvl.iloc[0]) else None)
    elif render_header(sym, names.get(sym, "")):
        f1, f2, f3 = st.tabs(["📋 ตัวชี้วัดพื้นฐาน", "🎯 นักวิเคราะห์ & ประมาณการ", "🧾 งบการเงิน & ปันผล"])
        with f1:
            render_metrics(sym)
        with f2:
            render_analyst(sym)
        with f3:
            render_financials(sym)
    else:
        st.info("สินทรัพย์นี้ไม่ใช่หุ้นรายตัว (เช่น ดัชนี/สินค้าโภคภัณฑ์/ค่าเงิน/คริปโต) จึงไม่มีข้อมูลพื้นฐาน "
                "ดูกราฟได้ที่แท็บ 📈 กราฟราคา")

# ---------------- Price chart ----------------
with tab1:
    sym = sym_selector("chart_sym")
    render_header(sym, names.get(sym, ""), chips_on=False)
    st.markdown(f"#### 📈 กราฟราคา ({period_label})")
    df = get_history(sym, period, interval)
    if df.empty:
        st.warning(f"ไม่พบข้อมูล {sym} สำหรับช่วง {period_label}")
    else:
        close_only = bool((df["Volume"] == 0).all() and (df["Open"] == df["Close"]).all())
        daily = interval in ("1d", "1wk", "1mo")
        view = st.segmented_control("มุมมองกราฟ", ["⚡ กราฟเร็ว + สัญญาณ", "📊 TradingView เต็มรูปแบบ"],
                                    default="⚡ กราฟเร็ว + สัญญาณ", key="chart_view")
        if view == "📊 TradingView เต็มรูปแบบ":
            tv_widget(sym, interval={"1m": "1", "5m": "5", "30m": "30", "1d": "D", "1wk": "W", "1mo": "M"}.get(interval, "D"))
            st.caption(f"กราฟและข้อมูลของ TradingView ({tv_symbol(sym)}) · วาดเส้นเทรนด์/Fibonacci และเพิ่มอินดิเคเตอร์ได้จากแถบเครื่องมือ · "
                       "เส้นที่วาดจะไม่ถูกบันทึกเมื่อเปิดใหม่ · หุ้นไทยอาจเป็นราคาหน่วงเวลา")
        else:
            o1, o2, o3 = st.columns([1, 1, 3])
            if close_only:
                chart_type = "เส้น"
                o1.caption("ข้อมูลนี้มีแค่ราคาปิด จึงแสดงเป็นกราฟเส้น")
            else:
                chart_type = o1.radio("รูปแบบกราฟ", ["แท่งเทียน", "เส้น"], horizontal=True)
            ma_kind = o2.radio("เส้นค่าเฉลี่ย", ["SMA", "EMA"], horizontal=True, key="chart_ma_kind",
                               help="จำนวนวันเลือกได้ที่แถบซ้าย (เส้นค่าเฉลี่ย MA)")
            keys = []
            if interval == "1d":
                keys = o3.multiselect("ปักสัญญาณบนกราฟ (กฎชุดเดียวกับแท็บสัญญาณเทคนิค)", list(MARKERS),
                                      default=DEFAULT_MARKERS, format_func=lambda k: MARKERS[k][0], key="chart_marks")
            else:
                o3.caption("ปักสัญญาณได้เฉพาะกราฟรายวัน (เลือกช่วงเวลา 3 เดือนขึ้นไปที่แถบซ้าย)")
            events = None
            if keys:
                with st.spinner("กำลังคำนวณสัญญาณย้อนหลัง ..."):
                    events = signal_events(sym, start=df.index[0])
            xl, xm = [], []
            if interval == "1d":  # จุดขาย X7 จากวันที่ซื้อ (ค่าเริ่มต้น = สัญญาณซื้อ/A ล่าสุดบนกราฟ)
                last_buy = None
                if events is not None and len(events):
                    b = events[events["key"].isin(["buy", "A"])]
                    last_buy = b["date"].max() if len(b) else None
                xl, xm = x7_panel(df, f"chart_{sym}", last_buy)
            lw_chart(df, ma=show_ma if daily else (), ma_kind=ma_kind, events=events, keys=keys, height=580,
                     intraday=not daily, line=chart_type == "เส้น", extra_lines=xl, extra_marks=xm)
            st.caption("ซูมด้วยล้อเมาส์ · ลากเพื่อเลื่อน · ดับเบิลคลิกแกนราคาเพื่อรีเซ็ต · ตัวอักษรบนกราฟ = วันแรกที่เกิดสัญญาณ "
                       "(💪 เข้ากลุ่มแข็งแกร่ง · ซื้อ/ขาย = ผลอ่านเปลี่ยน · A/C = สูตรที่ทดสอบย้อนหลังแล้วชนะตลาดใน 20 วัน) · "
                       "ไม่ใช่คำแนะนำซื้อขาย")

        first, last = df["Close"].iloc[0], df["Close"].iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ผลตอบแทนช่วงนี้", f"{(last / first - 1) * 100:+.2f}%")
        c2.metric("สูงสุด", fmt(df["High"].max()))
        c3.metric("ต่ำสุด", fmt(df["Low"].min()))
        ret = df["Close"].pct_change().dropna()
        ann = {"1d": 252, "1wk": 52, "1mo": 12}.get(interval)
        c4.metric("ความผันผวนต่อปี", f"{ret.std() * (ann ** 0.5) * 100:.1f}%" if ann and len(ret) > 2 else "-")

        st.download_button("⬇️ ดาวน์โหลด CSV", df.to_csv().encode("utf-8-sig"),
                           file_name=f"{sym}_{period}_{interval}.csv", mime="text/csv")

# ---------------- Compare ----------------
with tab2:
    picks = st.multiselect("เลือกสินทรัพย์ที่จะเปรียบเทียบ (เริ่มต้น = 100)", all_syms, default=watchlist[:4])
    if picks:
        series = {}
        for s in picks:
            h = get_history(s, period, interval)
            if not h.empty:
                series[s] = h["Close"]
        if series:
            # ตัด timezone ให้ index ต่างตลาดต่อกันได้
            comb = pd.DataFrame({k: v.tz_localize(None) if v.index.tz is not None else v
                                 for k, v in series.items()}).sort_index().ffill()
            norm = comb.div(comb.bfill().iloc[0]) * 100
            fig2 = go.Figure()
            for s in norm.columns:
                fig2.add_trace(go.Scatter(x=norm.index, y=norm[s], name=names.get(s, s), mode="lines"))
            fig2.add_hline(y=100, line_dash="dot", line_color="gray")
            fig2.update_layout(height=520, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                               yaxis_title="ดัชนี (เริ่ม = 100)", legend=dict(orientation="h", y=1.05))
            st.plotly_chart(fig2, use_container_width=True)
            perf = (norm.iloc[-1] - 100).sort_values(ascending=False).rename("ผลตอบแทน %").to_frame()
            st.dataframe(perf.style.format("{:+.2f}%"), use_container_width=True)

st.caption("ข้อมูลจาก Yahoo Finance ผ่าน yfinance · อาจดีเลย์ตามตลาด · เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน")
