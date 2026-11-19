"""กราฟสไตล์ TradingView 2 แบบ
1) lw_chart(): TradingView Lightweight Charts (โอเพนซอร์ส) + ข้อมูล Yahoo ของเรา + ปักสัญญาณลงบนกราฟ
2) tv_widget(): ฝัง TradingView Advanced Chart ตัวจริง (เครื่องมือวาดเส้น/อินดิเคเตอร์ครบ · ข้อมูลของ TradingView)
"""
import json

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

LW_JS = "https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"
MA_COLORS = ["#f28e2b", "#2a78d6", "#7b2cbf", "#e15759", "#59a14f", "#9c755f"]

# สัญญาณที่ปักบนกราฟได้: คีย์ -> (ชื่อในตัวเลือก, ตำแหน่ง, สี, รูป, ข้อความบนกราฟ)
MARKERS = {
    "strong": ("💪 เข้ากลุ่มแข็งแกร่ง (กล่อง ✓)", "belowBar", "#1b7837", "arrowUp", "💪"),
    "buy": ("ผลอ่านเปลี่ยนเป็น \"ซื้อ\"", "belowBar", "#2e7d32", "circle", "ซื้อ"),
    "sell": ("ผลอ่านเปลี่ยนเป็น \"ขาย\"", "aboveBar", "#c62828", "circle", "ขาย"),
    "A": ("A: โดนทุบหลังทำยอด ในขาขึ้น (ทดสอบ +1.2%/20 วัน)", "belowBar", "#7b2cbf", "arrowUp", "A"),
    "C": ("C: แท่งแดงในหุ้นแข็งแกร่ง (ทดสอบ +0.75%/20 วัน)", "belowBar", "#2a78d6", "arrowUp", "C"),
    "chan": ("ทะลุช่องขาลง", "belowBar", "#e6550d", "square", "ทะลุ"),
    "vv": ("V + วอลุ่มหนุน", "belowBar", "#00897b", "arrowUp", "V"),
    "hh": ("MACD Higher High", "belowBar", "#5e35b1", "circle", "HH"),
    "purple": ("◆ กรอบม่วง", "belowBar", "#7b2cbf", "circle", "◆"),
    "bull": ("▲ แท่งเทียนขาขึ้น", "belowBar", "#26a69a", "arrowUp", ""),
    "bear": ("▼ แท่งเทียนขาลง", "aboveBar", "#ef5350", "arrowDown", ""),
}
DEFAULT_MARKERS = ["strong", "buy", "sell", "A"]


def _theme():
    try:
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = False
    return ("#0e1117", "#d1d4dc", "#1f2430") if dark else ("#ffffff", "#333333", "#f0f3fa")


@st.cache_data(ttl=3600, show_spinner=False)
def signal_events(sym: str, start=None, end=None, n_max: int = 800) -> pd.DataFrame:
    """วันที่เกิดสัญญาณแต่ละแบบ (เฉพาะวันแรกที่เกิด) ของหุ้นตัวเดียว · คำนวณด้วยกฎชุดเดียวกับแท็บสัญญาณเทคนิค"""
    import set_signals as S

    def naive(x):  # วันที่จากกราฟอาจมี timezone แต่ราคาที่ใช้คำนวณสัญญาณไม่มี
        x = pd.Timestamp(x)
        return x.tz_convert("Asia/Bangkok").tz_localize(None) if x.tz is not None else x
    start = naive(start) if start is not None else None
    end = naive(end) if end is not None else None
    px = S.load_prices((sym,), "5y")
    if sym not in px:
        return pd.DataFrame(columns=["date", "key"])
    df = px[sym]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    if len(df) < 262:
        return pd.DataFrame(columns=["date", "key"])
    a = S.indicators(df)
    i0 = 261
    if start is not None:
        i0 = max(i0, int(np.searchsorted(df.index, pd.Timestamp(start))) - 1)
    i0 = max(i0, len(df) - n_max)
    prev, out, last = {}, [], {}
    COOLDOWN = 10  # ไม่ปักซ้ำสัญญาณเดิมภายใน 10 วันทำการ (กันกราฟรกเมื่อหุ้นเข้า-ออกกลุ่มถี่)
    for i in range(i0, len(df)):
        r = S.evaluate(a, i)
        up = r["box"] in ("G", "B", "Y", "EYE")
        f = {"strong": r["box"] in ("G", "B"), "buy": r["verdict"] == "ซื้อ", "sell": r["verdict"] == "ขาย / ลดสถานะ",
             "A": r["smash"] and up, "C": bool(r["bear"]) and r["box"] in ("G", "B"),
             "chan": r["chan_days"] is not None, "vv": "V + วอลุ่มหนุน" in r["sup"], "hh": r["macd_hh"] == "strong",
             "purple": r["purple"], "bull": bool(r["bull"]), "bear": bool(r["bear"])}
        if i > i0:  # วันแรกของสัญญาณ (เมื่อวานยังไม่มี) และไม่ซ้ำกับครั้งก่อนในช่วง COOLDOWN
            for k, v in f.items():
                if v and not prev.get(k) and i - last.get(k, -10 ** 9) > COOLDOWN:
                    out.append((df.index[i], k))
                    last[k] = i
        prev = f
    return pd.DataFrame(out, columns=["date", "key"])


def lw_chart(df: pd.DataFrame, ma=(10, 50, 200), ma_kind: str = "SMA", events: pd.DataFrame = None,
             keys=None, height: int = 560, intraday: bool = False, line: bool = False, visible_bars: int = 0):
    """วาดกราฟด้วย Lightweight Charts · df ต้องมี Open/High/Low/Close/Volume และ index เป็นวันที่"""
    df = df.dropna(subset=["Close"])
    if df.empty:
        st.warning("ไม่มีข้อมูลกราฟ")
        return
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Bangkok").tz_localize(None)
    tkey = (lambda t: int(pd.Timestamp(t).timestamp())) if intraday else (lambda t: pd.Timestamp(t).strftime("%Y-%m-%d"))
    times = [tkey(t) for t in idx]
    r4 = lambda x: None if pd.isna(x) else round(float(x), 4)
    o, h, l, c, v = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close", "Volume"))
    candles = [{"time": t, "open": r4(o[i]), "high": r4(h[i]), "low": r4(l[i]), "close": r4(c[i])} for i, t in enumerate(times)]
    closes = [{"time": t, "value": r4(c[i])} for i, t in enumerate(times)]
    vols = [{"time": t, "value": float(v[i] or 0), "color": "rgba(38,166,154,.45)" if c[i] >= o[i] else "rgba(239,83,80,.45)"}
            for i, t in enumerate(times)]
    lines = []
    s = pd.Series(c, index=idx)
    for n, col in zip(ma, MA_COLORS):
        if len(s) >= n:
            m = s.ewm(span=n, adjust=False).mean() if ma_kind == "EMA" else s.rolling(n).mean()
            lines.append({"name": f"{ma_kind}{n}", "color": col,
                          "data": [{"time": t, "value": r4(x)} for t, x in zip(times, m) if pd.notna(x)]})
    marks = []
    if events is not None and len(events) and keys:
        tset = set(times)
        for d, k in zip(events["date"], events["key"]):
            if k not in keys:
                continue
            t = tkey(d)
            if t in tset:
                _, pos, col, shape, txt = MARKERS[k]
                marks.append({"time": t, "position": pos, "color": col, "shape": shape, "text": txt})
        marks.sort(key=lambda m: m["time"])
    bg, fg, grid = _theme()
    payload = json.dumps({"candles": candles, "closes": closes, "vols": vols, "lines": lines, "marks": marks,
                          "line": line, "intraday": intraday, "bg": bg, "fg": fg, "grid": grid, "h": height,
                          "vis": int(visible_bars)})
    html = f"""
<div id="lwwrap" style="position:relative;font-family:sans-serif">
  <div id="legend" style="position:absolute;left:8px;top:6px;z-index:3;font-size:12px;color:{fg};pointer-events:none"></div>
  <div id="lw"></div>
</div>
<script src="{LW_JS}"></script>
<script>
const P = {payload};
const el = document.getElementById('lw');
const chart = LightweightCharts.createChart(el, {{
  height: P.h, width: el.clientWidth || 900,
  layout: {{ background: {{ type: 'solid', color: P.bg }}, textColor: P.fg }},
  grid: {{ vertLines: {{ color: P.grid }}, horzLines: {{ color: P.grid }} }},
  rightPriceScale: {{ borderVisible: false }},
  timeScale: {{ borderVisible: false, timeVisible: P.intraday, rightOffset: 4 }},
  crosshair: {{ mode: 0 }},
}});
const main = P.line
  ? chart.addAreaSeries({{ lineColor: '#2962ff', topColor: 'rgba(41,98,255,.25)', bottomColor: 'rgba(41,98,255,0)', lineWidth: 2 }})
  : chart.addCandlestickSeries({{ upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
                                  wickUpColor: '#26a69a', wickDownColor: '#ef5350' }});
main.setData(P.line ? P.closes : P.candles);
main.priceScale().applyOptions({{ scaleMargins: {{ top: 0.08, bottom: 0.22 }} }});
const vol = chart.addHistogramSeries({{ priceFormat: {{ type: 'volume' }}, priceScaleId: '' }});
vol.priceScale().applyOptions({{ scaleMargins: {{ top: 0.82, bottom: 0 }} }});
vol.setData(P.vols);
const ls = P.lines.map(L => {{
  const s = chart.addLineSeries({{ color: L.color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
                                   crosshairMarkerVisible: false }});
  s.setData(L.data); return [L, s];
}});
if (P.marks.length) main.setMarkers(P.marks);
const nb = P.candles.length;
if (P.vis > 0 && nb > P.vis) chart.timeScale().setVisibleLogicalRange({{ from: nb - P.vis, to: nb + 3 }});
else chart.timeScale().fitContent();
const leg = document.getElementById('legend');
const f = x => x == null ? '-' : (Math.abs(x) >= 1000 ? x.toLocaleString(undefined, {{maximumFractionDigits: 2}}) : (+x).toFixed(2));
function show(p) {{
  const d = p && p.seriesData ? p.seriesData.get(main) : null;
  const last = P.line ? P.closes[P.closes.length - 1] : P.candles[P.candles.length - 1];
  const x = d || last;
  let s = P.line ? `ปิด <b>${{f(x.value)}}</b>` :
    `เปิด <b>${{f(x.open)}}</b> สูง <b>${{f(x.high)}}</b> ต่ำ <b>${{f(x.low)}}</b> ปิด <b style="color:${{x.close >= x.open ? '#26a69a' : '#ef5350'}}">${{f(x.close)}}</b>`;
  ls.forEach(([L, ser]) => {{
    const v = p && p.seriesData ? p.seriesData.get(ser) : null;
    const val = v ? v.value : (L.data.length ? L.data[L.data.length - 1].value : null);
    s += ` &nbsp;<span style="color:${{L.color}}">${{L.name}} ${{f(val)}}</span>`;
  }});
  leg.innerHTML = s;
}}
show(null);
chart.subscribeCrosshairMove(show);
new ResizeObserver(() => chart.applyOptions({{ width: el.clientWidth }})).observe(el);
</script>"""
    components.html(html, height=height + 8)


# ---------------- TradingView Advanced Chart (ฝังของจริง) ----------------
# ดัชนีสหรัฐบางตัว (SP:SPX, DJ:DJI) ใช้ในวิดเจ็ตฟรีไม่ได้เพราะลิขสิทธิ์ข้อมูล จึงใช้ตัวแทนของ FOREXCOM ที่วิดเจ็ตเปิดให้
TV_MAP = {"^GSPC": "FOREXCOM:SPXUSD", "^IXIC": "FOREXCOM:NSXUSD", "^NDX": "FOREXCOM:NSXUSD", "^DJI": "FOREXCOM:DJI",
          "^RUT": "TVC:RUT",
          "^SET.BK": "SET:SET", "^SET50.BK": "SET:SET50", "^VIX": "TVC:VIX", "GC=F": "COMEX:GC1!", "CL=F": "NYMEX:CL1!",
          "BTC-USD": "BITSTAMP:BTCUSD", "ETH-USD": "BITSTAMP:ETHUSD", "THB=X": "FX_IDC:USDTHB", "DX-Y.NYB": "TVC:DXY",
          "^N225": "TVC:NI225", "^HSI": "TVC:HSI", "^KS11": "TVC:KOSPI", "^TWII": "TWSE:TAIEX"}


def tv_symbol(sym: str) -> str:
    if sym in TV_MAP:
        return TV_MAP[sym]
    if sym.endswith(".BK"):
        return "SET:" + sym[:-3].replace("-", ".")
    if sym.endswith("=X"):
        return "FX_IDC:" + sym[:-2]
    return sym.replace("-", ".").replace("^", "")


def tv_url(sym: str) -> str:
    return "https://www.tradingview.com/chart/?symbol=" + tv_symbol(sym).replace(":", "%3A")


def tv_supported(sym: str) -> bool:
    """วิดเจ็ตฟรีของ TradingView ไม่มีข้อมูลตลาดหุ้นไทย (SET) -> ต้องเปิดบนเว็บ tradingview.com แทน"""
    return not tv_symbol(sym).startswith("SET:")


def tv_widget(sym: str, height: int = 680, interval: str = "D"):
    st.link_button(f"↗️ เปิด {tv_symbol(sym)} บน tradingview.com (แท็บใหม่ · บันทึกเส้นที่วาดได้ถ้าล็อกอิน)", tv_url(sym))
    if not tv_supported(sym):
        st.info("หุ้นไทย (SET) ไม่มีในกราฟ TradingView แบบฝังฟรี จึงแสดงไม่ได้ในหน้านี้ · "
                "กดปุ่มด้านบนเพื่อเปิดกราฟเต็มรูปแบบบน tradingview.com · หรือใช้มุมมอง ⚡ กราฟเร็ว + สัญญาณ")
        return
    bg, _, _ = _theme()
    cfg = {"autosize": False, "width": "100%", "height": height - 40,
           "symbol": tv_symbol(sym), "interval": interval, "timezone": "Asia/Bangkok",
           "theme": "dark" if bg != "#ffffff" else "light", "style": "1", "locale": "th_TH",
           "allow_symbol_change": True, "hide_side_toolbar": False, "withdateranges": True, "details": False,
           "support_host": "https://www.tradingview.com"}
    html = f"""
<div class="tradingview-widget-container" style="width:100%">
  <div class="tradingview-widget-container__widget"></div>
  <div class="tradingview-widget-copyright" style="font-size:11px">
    <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">Chart by TradingView</a></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {json.dumps(cfg)}
  </script>
</div>"""
    components.html(html, height=height + 10)
