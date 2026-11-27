"""กราฟสไตล์ TradingView 2 แบบ
1) lw_chart(): TradingView Lightweight Charts (โอเพนซอร์ส) + ข้อมูล Yahoo ของเรา + ปักสัญญาณลงบนกราฟ
2) tv_widget(): ฝัง TradingView Advanced Chart ตัวจริง (เครื่องมือวาดเส้น/อินดิเคเตอร์ครบ · ข้อมูลของ TradingView)
"""
import datetime as dt
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


# ---------------- จุดขาย X7: Stop-loss −8% + Trailing stop 3×ATR (ผล backtest ดีที่สุดด้านคุมขาดทุน) ----------------
X7_STOP, X7_ATR = 0.08, 3.0


def x7_exit(df: pd.DataFrame, entry_date, entry_price: float = None, stop: float = X7_STOP, mult: float = X7_ATR) -> dict:
    """คำนวณเส้นขาย X7 ตั้งแต่วันที่ซื้อ
    - เส้น stop = ราคาซื้อ × (1 − 8%) · ราคาต่ำสุดของวันแตะเส้นนี้ = ขายทันที (ที่ราคา stop หรือราคาเปิดถ้ากระโดดผ่าน)
    - เส้น trailing = ราคาปิดสูงสุดหลังซื้อ − 3 × ATR(14) · ราคาปิดต่ำกว่าเส้นนี้ = ขายที่ราคาเปิดวันถัดไป
    คืน dict: lines (stop, trail), สถานะ, วันที่/ราคาขาย (ถ้าเกิดแล้ว)"""
    d = df.dropna(subset=["Close"]).copy()
    idx = pd.DatetimeIndex(d.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Bangkok").tz_localize(None)
    d.index = idx
    o, h, l, c = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, abs(h - prev_c), abs(l - prev_c)]), axis=0)
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    e = int(np.searchsorted(idx, pd.Timestamp(entry_date)))
    if e >= len(d):
        return {"ok": False, "msg": "วันที่ซื้อเลยข้อมูลล่าสุดของกราฟ"}
    if e < 14:
        return {"ok": False, "msg": "วันที่ซื้อเก่ากว่าช่วงกราฟ เลือกช่วงเวลาที่แถบซ้ายให้ยาวขึ้น"}
    p = float(entry_price) if entry_price else float(o[e])
    stop_px = p * (1 - stop)
    hi, trail, out = -np.inf, [], None
    for t in range(e, len(d)):
        if l[t] <= stop_px:  # stop-loss ระหว่างวัน
            out = {"date": idx[t], "price": min(o[t], stop_px) if t > e else stop_px, "why": f"แตะ stop −{stop:.0%}"}
            trail.append((idx[t], hi - mult * atr[t] if hi > -np.inf else np.nan))
            break
        hi = max(hi, c[t])
        tl = hi - mult * atr[t]
        trail.append((idx[t], tl))
        if c[t] < tl:
            nxt = t + 1
            out = {"date": idx[nxt] if nxt < len(d) else None, "price": float(o[nxt]) if nxt < len(d) else None,
                   "why": f"ปิด {c[t]:,.2f} ต่ำกว่าเส้น trailing {tl:,.2f} (วันที่ {idx[t]:%d/%m/%Y})", "signal_day": idx[t]}
            break
    last_c = float(c[-1])
    res = {"ok": True, "entry_date": idx[e], "entry": p, "stop": stop_px, "trail": trail, "exit": out,
           "last": last_c, "last_date": idx[-1], "trail_now": trail[-1][1] if trail else np.nan,
           "atr": float(atr[-1]), "high": hi}
    return res


def x7_overlays(res: dict, intraday: bool = False) -> tuple:
    """เส้นและจุดสำหรับวาดบน lw_chart"""
    if not res.get("ok"):
        return [], []
    tkey = (lambda t: int(pd.Timestamp(t).timestamp())) if intraday else (lambda t: pd.Timestamp(t).strftime("%Y-%m-%d"))
    end = res["exit"]["date"] if res["exit"] and res["exit"].get("date") is not None else res["last_date"]
    span = [t for t, _ in res["trail"]]
    if res["exit"] and res["exit"].get("date") is not None and res["exit"]["date"] not in span:
        span.append(res["exit"]["date"])
    lines = [{"name": "Stop −8%", "color": "#d62728", "dashed": True,
              "data": [{"time": tkey(t), "value": round(res["stop"], 4)} for t in span]},
             {"name": "Trailing 3×ATR", "color": "#ff7f0e", "dashed": False,
              "data": [{"time": tkey(t), "value": round(float(v), 4)} for t, v in res["trail"] if pd.notna(v)]}]
    marks = [{"time": tkey(res["entry_date"]), "position": "belowBar", "color": "#1565c0", "shape": "arrowUp",
              "text": f"ซื้อ {res['entry']:,.2f}"}]
    if res["exit"] and res["exit"].get("date") is not None:
        marks.append({"time": tkey(res["exit"]["date"]), "position": "aboveBar", "color": "#d62728",
                      "shape": "arrowDown", "text": f"ขาย {res['exit']['price']:,.2f}"})
    return lines, marks


def x7_panel(df: pd.DataFrame, key: str, default_date=None) -> tuple:
    """ช่องกรอกราคา/วันที่ซื้อ + สถานะการขายตามกฎ X7 · คืน (เส้น, จุด) ไปวาดบนกราฟ
    โหมดหลัก: ใส่ราคาซื้อ แล้วหาวันที่ที่ราคาเคยซื้อขายผ่านราคานั้น (ต่ำสุด ≤ ราคา ≤ สูงสุด) ให้เลือก"""
    ss = st.session_state
    on = st.toggle("🛡️ แสดงจุดขาย X7 (Stop −8% + Trailing 3×ATR) จากวันที่ซื้อ", value=True, key=f"{key}_x7on",
                   help="วิธีขายที่ backtest หุ้น SET100 ปี 2015–2026 แล้วคุมขาดทุนได้ดีที่สุด "
                        "(ไม้ขาดทุนเกิน 10% เหลือไม่ถึง 1% · ขาดทุนแย่สุดราว −13%)")
    if not on:
        return [], []
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Bangkok").tz_localize(None)
    if len(idx) < 16:
        st.caption("ข้อมูลกราฟสั้นเกินไปสำหรับ X7 · เลือกช่วงเวลาที่แถบซ้ายให้ยาวขึ้น")
        return [], []
    lo, hi = idx[min(14, len(idx) - 1)].date(), idx[-1].date()
    dflt = pd.Timestamp(default_date).date() if default_date is not None else idx[max(0, len(idx) - 21)].date()
    dflt = min(max(dflt, lo), hi)
    k_d, k_p, k_pick = f"{key}_x7d", f"{key}_x7p", f"{key}_x7pick"
    L, H = df["Low"].to_numpy(float), df["High"].to_numpy(float)

    def matches(price):
        """วันที่ราคาซื้อขายผ่านราคานี้ (ต่ำสุด ≤ ราคา ≤ สูงสุด) ล่าสุดก่อน · ข้าม 14 วันแรก (ต้องใช้คำนวณ ATR)"""
        ok = (L <= price) & (H >= price)
        ok[:14] = False
        return [idx[i].date() for i in np.where(ok)[0][::-1]]

    def on_price():  # ใส่ราคาแล้ว -> ตั้งวันที่ซื้อเป็นวันล่าสุดที่ราคาผ่านราคานี้ (เปลี่ยนวันที่เองได้)
        pr = ss.get(k_p) or 0
        if pr > 0:
            hits = matches(pr)
            if hits:
                ss[k_d] = hits[0]
                ss[k_pick] = hits[0]
            else:
                C = df["Close"].to_numpy(float)
                j = int(np.nanargmin(np.abs(np.where(np.arange(len(C)) >= 14, C, np.nan) - pr)))
                ss[k_d] = idx[j].date()

    def on_pick():
        ss[k_d] = ss[k_pick]

    if not isinstance(ss.get(k_d), dt.date) or not (lo <= ss[k_d] <= hi):
        ss[k_d] = dflt
    c1, c2, c3 = st.columns([1.1, 1.1, 2.8])
    p = c1.number_input("ราคาซื้อ", min_value=0.0, value=0.0, step=0.01, format="%.2f", key=k_p, on_change=on_price,
                        help="ใส่ราคาที่ซื้อจริง ระบบจะหาวันที่ราคาหุ้นเคยซื้อขายผ่านราคานี้ แล้วตั้งวันที่ซื้อให้ (วันล่าสุด) · "
                             "เว้น 0 = ใช้ราคาเปิดของวันที่ซื้อ")
    d = c2.date_input("วันที่ซื้อ", min_value=lo, max_value=hi, key=k_d, format="DD/MM/YYYY")
    note = ""
    if p > 0:
        hits = matches(p)
        if hits:
            if ss.get(k_pick) not in hits:
                ss[k_pick] = hits[0]
            rng = {idx[i].date(): (L[i], H[i]) for i in range(len(idx))}
            st.selectbox(f"🔎 วันที่ราคาผ่าน {p:,.2f} · พบ {len(hits)} วัน (ล่าสุดก่อน) · เลือกเพื่อเปลี่ยนวันที่ซื้อ", hits,
                         key=k_pick, on_change=on_pick,
                         format_func=lambda x: f"{x:%d/%m/%Y} · ช่วงราคาวันนั้น {rng[x][0]:,.2f}–{rng[x][1]:,.2f}")
            if d not in hits:
                note = f"⚠️ วันที่ {d:%d/%m/%Y} ราคาไม่ได้ผ่าน {p:,.2f} · "
        else:
            note = f"ไม่มีวันไหนในช่วงกราฟที่ราคาผ่าน {p:,.2f} จึงตั้งวันที่เป็นวันที่ราคาปิดใกล้สุด · "
        note += (f"ค้นหาในช่วงข้อมูลกราฟ ({idx[0]:%d/%m/%Y}–{idx[-1]:%d/%m/%Y}) "
                 "ถ้าซื้อก่อนหน้านั้นให้เลือกช่วงเวลาที่แถบซ้ายให้ยาวขึ้น")
    e = int(np.searchsorted(idx, pd.Timestamp(d)))
    open_px = float(df["Open"].iloc[min(e, len(df) - 1)])
    res = x7_exit(df, d, p or None)
    if not res["ok"]:
        c3.warning(res["msg"])
        return [], []
    x = res["exit"]
    if x:
        pnl = (x["price"] / res["entry"] - 1) if x.get("price") else None
        if x.get("date") is None:
            c3.error(f"⚠️ **ขายพรุ่งนี้ที่ราคาเปิด** · {x['why']}")
        else:
            c3.warning(f"🔴 **ขายแล้ว {x['date']:%d/%m/%Y}** ที่ {x['price']:,.2f} · "
                       f"{'กำไร' if pnl >= 0 else 'ขาดทุน'} {pnl:+.1%} · {x['why']}")
    else:
        pnl = res["last"] / res["entry"] - 1
        gap = min(res["last"] / res["stop"] - 1, res["last"] / res["trail_now"] - 1 if res["trail_now"] > 0 else 9)
        c3.success(f"🟢 **ถือต่อ** · ตอนนี้ {pnl:+.1%} · ราคาห่างเส้นขายที่ใกล้สุด {gap:.1%}")
    if note:
        st.caption(note)
    m = st.columns(4)
    m[0].metric("ราคาซื้อ", f"{res['entry']:,.2f}", f"{res['entry_date']:%d/%m/%Y}", delta_color="off")
    m[1].metric("เส้น Stop −8%", f"{res['stop']:,.2f}", "ขายทันทีถ้าราคาแตะ", delta_color="off")
    m[2].metric("เส้น Trailing ล่าสุด", f"{res['trail_now']:,.2f}" if pd.notna(res["trail_now"]) else "-",
                "ปิดต่ำกว่า = ขายเปิดวันถัดไป", delta_color="off")
    eff = max(res["stop"], res["trail_now"]) if pd.notna(res["trail_now"]) else res["stop"]
    m[3].metric("เส้นขายที่ใช้จริงตอนนี้", f"{eff:,.2f}", "ค่าที่สูงกว่าของสองเส้น", delta_color="off")
    st.caption((f"ราคาซื้อเริ่มต้น = ราคาเปิดวันที่เลือก ({open_px:,.2f}) · " if not p else "")
               + f"ATR(14) ล่าสุด {res['atr']:,.2f} · "
               "เส้น trailing ขยับขึ้นตามราคาปิดสูงสุดหลังซื้อ (อาจลดลงเล็กน้อยเมื่อความผันผวนเพิ่ม ตามกฎที่ใช้ทดสอบ) · "
               "ไม่ใช่คำแนะนำซื้อขาย")
    return x7_overlays(res)


def lw_chart(df: pd.DataFrame, ma=(10, 50, 200), ma_kind: str = "SMA", events: pd.DataFrame = None,
             keys=None, height: int = 560, intraday: bool = False, line: bool = False, visible_bars: int = 0,
             extra_lines=None, extra_marks=None):
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
    if extra_marks:
        tset = set(times)
        marks += [m for m in extra_marks if m["time"] in tset]
    marks.sort(key=lambda m: m["time"])
    for L in extra_lines or []:
        lines.append({"name": L["name"], "color": L["color"], "dashed": L.get("dashed", False), "extra": True,
                      "data": L["data"]})
    bg, fg, grid = _theme()
    payload = json.dumps({"candles": candles, "closes": closes, "vols": vols, "lines": lines, "marks": marks,
                          "line": line, "intraday": intraday, "bg": bg, "fg": fg, "grid": grid, "h": height,
                          "vis": int(visible_bars)})
    html = f"""
<style>html,body{{margin:0;background:{bg}}}
#fs{{position:absolute;right:62px;top:4px;z-index:4;font-size:12px;padding:2px 8px;border-radius:6px;cursor:pointer;
     border:1px solid rgba(128,128,128,.45);background:{bg};color:{fg};opacity:.85}}
#fs:hover{{opacity:1}}</style>
<div id="lwwrap" style="position:relative;font-family:sans-serif">
  <div id="legend" style="position:absolute;left:8px;top:6px;z-index:3;font-size:12px;color:{fg};pointer-events:none"></div>
  <button id="fs" title="เปิดกราฟเต็มจอ (กด Esc เพื่อออก)">⛶ เต็มจอ</button>
  <div id="lw"></div>
</div>
<script src="{LW_JS}"></script>
<script>
const P = {payload};
const el = document.getElementById('lw');
// จอมือถือ (กว้าง ≤ 640px): กราฟเตี้ยลง และย่อกรอบตามไม่ให้เหลือที่ว่าง
const isSmall = () => {{ try {{ return window.parent.innerWidth <= 640; }} catch (e) {{ return window.innerWidth <= 640; }} }};
const hOf = () => isSmall() ? Math.max(260, Math.round(P.h * 0.65)) : P.h;
let H = hOf();
const chart = LightweightCharts.createChart(el, {{
  height: H, width: el.clientWidth || 900,
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
  const s = chart.addLineSeries({{ color: L.color, lineWidth: L.extra ? 2 : 1, lineStyle: L.dashed ? 2 : 0,
                                   priceLineVisible: false, lastValueVisible: !!L.extra, title: L.extra ? L.name : '',
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
// ---- เต็มจอ: ขยายกรอบ iframe ของกราฟ (หน้าเว็บกับกราฟอยู่ origin เดียวกัน) · Esc เพื่อออก ----
const fsBtn = document.getElementById('fs');
function layout() {{  // ตรวจความกว้างจอทุกครั้งที่ขนาดเปลี่ยน (เช่นหมุนมือถือ)
  const s = isSmall();
  H = hOf();
  if (window.frameElement && !isFull()) window.frameElement.style.height = (H + 8) + 'px';
  leg.style.fontSize = s ? '10px' : '12px';  // มือถือ: ตัวหนังสือเล็กลง และย้ายปุ่มเต็มจอไปมุมล่างซ้าย
  Object.assign(fsBtn.style, s ? {{ top: 'auto', bottom: '34px', right: 'auto', left: '6px' }}
                               : {{ top: '4px', bottom: 'auto', right: '62px', left: 'auto' }});
}}
const pdoc = (() => {{ try {{ return window.parent.document; }} catch (e) {{ return null; }} }})();
const isFull = () => !!((pdoc && pdoc.fullscreenElement === window.frameElement) || document.fullscreenElement);
function fit() {{
  const full = isFull();
  chart.applyOptions({{ width: el.clientWidth, height: full ? Math.max(300, window.innerHeight - 8) : H }});
  fsBtn.textContent = full ? '✕ ออกจากเต็มจอ' : '⛶ เต็มจอ';
}}
fsBtn.onclick = async () => {{
  try {{
    if (isFull()) {{ await (pdoc && pdoc.fullscreenElement ? pdoc.exitFullscreen() : document.exitFullscreen()); }}
    else {{
      try {{ await window.frameElement.requestFullscreen(); }}          // ขยายทั้งกรอบกราฟ
      catch (e1) {{ await document.documentElement.requestFullscreen(); }} // สำรอง: เต็มจอจากในกรอบเอง
    }}
  }} catch (e) {{ fsBtn.textContent = 'เบราว์เซอร์ไม่อนุญาตเต็มจอ'; return; }}
  setTimeout(fit, 150);
}};
layout(); fit();
window.addEventListener('resize', () => setTimeout(fit, 50));
try {{ window.parent.addEventListener('resize', () => setTimeout(() => {{ layout(); fit(); }}, 80)); }} catch (e) {{}}
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
    _, fg, _ = _theme()
    # autosize = กราฟขยายเต็มกรอบ จึงโตตามเมื่อกดเต็มจอ (html/body ต้องสูง 100% ไม่งั้นวิดเจ็ตเตี้ย)
    cfg = {"autosize": True, "symbol": tv_symbol(sym), "interval": interval, "timezone": "Asia/Bangkok",
           "theme": "dark" if bg != "#ffffff" else "light", "style": "1", "locale": "th_TH",
           "allow_symbol_change": True, "hide_side_toolbar": False, "withdateranges": True, "details": False,
           "support_host": "https://www.tradingview.com"}
    html = f"""
<style>html,body{{height:100%;margin:0;background:{bg};font-family:sans-serif}}
#bar{{height:30px;display:flex;align-items:center;justify-content:flex-end;gap:10px;font-size:11px}}
#bar a{{color:{fg};opacity:.7}}
#fs{{font-size:12px;padding:2px 8px;border-radius:6px;cursor:pointer;border:1px solid rgba(128,128,128,.45);
     background:{bg};color:{fg}}}
.tradingview-widget-container{{height:calc(100% - 32px);width:100%}}
.tradingview-widget-container__widget{{height:100%;width:100%}}</style>
<div id="bar">
  <a href="https://www.tradingview.com/" rel="noopener nofollow" target="_blank">Chart by TradingView</a>
  <button id="fs" title="เปิดกราฟเต็มจอ (กด Esc เพื่อออก)">⛶ เต็มจอ</button>
</div>
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {json.dumps(cfg)}
  </script>
</div>
<script>
const fsBtn = document.getElementById('fs');
const pdoc = (() => {{ try {{ return window.parent.document; }} catch (e) {{ return null; }} }})();
const isFull = () => !!((pdoc && pdoc.fullscreenElement === window.frameElement) || document.fullscreenElement);
fsBtn.onclick = async () => {{
  try {{
    if (isFull()) {{ await (pdoc && pdoc.fullscreenElement ? pdoc.exitFullscreen() : document.exitFullscreen()); }}
    else {{
      try {{ await window.frameElement.requestFullscreen(); }}          // ขยายทั้งกรอบกราฟ
      catch (e1) {{ await document.documentElement.requestFullscreen(); }} // สำรอง: เต็มจอจากในกรอบเอง
    }}
  }} catch (e) {{ fsBtn.textContent = 'เบราว์เซอร์ไม่อนุญาตเต็มจอ'; return; }}
}};
window.addEventListener('resize', () => {{ fsBtn.textContent = isFull() ? '✕ ออกจากเต็มจอ' : '⛶ เต็มจอ'; }});
// จอมือถือ: กรอบเตี้ยลง (วิดเจ็ตขยายตามกรอบอยู่แล้ว)
try {{ if (window.parent.innerWidth <= 640 && window.frameElement) window.frameElement.style.height = '480px'; }} catch (e) {{}}
</script>"""
    components.html(html, height=height + 10)
