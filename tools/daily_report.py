"""แท็บ "รายงานประจำวัน" – สรุปจากเครื่องมือที่มีอยู่แล้วในแดชบอร์ดไว้หน้าเดียว (ไม่มีกฎใหม่)
1. อารมณ์ตลาดสหรัฐฯ วันนี้ เทียบเมื่อวาน / สัปดาห์ก่อน / เดือนก่อน + ความแข็งแรงของหุ้นใน SET100 และ S&P 500
2. สัดส่วนที่กฎของระบบให้ถือ: สูตรหุ้น-พันธบัตร 3 ชั้น + กฎ B (MA200 + อารมณ์ตลาด) รายดัชนี
3. ดัชนีไหนยังเป็นขาขึ้น / พักตัว / ขาลง (ราคาเทียบเส้นค่าเฉลี่ย 50 และ 200 วัน)
4. หุ้นผ่านเกณฑ์วันนี้ (โมเมนตัม 12-1 Top 20 ที่ไม่อยู่ในกล่อง ✕) ทั้ง SET100 และ S&P 500
   จุดตัดขาดทุนตามกฎขาย X7 (stop −8% หรือ ราคาปิด − 3×ATR14 ที่สูงกว่า) · เป้าหมายอ้างอิง = 2 เท่าของระยะตัดขาดทุน
ทั้งหมดเป็นผลของกฎที่ backtest ไว้ ไม่ใช่คำแนะนำการลงทุนเฉพาะบุคคล

build_report() คำนวณเป็น dict (ไม่มี UI) · render_report() แสดงผลจาก dict
GitHub Actions บันทึกรายงานทุกวันทำการ -> files/daily_report/<YYYY-MM-DD>.json (tools/snapshot_report.py) ให้ย้อนดูได้
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from market_mood import mood_data, compute_mood, divergence_warning, signal_states, zone_of, COMPONENTS, STATES
from market_allocation import allocation_now
from set_signals import (load_cache, load_prices, streak_table, cache_updated_at, BOX, signal_table, buy_streaks,
                         load_my_list, newbuy_results, _nb_files, NEWBUY_MAX)

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "..", "files")
PRICES_CSV = os.path.join(FILES, "assets_prices.csv")
META_CSV = os.path.join(FILES, "assets_meta.csv")
REPORT_DIR = os.path.join(FILES, "daily_report")

INDEX_GROUPS = ["หุ้นสหรัฐฯ", "หุ้นเอเชีย", "หุ้นยุโรป", "หุ้นอเมริกาใต้"]
EXTRA = ["GC=F", "CL=F", "BTC-USD", "TLT"]  # สินทรัพย์อื่นที่น่าดูคู่กัน
US_IDX = {"^GSPC", "^IXIC", "^DJI", "^RUT"}  # ใช้กฎ B ได้ (อารมณ์ตลาดคำนวณจากตลาดสหรัฐฯ)
HOLD_IDX = ["^GSPC", "^IXIC", "^DJI", "^RUT", "^SET.BK"]
MARKETS = [("TH", "🇹🇭 SET100", "^SET.BK"), ("US", "🇺🇸 S&P 500", "^GSPC")]
NB_MARKETS = MARKETS + [("MY", "⭐ รายชื่อของฉัน", "^GSPC")]
NEAR_EXIT = 3.0  # % · ราคาห่างเส้นขาย X7 ไม่เกินเท่านี้ = ใกล้จุดขาย
X7_STOP, X7_ATR, RR = 0.08, 3.0, 2.0
TOP_N = 10
PERIODS = [("เมื่อวาน", 1), ("สัปดาห์ก่อน", 7), ("เดือนก่อน", 30)]

DISCLAIMER = ("⚠️ รายงานนี้สร้างอัตโนมัติจากกฎที่ backtest ไว้ในแดชบอร์ด ไม่ใช่คำแนะนำการลงทุนเฉพาะบุคคล "
              "ไม่ได้คำนึงถึงเป้าหมาย ระยะเวลา หรือความเสี่ยงที่รับได้ของแต่ละคน · ผลในอดีตไม่รับประกันอนาคต")


# ======================= ข้อมูล =======================
@st.cache_data(ttl=1800, show_spinner=False)
def _local_prices(mtime: float) -> tuple:
    p = pd.read_csv(PRICES_CSV, encoding="utf-8-sig")
    px = p.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    px.index = pd.to_datetime(px.index)
    meta = pd.read_csv(META_CSV, encoding="utf-8-sig")
    return px.sort_index(), meta


def local_prices():
    return _local_prices(os.path.getmtime(PRICES_CSV))


def _ago(s: pd.Series, days: int):
    """ค่า ณ วันทำการก่อนหน้า (days=1) หรือ ณ วันทำการล่าสุดที่ไม่เกิน (วันล่าสุด − days วันปฏิทิน)"""
    s = s.dropna()
    if days == 1:
        return s.iloc[-2] if len(s) > 1 else np.nan
    s2 = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
    return s2.iloc[-1] if len(s2) else np.nan


def trend_status(s: pd.Series) -> tuple:
    s = s.dropna()
    last, ma50, ma200 = s.iloc[-1], s.rolling(50).mean().iloc[-1], s.rolling(200).mean().iloc[-1]
    if pd.isna(ma200):
        return "-", ma50, ma200
    if last > ma50 and last > ma200:
        return ("🟢 ขาขึ้น ไปต่อ" if ma50 > ma200 else "🟢 ฟื้นขึ้นเหนือทุกเส้น"), ma50, ma200
    if last > ma200:
        return "🟡 พักตัวในขาขึ้น", ma50, ma200
    if last > ma50:
        return "🟠 เด้งในขาลง", ma50, ma200
    return "🔴 ขาลง", ma50, ma200


@st.cache_data(ttl=3600, show_spinner=False)
def _atr_high(syms: tuple) -> dict:
    out = {}
    for s, df in load_prices(syms, "1y").items():
        h, l, c = df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float)
        pc = np.r_[np.nan, c[:-1]]
        tr = np.nanmax(np.vstack([h - l, abs(h - pc), abs(l - pc)]), axis=0)
        atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
        out[s] = {"atr": float(atr), "hi52": float(np.nanmax(h[-252:]))}
    return out


def picks(market: str):
    """หุ้นผ่านเกณฑ์ (โมเมนตัม Top 20 + ไม่อยู่กล่อง ✕) จากผลที่ GitHub Actions คำนวณไว้ + จุดตัดขาดทุน/เป้าหมาย"""
    t, h = load_cache(market)
    if t is None:
        return None, None
    t = t[~t["stale"]]
    p = t[(t["mom_rank"] <= 20) & (t["box"] != "X")].sort_values("mom_rank").head(TOP_N).copy()
    stk, _, _ = streak_table(h)
    p["days"] = p["yahoo"].map(stk["pass"]).fillna(0).astype(int)
    ah = _atr_high(tuple(p["yahoo"])) if len(p) else {}
    rows = []
    for r in p.itertuples():
        a, price = ah.get(r.yahoo), float(r.price)
        atr, hi52 = (a["atr"], a["hi52"]) if a else (np.nan, np.nan)
        cut = price * (1 - X7_STOP) if a is None else max(price * (1 - X7_STOP), price - X7_ATR * atr)
        risk = price - cut
        rows.append({"หุ้น": r.symbol, "ใหม่": bool(r.days <= 3),
                     "กลุ่ม": r.sector_th if isinstance(r.sector_th, str) else "-",
                     "กล่อง": BOX.get(r.box, (r.box,))[0], "ราคาปิด": price, "วันนี้ %": r.chg1 * 100,
                     "ตัดขาดทุน": cut, "ตัดขาดทุน %": -risk / price * 100,
                     "เป้าหมาย": price + RR * risk, "เป้าหมาย %": RR * risk / price * 100,
                     "โมเมนตัม 12-1 %": r.mom * 100, "ติดกลุ่ม (วัน)": int(r.days), "คำแนะนำกล่อง": r.verdict,
                     "ห่างจุดสูงสุด %": (price / hi52 - 1) * 100 if hi52 == hi52 else np.nan})
    return rows, t["date"].max()


def strong_share(market: str):
    """% หุ้นแข็งแกร่ง (กล่อง G/B) รายวัน จากประวัติสมาชิกกลุ่ม"""
    _, h = load_cache(market)
    if h is None:
        return None
    n = h.groupby("date")["yahoo"].nunique()
    return h.groupby("date")["strong"].sum() / n * 100


def _market_table(mk: str):
    """ตารางสัญญาณล่าสุดของตลาด: ผลคำนวณล่วงหน้า (ถ้ามี) ไม่งั้นคำนวณสด (รายชื่อของฉันมีไม่กี่ตัว คำนวณสดได้เร็ว)"""
    t, _ = load_cache(mk)
    if t is None:
        syms = tuple(load_my_list()["yahoo"]) if mk == "MY" else ()
        if not syms:
            return None
        t = signal_table(syms, "D", mk)
    return t[~t["stale"]] if t is not None and len(t) else None


def newbuy_list(mk: str):
    """หุ้นที่ขึ้นสัญญาณ "ซื้อ" มาไม่เกิน NEWBUY_MAX วัน + ✅ + จุดตัดขาดทุน/เป้าหมาย · คืน (แถว, วันที่ราคาปิด)"""
    t = _market_table(mk)
    if t is None:
        return None, None
    day = t["date"].max()
    b = t[t["verdict"] == "ซื้อ"].sort_values("yahoo")
    if b.empty:
        return [], day
    st_ = buy_streaks(tuple((y, f"{d:%Y-%m-%d}") for y, d in zip(b["yahoo"], b["date"])), mk)
    b = b.assign(days=b["yahoo"].map(st_).fillna(0).astype(int))
    b = b[b["days"].between(1, NEWBUY_MAX)]
    nmom = t["mom_n"].max() if "mom_n" in t and t["mom_n"].notna().any() else t["mom_rank"].max()
    b = b.assign(n_sup=b["sup"].map(len), best=lambda x: (x["sup"].map(len) >= 2) & (x["mom_rank"] <= max(1, round(nmom * 0.2))))
    ah = _atr_high(tuple(b["yahoo"])) if len(b) else {}
    rows = []
    for r in b.sort_values(["best", "days", "score"], ascending=[False, True, False]).itertuples():
        a, price = ah.get(r.yahoo), float(r.price)
        cut = price * (1 - X7_STOP) if a is None else max(price * (1 - X7_STOP), price - X7_ATR * a["atr"])
        risk = price - cut
        rows.append({"หุ้น": r.symbol, "✅": bool(r.best), "วันที่ขึ้น": int(r.days), "หนุน": int(r.n_sup),
                     "ราคาปิด": price, "ตัดขาดทุน": cut, "ตัดขาดทุน %": -risk / price * 100,
                     "เป้าหมาย": price + RR * risk, "เป้าหมาย %": RR * risk / price * 100,
                     "อันดับโมเมนตัม": None if pd.isna(r.mom_rank) else int(r.mom_rank),
                     "กล่อง": BOX.get(r.box, (r.box,))[0]})
    return rows, day


def sell_alerts() -> dict:
    """จากระบบติดตามผลจริง (พอร์ตจำลองตามสัญญาณซื้อใหม่): ไม้ที่ต้องขาย / เพิ่งขาย / ใกล้จุดขาย"""
    out = {"sell": [], "sold": [], "near": [], "open": 0}
    for mk, name, _ in NB_MARKETS:
        r = newbuy_results(_nb_files(mk), mk)
        if r is None or r.empty or "ผลตอบแทน %" not in r:
            continue
        r = r[r["ผลตอบแทน %"].notna()]
        last = pd.to_datetime(r["ถึงวันที่"]).max()
        for x in r.to_dict("records"):
            base = {"ตลาด": name, "หุ้น": x["หุ้น"], "✅": x["✅"], "ซื้อวันที่": x["ซื้อวันที่"], "ราคาซื้อ": x["ราคาซื้อ"],
                    "ล่าสุด/ขาย": x["ขาย/ล่าสุด"], "ผลตอบแทน %": x["ผลตอบแทน %"]}
            stt = str(x["สถานะ"])
            if stt.startswith("🟡"):
                out["sell"].append({**base, "เหตุผล": "ปิดต่ำกว่าเส้น trailing → ขายราคาเปิดวันทำการถัดไป"})
            elif stt.startswith("🔴"):
                if pd.Timestamp(x["ถึงวันที่"]) >= last - pd.Timedelta(days=4):
                    out["sold"].append({**base, "เหตุผล": stt.replace("🔴 ขายแล้ว · ", "") + f" ({pd.Timestamp(x['ถึงวันที่']):%d/%m})"})
            else:
                out["open"] += 1
                lines = [v for v in (x.get("เส้น stop"), x.get("เส้น trailing")) if v is not None and pd.notna(v)]
                if not lines:
                    continue
                line = max(lines)
                gap = (x["ขาย/ล่าสุด"] / line - 1) * 100
                if gap <= NEAR_EXIT:
                    out["near"].append({**base, "เส้นขาย": line, "ห่างเส้นขาย %": gap})
    return out


def _prev_report(date: str):
    ds = [d for d in saved_dates() if d < date]
    return load_report(ds[0]) if ds else None


def changes_vs(prev: dict, rep: dict) -> dict:
    """สิ่งที่เปลี่ยนจากรายงานฉบับก่อน"""
    if not prev:
        return {"since": None, "items": []}
    items = []
    m0, m1 = prev.get("mood", {}), rep["mood"]
    if m0:
        z0, z1 = zone_of(m0["now"])[0], zone_of(m1["now"])[0]
        if z0 != z1:
            items.append(f"🌡️ อารมณ์ตลาด: {z0} ({m0['now']}) → {z1} ({m1['now']})")
        if bool(m0.get("warn")) != bool(m1.get("warn")):
            items.append("🚨 สัญญาณเตือนล่วงหน้า: " + ("เริ่มทำงาน" if m1["warn"] else "หยุดทำงานแล้ว"))
    a0, a1 = (prev.get("alloc") or {}).get("w"), (rep.get("alloc") or {}).get("w")
    if a0 is not None and a1 is not None and round(a0 * 100) != round(a1 * 100):
        items.append(f"⚖️ สูตรหุ้น-พันธบัตร: หุ้น {a0 * 100:.0f}% → {a1 * 100:.0f}%")
    h0 = {h["ดัชนี"]: h["ถือ %"] for h in prev.get("holds", [])}
    for h in rep.get("holds", []):
        v0 = h0.get(h["ดัชนี"])
        if v0 is not None and h["ถือ %"] is not None and round(v0) != round(h["ถือ %"]):
            items.append(f"📌 {h['ดัชนี']}: ถือ {v0:.0f}% → {h['ถือ %']:.0f}% ({h['สถานะ']})")
    i0 = {r["ดัชนี / สินทรัพย์"]: r["สถานะ"] for r in prev.get("indices", [])}
    for r in rep.get("indices", []):
        s0 = i0.get(r["ดัชนี / สินทรัพย์"])
        if s0 and s0 != r["สถานะ"]:
            items.append(f"🧭 {r['ดัชนี / สินทรัพย์']}: {s0} → {r['สถานะ']}")
    n0 = {m["key"]: {x["หุ้น"] for x in (m.get("rows") or [])} for m in prev.get("newbuy", [])}
    for m in rep.get("newbuy", []):
        if m["key"] not in n0:
            continue
        cur = {x["หุ้น"] for x in (m.get("rows") or [])}
        new, gone = sorted(cur - n0[m["key"]]), sorted(n0[m["key"]] - cur)
        if new:
            items.append(f"🟢 {m['name']} สัญญาณซื้อใหม่เพิ่ม: " + ", ".join(new))
        if gone:
            items.append(f"⚪ {m['name']} หลุดจากรายชื่อสัญญาณซื้อใหม่: " + ", ".join(gone))
    return {"since": prev.get("date"), "items": items}


def summarize(rep: dict) -> list:
    """บทสรุป 3 บรรทัด: [(ข้อความ, สี)]"""
    m = rep["mood"]
    zn, _ = zone_of(m["now"])
    idx = [r for r in rep.get("indices", []) if r["ดัชนี / สินทรัพย์"] not in ("ทองคำ", "น้ำมันดิบ WTI", "Bitcoin")
           and "TLT" not in r["ดัชนี / สินทรัพย์"]]
    up = sum(r["สถานะ"].startswith("🟢") for r in idx)
    fl = [x["filter"] for x in rep.get("markets", []) if x.get("filter")]
    ok = [f["index"] for f in fl if f["above"]]
    if fl and len(ok) == len(fl) and up >= len(idx) / 2:
        l1 = ("🟢 ตลาดเป็นขาขึ้น", "#1b7837")
    elif not ok:
        l1 = ("🔴 ตลาดอยู่ในโหมดระวัง", "#b2182b")
    else:
        l1 = ("🟡 ตลาดผสม ไม่ไปทางเดียวกัน", "#d4a017")
    line1 = f"{l1[0]} · อารมณ์ตลาดสหรัฐฯ {zn} ({m['now']}) · ดัชนีหุ้นที่ยังเป็นขาขึ้น {up} จาก {len(idx)}"
    a = rep.get("alloc") or {}
    hold = {h["ดัชนี"]: h["ถือ %"] for h in rep.get("holds", [])}
    parts = ([f"สูตรหุ้น-พันธบัตร: หุ้น {a['w'] * 100:.0f}%"] if "w" in a else []) + \
            [f"{k} {v:.0f}%" for k, v in hold.items() if k in ("S&P 500", "SET Index") and v is not None]
    avg = np.mean([v for v in hold.values() if v is not None]) if hold else 50
    line2 = ("⚖️ กฎให้ถือหุ้น" + ("มาก" if avg >= 75 else "น้อย" if avg <= 25 else "ปานกลาง") + " · " + " · ".join(parts))
    best = [f"{x['หุ้น']}" for nb in rep.get("newbuy", []) for x in (nb.get("rows") or []) if x["✅"]]
    sells = rep.get("sells") or {}
    n_sell = len(sells.get("sell", [])) + len(sells.get("near", []))
    if not ok:
        line3 = "👀 วันนี้เฝ้าดู ยังไม่ใช่จังหวะซื้อตามกฎ (ดัชนีอยู่ใต้เส้น 200 วัน)"
    else:
        line3 = "🟢 ตลาดผ่านตัวกรอง: " + ", ".join(ok)
    if "newbuy" in rep:  # รายงานฉบับเก่าก่อน 21 พ.ย. 2026 ไม่มีส่วนนี้
        line3 += " · สัญญาณซื้อใหม่ ✅ " + (f"{len(best)} ตัว: " + ", ".join(best[:8]) if best else "ไม่มีวันนี้")
    if n_sell:
        line3 += f" · 🔴 ไม้ที่ต้องขาย/ใกล้จุดขาย {n_sell} ไม้"
    out = [(line1, l1[1]), (line2, "#2a78d6"), (line3, "#555")]
    # บรรทัดรายชื่อ: สัญญาณซื้อใหม่ของแต่ละตลาด (✅ ก่อน แล้วตามจำนวนหนุน) ตลาดละไม่เกิน 6 ตัว
    parts = []
    for nb in rep.get("newbuy", []):
        rows = nb.get("rows") or []
        if not rows:
            continue
        top = sorted(rows, key=lambda x: (not x["✅"], -x["หนุน"], x["วันที่ขึ้น"]))[:6]
        names = ", ".join(("✅" if x["✅"] else "") + x["หุ้น"] for x in top)
        more = f" +{len(rows) - len(top)}" if len(rows) > len(top) else ""
        parts.append(f"{nb['name'].split(' ', 1)[1]} {len(rows)} ตัว: {names}{more}")
    if parts:
        out.append(("🟢 สัญญาณซื้อใหม่ · " + " · ".join(parts), "#1b7837"))
    return out


# ======================= คำนวณรายงาน (ไม่มี UI) =======================
def build_report() -> dict:
    now_th = pd.Timestamp.now(tz="Asia/Bangkok")
    rep = {"date": f"{now_th:%Y-%m-%d}", "generated": now_th.isoformat(timespec="minutes")}
    upd = cache_updated_at()
    rep["lists_updated"] = upd.isoformat(timespec="minutes") if upd is not None else None

    # 1. อารมณ์ตลาด
    d = mood_data()
    sc, _ = compute_mood(d)
    warn = divergence_warning(d)
    s = sc["score"]
    now = round(float(s.iloc[-1]))
    refs = []
    for lab, n in PERIODS:
        v = _ago(s, n)
        refs.append({"label": lab, "value": None if pd.isna(v) else round(float(v))})
    drivers = []
    wk = sc[sc.index <= sc.index[-1] - pd.Timedelta(days=7)]
    if len(wk):
        ch = pd.Series({COMPONENTS[k][0]: sc[k].iloc[-1] - wk[k].iloc[-1] for k in COMPONENTS}).dropna().sort_values()
        drivers = [f"{k} {v:+.0f}" for k, v in list(ch.items())[:2] + list(ch.items())[-2:] if abs(v) >= 5]
    wl = warn.iloc[-1]
    rep["mood"] = {"asof": f"{s.index[-1]:%Y-%m-%d}", "now": now, "refs": refs, "drivers": drivers,
                   "warn": bool(wl["warn"]), "breadth200": float(wl["breadth200"]), "off_high": float(wl["off_high"])}
    rep["breadth"] = []
    for mk, name, _ in MARKETS:
        sg = strong_share(mk)
        if sg is None:
            continue
        rep["breadth"].append({"name": name.split(" ", 1)[1], "now": float(sg.iloc[-1]),
                               "refs": [{"label": lab, "value": None if pd.isna(v) else float(v)}
                                        for lab, n in PERIODS for v in [_ago(sg, n)]]})

    # 2. สัดส่วน
    px, meta = local_prices()
    names = dict(zip(meta["symbol"], meta["name"]))
    try:
        al = allocation_now()
        why = ["Valuation " + ("ถูก" if al["z"] > 1 else "แพง" if al["z"] < -1 else "ปกติ")
               + f" (z {al['z']:+.2f}) → ตั้งต้นหุ้น {al['strategic'] * 100:.0f}%",
               "S&P 500 หลุดค่าเฉลี่ย 10 เดือน → ลดหุ้น" if al["down"] else "แนวโน้มยังเป็นขาขึ้น → ไม่ลด"]
        if al["down"] and al["fear"]:
            why.append("กลัวสุดขีดและเริ่มคลาย → เพิ่มกลับบางส่วน")
        rep["alloc"] = {"w": float(al["w"]), "why": why}
    except Exception as e:
        rep["alloc"] = {"error": type(e).__name__}
    holds = []
    for sym in HOLD_IDX:
        p = px[sym].dropna() if sym in px else pd.Series(dtype=float)
        if len(p) < 210:
            continue
        if sym in US_IDX:
            stt = signal_states(s, p, "B").iloc[-1]
            label, pos = (STATES[stt["state"]][0] if stt["state"] in STATES else "-"), stt["pos"]
        else:
            up = p.iloc[-1] > p.rolling(200).mean().iloc[-1]
            label, pos = ("🔵 ถือ (เหนือเส้น 200 วัน)" if up else "🔴 ถือเงินสด (ใต้เส้น 200 วัน)"), (1.0 if up else 0.0)
        holds.append({"ดัชนี": names.get(sym, sym), "กฎ": "B (MA200 + อารมณ์)" if sym in US_IDX else "MA200",
                      "สถานะ": label, "ถือ %": float(pos) * 100 if pd.notna(pos) else None})
    rep["holds"] = holds

    # 3. ดัชนี
    idx_syms = [x for x in meta.loc[meta["group"].isin(INDEX_GROUPS), "symbol"] if x in px] + [x for x in EXTRA if x in px]
    rows = []
    for sym in idx_syms:
        p = px[sym].dropna()
        if len(p) < 30:
            continue
        stt, ma50, ma200 = trend_status(p)
        last = float(p.iloc[-1])
        r = {"ดัชนี / สินทรัพย์": names.get(sym, sym), "ล่าสุด": last, "ณ": f"{p.index[-1]:%d/%m}"}
        for lab, n in [("1 วัน %", 1), ("1 สัปดาห์ %", 7), ("1 เดือน %", 30)]:
            ref = _ago(p, n)
            r[lab] = (last / ref - 1) * 100 if pd.notna(ref) else None
        r["เทียบเส้น 50 วัน %"] = (last / ma50 - 1) * 100
        r["เทียบเส้น 200 วัน %"] = (last / ma200 - 1) * 100 if pd.notna(ma200) else None
        r["สถานะ"] = stt
        rows.append(r)
    rep["indices"] = rows

    # 4. หุ้น
    rep["markets"] = []
    for mk, name, idx in MARKETS:
        pk, day = picks(mk)
        m = {"key": mk, "name": name, "day": f"{day:%Y-%m-%d}" if day is not None else None, "picks": pk, "filter": None}
        ip = px[idx].dropna() if idx in px else pd.Series(dtype=float)
        if len(ip) >= 200:
            ma = ip.rolling(200).mean().iloc[-1]
            m["filter"] = {"index": names.get(idx, idx), "above": bool(ip.iloc[-1] > ma), "gap": float((ip.iloc[-1] / ma - 1) * 100)}
        rep["markets"].append(m)

    # สัญญาณซื้อใหม่ ทั้ง 3 รายชื่อ
    rep["newbuy"] = []
    for mk, name, idx in NB_MARKETS:
        rows, day = newbuy_list(mk)
        f = next((x["filter"] for x in rep["markets"] if x["key"] == mk), None)
        if f is None:
            f = next((x["filter"] for x in rep["markets"] if x["key"] == "US"), None)
        rep["newbuy"].append({"key": mk, "name": name, "day": f"{day:%Y-%m-%d}" if day is not None else None,
                              "rows": rows, "filter": f})
    try:
        rep["sells"] = sell_alerts()
    except Exception as e:
        rep["sells"] = {"error": type(e).__name__}
    rep["changes"] = changes_vs(_prev_report(rep["date"]), rep)
    rep["summary"] = summarize(rep)
    rep["text"] = report_text(rep)
    return rep


def report_text(rep: dict) -> str:
    """สรุปแบบข้อความ (คัดลอกส่ง LINE ได้)"""
    d = pd.Timestamp(rep["date"])
    L = [f"📰 รายงานประจำวัน {d:%d/%m/%Y}", ""]
    L += [t for t, _ in (rep.get("summary") or summarize(rep))]
    sl = rep.get("sells") or {}
    if sl.get("sell") or sl.get("near"):
        L.append("\n🔴 ถึงจุดขาย / ใกล้จุดขาย (พอร์ตจำลองตามสัญญาณ):")
        for x in sl.get("sell", []):
            L.append(f"• {x['หุ้น']} ({x['ตลาด']}) ขายราคาเปิดวันถัดไป · ผล {x['ผลตอบแทน %']:+.1f}%")
        for x in sl.get("near", []):
            L.append(f"• {x['หุ้น']} ({x['ตลาด']}) ห่างเส้นขาย {x['ห่างเส้นขาย %']:.1f}% (เส้น {x['เส้นขาย']:,.2f})")
    for nb in rep.get("newbuy", []):
        if not nb.get("rows"):
            continue
        f = nb.get("filter")
        L.append(f"\n🟢 สัญญาณซื้อใหม่ {nb['name']}" + (" (ตลาดใต้เส้น 200 วัน → เฝ้าดู)" if f and not f["above"] else "") + ":")
        rows = [x for x in nb["rows"] if x["✅"]] + [x for x in nb["rows"] if not x["✅"]][:5]
        for x in rows:
            L.append(f"• {'✅ ' if x['✅'] else ''}{x['หุ้น']} · {x['วันที่ขึ้น']} วัน · หนุน {x['หนุน']} · ปิด {x['ราคาปิด']:,.2f} · "
                     f"คัทลอส {x['ตัดขาดทุน']:,.2f} ({x['ตัดขาดทุน %']:+.1f}%) · เป้า {x['เป้าหมาย']:,.2f} ({x['เป้าหมาย %']:+.1f}%)")
        if len(nb["rows"]) > len(rows):
            L.append(f"• และอีก {len(nb['rows']) - len(rows)} ตัว: " + ", ".join(x["หุ้น"] for x in nb["rows"] if x not in rows))
    ch = rep.get("changes") or {}
    if ch.get("items"):
        L.append(f"\n🆕 เปลี่ยนจากรายงาน {pd.Timestamp(ch['since']):%d/%m}:")
        L += [f"• {x}" for x in ch["items"][:12]]
    m = rep["mood"]
    zn, _ = zone_of(m["now"])
    L.append(f"\n🌡️ อารมณ์ตลาดสหรัฐฯ: {m['now']} {zn} · " + " · ".join(
        f"{r['label']} {r['value']} ({m['now'] - r['value']:+d})" if r["value"] is not None else f"{r['label']} -"
        for r in m["refs"]))
    if m["warn"]:
        L.append("🚨 สัญญาณเตือน: ข้างในตลาดเริ่มผุ ขณะดัชนียังใกล้จุดสูงสุด")
    for b in rep["breadth"]:
        L.append(f"💪 {b['name']} หุ้นแข็งแกร่ง {b['now']:.0f}% (" +
                 ", ".join(f"{r['label']} {r['value']:.0f}%" for r in b["refs"] if r["value"] is not None) + ")")
    a = rep.get("alloc") or {}
    if "w" in a:
        L.append(f"\n⚖️ สูตรหุ้น-พันธบัตร: หุ้น {a['w'] * 100:.0f}% / พันธบัตร {100 - a['w'] * 100:.0f}%")
    if rep["holds"]:
        L.append("📌 ถือตามกฎ: " + " · ".join(f"{h['ดัชนี']} {h['ถือ %']:.0f}%" for h in rep["holds"] if h["ถือ %"] is not None))
    up = [r["ดัชนี / สินทรัพย์"] for r in rep["indices"] if r["สถานะ"].startswith("🟢")]
    dn = [r["ดัชนี / สินทรัพย์"] for r in rep["indices"] if r["สถานะ"].startswith("🔴")]
    L.append("\n🟢 ยังเป็นขาขึ้น: " + (", ".join(up) or "-"))
    L.append("🔴 ขาลง: " + (", ".join(dn) or "-"))
    L.append("\n" + DISCLAIMER)
    return "\n".join(L)


# ======================= บันทึก / อ่านย้อนหลัง =======================
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def save_report(rep: dict) -> str:
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"{rep['date']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(rep), f, ensure_ascii=False, indent=1)
    return path


def saved_dates() -> list:
    return sorted((os.path.basename(p)[:-5] for p in glob.glob(os.path.join(REPORT_DIR, "*.json"))), reverse=True)


def load_report(date: str) -> dict:
    with open(os.path.join(REPORT_DIR, f"{date}.json"), encoding="utf-8") as f:
        return json.load(f)


# ======================= UI =======================
def _card(title, value, sub, color):
    return (f'<div style="border-radius:12px;padding:10px 14px;background:{color}1f;border:1px solid {color};'
            f'height:100%;margin-bottom:6px"><div style="font-size:.82rem;opacity:.75">{title}</div>'
            f'<div style="font-size:1.35rem;font-weight:700;color:{color}">{value}</div>'
            f'<div style="font-size:.82rem">{sub}</div></div>')


def _pct_style(df):
    cols = [c for c in df.columns if c.endswith("%") and c != "ถือ %"]
    return df.style.map(lambda v: "color:#1b7837" if isinstance(v, float) and v > 0 else
                        "color:#b2182b" if isinstance(v, float) and v < 0 else "", subset=cols)


LIVE = "วันนี้ (คำนวณจากข้อมูลล่าสุด)"


def render_daily_report():
    dates = saved_dates()
    c1, c2 = st.columns([1.3, 2])
    pick = c1.selectbox("ดูรายงานวันที่", [LIVE] + dates, key="rep_date",
                        format_func=lambda x: x if x == LIVE else f"{pd.Timestamp(x):%d/%m/%Y} (บันทึกไว้)",
                        help="GitHub Actions บันทึกรายงานทุกวันทำการราว 10:30–11:30 น. เลือกวันที่ย้อนหลังเพื่อดูรายงานตามที่ออกในวันนั้น")
    c2.caption(f"รายงานที่บันทึกไว้ {len(dates)} วัน" + (f" · ตั้งแต่ {pd.Timestamp(dates[-1]):%d/%m/%Y}" if dates else "")
               + " · ไฟล์อยู่ใน files/daily_report/")
    if pick == LIVE:
        with st.spinner("กำลังรวบรวมรายงาน (ครั้งแรกอาจใช้เวลาหลายวินาที) ..."):
            rep = build_report()
    else:
        rep = load_report(pick)
    render_report(rep, live=pick == LIVE)


def render_report(rep: dict, live: bool = True):
    d = pd.Timestamp(rep["date"])
    st.markdown(f"### 📰 รายงานประจำวัน · {d:%d/%m/%Y}" + ("" if live else " (บันทึกไว้)"))
    upd = rep.get("lists_updated")
    st.caption("รวมผลจากแท็บอารมณ์ตลาด · สัดส่วนหุ้น-พันธบัตร · สัญญาณเทคนิค ไว้หน้าเดียว"
               + (f" · รายชื่อหุ้นอัปเดต {pd.Timestamp(upd):%d/%m/%Y %H:%M} น." if upd else "")
               + ("" if live else f" · สร้างรายงานเมื่อ {pd.Timestamp(rep['generated']):%d/%m/%Y %H:%M} น."))
    # ---------- บทสรุป ----------
    sm = rep.get("summary") or summarize(rep)
    st.markdown("<div style='border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:10px 14px;margin:4px 0 10px'>"
                + "".join(f"<div style='margin:3px 0;font-size:1.0rem;{'font-weight:700;' if i == 0 else ''}color:{c}'>{t}</div>"
                          for i, (t, c) in enumerate(sm)) + "</div>", unsafe_allow_html=True)
    st.caption(DISCLAIMER)

    # ---------- 1. ถึงจุดขาย ----------
    st.markdown("#### 🔴 1. ถึงจุดขาย / ใกล้จุดขาย")
    with st.expander("ⓘ ส่วนนี้คืออะไร"):
        st.markdown("ระบบติดตาม **พอร์ตจำลอง**: ทุกครั้งที่หุ้นขึ้นสัญญาณซื้อใหม่ ระบบสมมติว่าซื้อที่ราคาเปิดวันถัดไป "
                    "แล้วขายตามกฎ X7 (stop −8% หรือราคาปิดหลุดเส้น trailing = ราคาปิดสูงสุดหลังซื้อ − 3×ATR) · "
                    f"**ใกล้จุดขาย** = ราคาล่าสุดห่างเส้นขายไม่เกิน {NEAR_EXIT:.0f}% · "
                    "ถ้าคุณซื้อตามสัญญาณจริง ใช้ส่วนนี้เตือนจุดขายได้ ถ้าซื้อคนละวัน ดูเส้นของตัวเองได้ที่หน้ากราฟ (🛡️ X7)")
    sl = rep.get("sells")
    if sl is None:
        st.caption("— รายงานฉบับนี้ยังไม่มีส่วนนี้ —")
    elif "error" in sl:
        st.caption(f"คำนวณไม่ได้ ({sl['error']})")
    else:
        pcfg = {"ผลตอบแทน %": st.column_config.NumberColumn(format="%+.1f%%"),
                "ห่างเส้นขาย %": st.column_config.NumberColumn(format="%+.1f%%"),
                **{k: st.column_config.NumberColumn(format="%,.2f") for k in ["ราคาซื้อ", "ล่าสุด/ขาย", "เส้นขาย"]}}
        any_ = False
        for key, ttl in [("sell", "🟡 ต้องขายที่ราคาเปิดวันทำการถัดไป"), ("near", "⚠️ ใกล้จุดขาย"),
                         ("sold", "🔴 เพิ่งขายไป (ไม่เกิน 4 วัน)")]:
            if sl.get(key):
                any_ = True
                st.markdown(f"**{ttl}** · {len(sl[key])} ไม้")
                st.dataframe(_pct_style(pd.DataFrame(sl[key])), hide_index=True, use_container_width=True, column_config=pcfg)
        if not any_:
            st.success(f"✅ ไม่มีไม้ที่ถึงหรือใกล้จุดขาย · ถืออยู่ {sl.get('open', 0)} ไม้ (พอร์ตจำลอง)")
        else:
            st.caption(f"ไม้ที่ยังถืออยู่ทั้งหมด {sl.get('open', 0)} ไม้ · รายละเอียดทุกไม้: แท็บสัญญาณเทคนิค → 📒 ผลจริงจากบันทึก")

    # ---------- 2. สัญญาณซื้อใหม่ ----------
    st.markdown(f"#### 🟢 2. สัญญาณซื้อใหม่ (ขึ้นมาไม่เกิน {NEWBUY_MAX} วัน)")
    with st.expander("ⓘ อ่านตารางนี้ยังไง"):
        st.markdown("- **สัญญาณซื้อ** = ราคาเหนือเส้น EMA 10/50/200 ทุกเส้น + คะแนนกฎ ≥ 4 + ไม่มีสัญญาณเตือน (จุดเขียวบนกราฟ)\n"
                    "- **✅** = มีสัญญาณหนุน ≥ 2 ข้อ และโมเมนตัม 12 เดือนอยู่ 20% บนสุด · backtest SET100 ปี 2015–2026 "
                    "ชนะตลาดเฉลี่ย +2.7% ต่อไม้ (ชนะราว 1 ใน 3 ไม้ แต่ไม้ชนะได้มากกว่าไม้แพ้)\n"
                    "- **ตัดขาดทุน** = ราคาปิด −8% หรือ ราคาปิด − 3×ATR (ค่าที่สูงกว่า) · **เป้าหมาย** = 2 เท่าของระยะตัดขาดทุน "
                    "(อ้างอิง · การขายจริงใช้เส้น trailing ของ X7)\n"
                    "- S&P 500: สัญญาณซื้อแบบไม่มี ✅ ยังแพ้ตลาดใน backtest · รายชื่อของฉันมี ETF ทวีคูณ/กลับทิศ ใช้กฎได้จำกัด")
    nbs = rep.get("newbuy")
    if nbs is None:
        st.caption("— รายงานฉบับนี้ยังไม่มีส่วนนี้ —")
    else:
        ncfg = {k: st.column_config.NumberColumn(format="%,.2f") for k in ["ราคาปิด", "ตัดขาดทุน", "เป้าหมาย"]}
        ncfg.update({k: st.column_config.NumberColumn(format="%+.1f%%") for k in ["ตัดขาดทุน %", "เป้าหมาย %"]})
        for nb in nbs:
            f = nb.get("filter")
            watch = f is not None and not f["above"]
            st.markdown(f"**{nb['name']}**" + (f" · ราคาปิด {pd.Timestamp(nb['day']):%d/%m/%Y}" if nb.get("day") else "")
                        + (" · <span style='color:#b2182b'>รายชื่อเฝ้าดู (ตลาดใต้เส้น 200 วัน)</span>" if watch else ""),
                        unsafe_allow_html=True)
            if nb.get("rows") is None:
                st.caption("ยังไม่มีผลคำนวณ")
            elif not nb["rows"]:
                st.caption("ไม่มีสัญญาณซื้อใหม่")
            else:
                st.dataframe(pd.DataFrame(nb["rows"]), hide_index=True, use_container_width=True, column_config=ncfg,
                             height=38 + 35 * len(nb["rows"]))

    # ---------- 3. อะไรเปลี่ยน ----------
    ch = rep.get("changes")
    st.markdown("#### 🆕 3. อะไรเปลี่ยนจากรายงานฉบับก่อน" +
                (f" ({pd.Timestamp(ch['since']):%d/%m/%Y})" if ch and ch.get("since") else ""))
    if not ch or not ch.get("since"):
        st.caption("ยังไม่มีรายงานฉบับก่อนให้เทียบ")
    elif not ch["items"]:
        st.caption("ไม่มีอะไรเปลี่ยนสถานะ")
    else:
        st.markdown("\n".join(f"- {x}" for x in ch["items"]))

    # ---------- 4 ----------
    st.markdown("#### 🌡️ 4. อารมณ์ตลาดวันนี้ เทียบกับเมื่อวาน / สัปดาห์ก่อน / เดือนก่อน")
    with st.expander("ⓘ คะแนนอารมณ์ตลาดคืออะไร"):
        st.markdown("คะแนน 0–100 จากตัวชี้วัดตลาดสหรัฐฯ 7 ตัว (VIX, แนวโน้ม S&P 500, จำนวนกลุ่มอุตสาหกรรมที่เป็นขาขึ้น, "
                    "หุ้นเทียบพันธบัตร, หุ้นกู้ขยะ, กลุ่มเสี่ยงเทียบกลุ่มป้องกัน) · ต่ำ = นักลงทุนกลัว · สูง = นักลงทุนโลภ · "
                    "**% หุ้นแข็งแกร่ง** = สัดส่วนหุ้นที่ราคาอยู่เหนือเส้นค่าเฉลี่ย 10/50/200 วันทุกเส้น")
    m = rep["mood"]
    now = m["now"]
    zn, zc = zone_of(now)
    c = st.columns(4)
    c[0].markdown(_card(f"อารมณ์ตลาดสหรัฐฯ ณ {pd.Timestamp(m['asof']):%d %b}", f"{now} · {zn}",
                        "0 = กลัวสุดขีด · 100 = โลภสุดขีด", zc), unsafe_allow_html=True)
    for col, r in zip(c[1:], m["refs"]):
        ref = r["value"]
        if ref is None:
            col.markdown(_card(r["label"], "-", "", "#999"), unsafe_allow_html=True)
            continue
        rz, rc = zone_of(ref)
        arrow = "▲ โลภขึ้น" if now > ref else "▼ กลัวขึ้น" if now < ref else "≈ ทรงตัว"
        col.markdown(_card(r["label"], f"{ref} · {rz}", f"วันนี้ {now - ref:+d} จุด {arrow}", rc), unsafe_allow_html=True)
    if m["drivers"]:
        st.caption("ตัวที่ขยับมากสุดใน 1 สัปดาห์ (คะแนน 0–100): " + " · ".join(m["drivers"]))
    if m["warn"]:
        st.error(f"🚨 สัญญาณเตือนล่วงหน้าทำงาน: หุ้นกู้ขยะอ่อนผิดปกติ + กลุ่มอุตสาหกรรมเหนือเส้น 200 วันแค่ {m['breadth200']:.0f}% "
                 f"ขณะ S&P 500 ห่างจุดสูงสุดเพียง {m['off_high']:.1f}%")
    else:
        st.caption(f"🚨 สัญญาณเตือนล่วงหน้า (เครดิต + breadth): ไม่ทำงาน · กลุ่มเหนือเส้น 200 วัน {m['breadth200']:.0f}% · "
                   f"S&P 500 ห่างจุดสูงสุด {m['off_high']:.1f}%")
    bc = st.columns(2)
    for col, b in zip(bc, rep["breadth"]):
        col.markdown(_card(f"{b['name']} · % หุ้นแข็งแกร่ง (กล่องเขียว/น้ำเงิน)", f"{b['now']:.0f}%",
                           " · ".join(f"{r['label']} {r['value']:.0f}%" for r in b["refs"] if r["value"] is not None),
                           "#1b7837" if b["now"] >= 40 else "#b2182b" if b["now"] < 20 else "#888"), unsafe_allow_html=True)

    # ---------- 2 ----------
    st.markdown("#### ⚖️ 5. สัดส่วนที่กฎของระบบให้ถือ")
    with st.expander("ⓘ อ่านส่วนนี้ยังไง"):
        st.markdown("- **สูตรหุ้น-พันธบัตร 3 ชั้น**: ชั้น 1 ดูว่าหุ้นแพงหรือถูกเทียบพันธบัตร (Earnings Yield − ดอกเบี้ย 10 ปี, z = ห่างจากค่าเฉลี่ย 10 ปีกี่เท่าของความผันผวน) "
                    "· ชั้น 2 ถ้า S&P 500 ต่ำกว่าค่าเฉลี่ย 10 เดือน ลดหุ้นครึ่งหนึ่ง · ชั้น 3 ถ้ากลัวสุดขีดแล้วเริ่มคลาย เพิ่มกลับบางส่วน\n"
                    "- **กฎ B** (รายดัชนี): ราคาเหนือเส้น 200 วัน = ถือ (ลดครึ่งถ้าโลภสุดขีด) · ใต้เส้น = ถือเงินสด "
                    "(ยกเว้นกลัวสุดขีดและเริ่มคลาย = ซื้อครึ่งเดียว)")
    a1, a2 = st.columns([1, 1.4])
    a = rep.get("alloc") or {}
    if "w" in a:
        a1.markdown(_card("สูตรหุ้น-พันธบัตร 3 ชั้น (พอร์ตสหรัฐฯ)", f"หุ้น {a['w'] * 100:.0f}% · พันธบัตร {100 - a['w'] * 100:.0f}%",
                          " · ".join(a["why"]), "#2a78d6"), unsafe_allow_html=True)
    else:
        a1.warning(f"คำนวณสูตรหุ้น-พันธบัตรไม่ได้ ({a.get('error', '-')})")
    if rep["holds"]:
        a2.dataframe(pd.DataFrame(rep["holds"]), hide_index=True, use_container_width=True,
                     column_config={"ถือ %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
    a2.caption("ถือ % = สัดส่วนของเงินส่วนที่ลงในดัชนีนั้น ตามกฎ B (backtest 2008–2026 ในแท็บอารมณ์ตลาด) · "
               "ดัชนีนอกสหรัฐฯ ใช้เส้น 200 วันอย่างเดียว")

    # ---------- 3 ----------
    st.markdown("#### 🧭 6. ดัชนีไหนยังไปต่อ อันไหนลง")
    tr = pd.DataFrame(rep["indices"])
    if len(tr):
        pct = {c_: st.column_config.NumberColumn(format="%+.1f%%") for c_ in tr.columns if c_.endswith("%")}
        st.dataframe(_pct_style(tr), hide_index=True, use_container_width=True, height=min(38 + 35 * len(tr), 760),
                     column_config={"ล่าสุด": st.column_config.NumberColumn(format="%,.2f"), **pct})
    st.caption("🟢 ขาขึ้น = เหนือเส้น 50 และ 200 วัน · 🟡 พักตัว = ใต้เส้น 50 แต่ยังเหนือ 200 · "
               "🟠 เด้งในขาลง = เหนือเส้น 50 แต่ใต้ 200 · 🔴 ขาลง = ใต้ทั้งสองเส้น")

    # ---------- รายชื่อโมเมนตัม (เดิม) ----------
    exp = st.expander("🚀 รายชื่อหุ้นโมเมนตัมผ่านเกณฑ์ (โมเมนตัม 12-1 Top 20 ที่ไม่อยู่ในกล่อง ✕)")
    exp.caption(f"เรียงตามโมเมนตัม แสดง {TOP_N} อันดับแรก · ตัดขาดทุน = ราคาปิด −8% หรือ ราคาปิด − 3×ATR(14) ค่าที่สูงกว่า "
               "(จุดเริ่มต้นของกฎขาย X7) · เป้าหมาย = 2 เท่าของระยะตัดขาดทุน (อ้างอิงเท่านั้น กฎ X7 ใช้เส้น trailing ขายจริง) · "
               "🆕 = เพิ่งเข้ากลุ่มไม่เกิน 3 วัน")
    cfg = {k: st.column_config.NumberColumn(format="%,.2f") for k in ["ราคาปิด", "ตัดขาดทุน", "เป้าหมาย"]}
    cfg.update({k: st.column_config.NumberColumn(format="%+.1f%%") for k in
                ["วันนี้ %", "โมเมนตัม 12-1 %", "ตัดขาดทุน %", "เป้าหมาย %", "ห่างจุดสูงสุด %"]})
    cfg["ใหม่"] = st.column_config.CheckboxColumn("🆕")
    for mk in rep["markets"]:
        exp.markdown(f"**{mk['name']}**" + (f" · ราคาปิด {pd.Timestamp(mk['day']):%d/%m/%Y}" if mk["day"] else ""))
        f = mk["filter"]
        if f and not f["above"]:
            exp.warning(f"🚦 ตัวกรองตลาด: {f['index']} อยู่ใต้เส้น 200 วัน ({f['gap']:+.1f}%) · "
                       "ใน backtest วิธีโมเมนตัม + ไม่ ✕ ได้ผลดีที่สุดเมื่อซื้อเฉพาะตอนดัชนีอยู่เหนือเส้นนี้ "
                       "ช่วงนี้รายชื่อด้านล่างจึงเหมาะกับการเฝ้าดูมากกว่าเข้าซื้อ")
        elif f:
            exp.success(f"🚦 ตัวกรองตลาด: {f['index']} อยู่เหนือเส้น 200 วัน ({f['gap']:+.1f}%) · ผ่านเงื่อนไขตลาด")
        if not mk["picks"]:
            exp.caption("ยังไม่มีผลคำนวณล่าสุด (ดูแท็บสัญญาณเทคนิคแทน)")
            continue
        pk = pd.DataFrame(mk["picks"])
        exp.dataframe(_pct_style(pk), hide_index=True, use_container_width=True, column_config=cfg, height=38 + 35 * len(pk))

    txt = rep.get("text") or report_text(rep)
    with st.expander("📋 สรุปแบบข้อความ (กดไอคอนมุมขวาบนเพื่อคัดลอก ส่งต่อทาง LINE ได้)"):
        st.code(txt, language=None)
        st.download_button("⬇️ ดาวน์โหลดเป็นไฟล์ .txt", txt.encode("utf-8"),
                           file_name=f"daily_report_{d:%Y%m%d}.txt", mime="text/plain", key=f"rep_dl_{rep['date']}")
