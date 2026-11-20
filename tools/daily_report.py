"""แท็บ "รายงานประจำวัน" – สรุปจากเครื่องมือที่มีอยู่แล้วในแดชบอร์ดไว้หน้าเดียว (ไม่มีกฎใหม่)
1. อารมณ์ตลาดสหรัฐฯ วันนี้ เทียบเมื่อวาน / สัปดาห์ก่อน / เดือนก่อน + ความแข็งแรงของหุ้นใน SET100 และ S&P 500
2. สัดส่วนที่กฎของระบบให้ถือ: สูตรหุ้น-พันธบัตร 3 ชั้น + กฎ B (MA200 + อารมณ์ตลาด) รายดัชนี
3. ดัชนีไหนยังเป็นขาขึ้น / พักตัว / ขาลง (ราคาเทียบเส้นค่าเฉลี่ย 50 และ 200 วัน)
4. หุ้นผ่านเกณฑ์วันนี้ (โมเมนตัม 12-1 Top 20 ที่ไม่อยู่ในกล่อง ✕) ทั้ง SET100 และ S&P 500
   จุดตัดขาดทุนตามกฎขาย X7 (stop −8% หรือ ราคาปิด − 3×ATR14 ที่สูงกว่า) · เป้าหมายอ้างอิง = 2 เท่าของระยะตัดขาดทุน
ทั้งหมดเป็นผลของกฎที่ backtest ไว้ ไม่ใช่คำแนะนำการลงทุนเฉพาะบุคคล
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

from market_mood import mood_data, compute_mood, divergence_warning, signal_states, zone_of, COMPONENTS, STATES
from market_allocation import allocation_now
from set_signals import load_cache, load_prices, streak_table, cache_updated_at, BOX

HERE = os.path.dirname(os.path.abspath(__file__))
PRICES_CSV = os.path.join(HERE, "..", "files", "assets_prices.csv")
META_CSV = os.path.join(HERE, "..", "files", "assets_meta.csv")

INDEX_GROUPS = ["หุ้นสหรัฐฯ", "หุ้นเอเชีย", "หุ้นยุโรป", "หุ้นอเมริกาใต้"]
EXTRA = ["GC=F", "CL=F", "BTC-USD", "TLT"]  # สินทรัพย์อื่นที่น่าดูคู่กัน
US_IDX = {"^GSPC", "^IXIC", "^DJI", "^RUT"}  # ใช้กฎ B ได้ (อารมณ์ตลาดคำนวณจากตลาดสหรัฐฯ)
X7_STOP, X7_ATR, RR = 0.08, 3.0, 2.0
TOP_N = 10

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
    """ค่า ณ วันทำการล่าสุดที่ไม่เกิน (วันล่าสุด − days วันปฏิทิน)"""
    s = s.dropna()
    if days == 1:
        return s.iloc[-2] if len(s) > 1 else np.nan
    t = s.index[-1] - pd.Timedelta(days=days)
    s2 = s[s.index <= t]
    return s2.iloc[-1] if len(s2) else np.nan


PERIODS = [("เมื่อวาน", 1), ("สัปดาห์ก่อน", 7), ("เดือนก่อน", 30)]


def trend_status(s: pd.Series) -> tuple:
    s = s.dropna()
    last, ma50, ma200 = s.iloc[-1], s.rolling(50).mean().iloc[-1], s.rolling(200).mean().iloc[-1]
    if pd.isna(ma200):
        return "-", "#999", ma50, ma200
    if last > ma50 and last > ma200:
        return ("🟢 ขาขึ้น ไปต่อ", "#1b7837", ma50, ma200) if ma50 > ma200 else ("🟢 ฟื้นขึ้นเหนือทุกเส้น", "#5aae61", ma50, ma200)
    if last > ma200:
        return "🟡 พักตัวในขาขึ้น", "#d4a017", ma50, ma200
    if last > ma50:
        return "🟠 เด้งในขาลง", "#e08214", ma50, ma200
    return "🔴 ขาลง", "#b2182b", ma50, ma200


@st.cache_data(ttl=3600, show_spinner=False)
def _atr_high(syms: tuple) -> dict:
    out = {}
    for s, df in load_prices(syms, "1y").items():
        h, l, c = df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float)
        pc = np.r_[np.nan, c[:-1]]
        tr = np.nanmax(np.vstack([h - l, abs(h - pc), abs(l - pc)]), axis=0)
        atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
        out[s] = {"atr": float(atr), "hi52": float(np.nanmax(h[-252:])), "close": float(c[-1])}
    return out


def picks(market: str):
    """หุ้นผ่านเกณฑ์ (โมเมนตัม Top 20 + ไม่อยู่กล่อง ✕) จากผลที่ GitHub Actions คำนวณไว้ + จุดตัดขาดทุน/เป้าหมาย"""
    t, h = load_cache(market)
    if t is None:
        return None, None
    t = t[~t["stale"]]
    p = t[(t["mom_rank"] <= 20) & (t["box"] != "X")].sort_values("mom_rank").head(TOP_N).copy()
    stk, _, dates = streak_table(h)
    p["days"] = p["yahoo"].map(stk["pass"]).fillna(0).astype(int)
    p["strong"] = p["yahoo"].map(stk["strong"]).fillna(0).astype(int) > 0
    ah = _atr_high(tuple(p["yahoo"]))
    rows = []
    for r in p.itertuples():
        a = ah.get(r.yahoo)
        price = r.price
        if a is None:
            cut = price * (1 - X7_STOP)
            atr, hi52 = np.nan, np.nan
        else:
            atr, hi52 = a["atr"], a["hi52"]
            cut = max(price * (1 - X7_STOP), price - X7_ATR * atr)
        risk = price - cut
        rows.append({"yahoo": r.yahoo, "หุ้น": r.symbol, "กลุ่ม": r.sector_th if isinstance(r.sector_th, str) else "-",
                     "ราคาปิด": price, "วันนี้ %": r.chg1 * 100, "โมเมนตัม 12-1 %": r.mom * 100,
                     "อันดับ": int(r.mom_rank), "กล่อง": BOX.get(r.box, (r.box,))[0],
                     "คำแนะนำกล่อง": r.verdict, "ติดกลุ่ม (วัน)": r.days, "ใหม่": r.days <= 3,
                     "ตัดขาดทุน": cut, "ตัดขาดทุน %": -risk / price * 100,
                     "เป้าหมาย": price + RR * risk, "เป้าหมาย %": RR * risk / price * 100,
                     "สูงสุด 52 สัปดาห์": hi52, "ห่างจุดสูงสุด %": (price / hi52 - 1) * 100 if hi52 else np.nan})
    return pd.DataFrame(rows), t["date"].max()


def breadth(market: str):
    """จำนวนหุ้นแข็งแกร่ง (กล่อง G/B) และผ่านเกณฑ์ รายวัน จากประวัติสมาชิกกลุ่ม"""
    t, h = load_cache(market)
    if h is None:
        return None
    n = h.groupby("date")["yahoo"].nunique()
    g = h.groupby("date")[["strong", "watch"]].sum()
    return (g["strong"] / n * 100).rename("strong"), (g["watch"] / n * 100).rename("watch")


# ======================= UI =======================
def _card(title, value, sub, color):
    return (f'<div style="border-radius:12px;padding:10px 14px;background:{color}1f;border:1px solid {color};'
            f'height:100%;margin-bottom:6px"><div style="font-size:.82rem;opacity:.75">{title}</div>'
            f'<div style="font-size:1.35rem;font-weight:700;color:{color}">{value}</div>'
            f'<div style="font-size:.82rem">{sub}</div></div>')


def _delta(v, ref, fmt="{:+.0f}"):
    return "-" if pd.isna(ref) else fmt.format(v - ref)


def render_daily_report():
    upd = cache_updated_at()
    st.markdown(f"### 📰 รายงานประจำวัน · {pd.Timestamp.now(tz='Asia/Bangkok'):%d/%m/%Y}")
    st.caption("รวมผลจากแท็บอารมณ์ตลาด · สัดส่วนหุ้น-พันธบัตร · สัญญาณเทคนิค ไว้หน้าเดียว"
               + (f" · รายชื่อหุ้นอัปเดต {upd:%d/%m/%Y %H:%M} น." if upd is not None else ""))
    st.info(DISCLAIMER)
    lines = [f"📰 รายงานประจำวัน {pd.Timestamp.now(tz='Asia/Bangkok'):%d/%m/%Y}"]

    # ---------- 1. อารมณ์ตลาด ----------
    st.markdown("#### 🌡️ 1. อารมณ์ตลาดวันนี้ เทียบกับเมื่อวาน / สัปดาห์ก่อน / เดือนก่อน")
    with st.spinner("กำลังคำนวณดัชนีอารมณ์ตลาด ..."):
        d = mood_data()
        sc, _ = compute_mood(d)
        warn = divergence_warning(d)
    s = sc["score"]
    now = round(s.iloc[-1])
    zn, zc = zone_of(now)
    c = st.columns(4)
    c[0].markdown(_card(f"อารมณ์ตลาดสหรัฐฯ ณ {s.index[-1]:%d %b}", f"{now:.0f} · {zn}", "0 = กลัวสุดขีด · 100 = โลภสุดขีด", zc),
                  unsafe_allow_html=True)
    cmp = []
    for col, (lab, n) in zip(c[1:], PERIODS):
        ref = _ago(s, n)
        ref = round(ref) if pd.notna(ref) else ref
        rz, rc = zone_of(ref) if pd.notna(ref) else ("-", "#999")
        arrow = "" if pd.isna(ref) else ("▲ โลภขึ้น" if now > ref + 1 else "▼ กลัวขึ้น" if now < ref - 1 else "≈ ทรงตัว")
        col.markdown(_card(lab, "-" if pd.isna(ref) else f"{ref:.0f} · {rz}",
                           f"วันนี้ {_delta(now, ref)} จุด {arrow}", rc), unsafe_allow_html=True)
        cmp.append(f"{lab} {ref:.0f} ({_delta(now, ref)})" if pd.notna(ref) else f"{lab} -")
    lines.append(f"\n🌡️ อารมณ์ตลาดสหรัฐฯ: {now:.0f} {zn} · " + " · ".join(cmp))

    # ส่วนประกอบที่เปลี่ยนมากสุดใน 1 สัปดาห์
    wk = sc[sc.index <= sc.index[-1] - pd.Timedelta(days=7)]
    if len(wk):
        ch = pd.Series({COMPONENTS[k][0]: sc[k].iloc[-1] - wk[k].iloc[-1] for k in COMPONENTS}).dropna().sort_values()
        drv = [f"{k} {v:+.0f}" for k, v in list(ch.items())[:2] + list(ch.items())[-2:] if abs(v) >= 5]
        if drv:
            st.caption("ตัวที่ขยับมากสุดใน 1 สัปดาห์ (คะแนน 0–100): " + " · ".join(drv))
    wl = warn.iloc[-1]
    if bool(wl["warn"]):
        st.error(f"🚨 สัญญาณเตือนล่วงหน้าทำงาน: หุ้นกู้ขยะอ่อนผิดปกติ + กลุ่มอุตสาหกรรมเหนือเส้น 200 วันแค่ {wl['breadth200']:.0f}% "
                 f"ขณะ S&P 500 ห่างจุดสูงสุดเพียง {wl['off_high']:.1f}%")
        lines.append("🚨 สัญญาณเตือน: ข้างในตลาดเริ่มผุ ขณะดัชนียังใกล้จุดสูงสุด")
    else:
        st.caption(f"🚨 สัญญาณเตือนล่วงหน้า (เครดิต + breadth): ไม่ทำงาน · กลุ่มเหนือเส้น 200 วัน {wl['breadth200']:.0f}% · "
                   f"S&P 500 ห่างจุดสูงสุด {wl['off_high']:.1f}%")

    # ความแข็งแรงของหุ้นในแต่ละตลาด
    bc = st.columns(2)
    for col, (mk, name) in zip(bc, [("TH", "SET100"), ("US", "S&P 500")]):
        b = breadth(mk)
        if b is None:
            col.caption(f"{name}: ยังไม่มีผลคำนวณล่าสุด")
            continue
        sg, _ = b
        refs = [(lab, _ago(sg, n)) for lab, n in PERIODS]
        col.markdown(_card(f"{name} · % หุ้นแข็งแกร่ง (กล่องเขียว/น้ำเงิน)", f"{sg.iloc[-1]:.0f}%",
                           " · ".join(f"{lab} {v:.0f}%" for lab, v in refs if pd.notna(v)),
                           "#1b7837" if sg.iloc[-1] >= 40 else "#b2182b" if sg.iloc[-1] < 20 else "#888"),
                     unsafe_allow_html=True)
        lines.append(f"💪 {name} หุ้นแข็งแกร่ง {sg.iloc[-1]:.0f}% (" +
                     ", ".join(f"{lab} {v:.0f}%" for lab, v in refs if pd.notna(v)) + ")")

    # ---------- 2. สัดส่วน ----------
    st.markdown("#### ⚖️ 2. สัดส่วนที่กฎของระบบให้ถือ")
    px, meta = local_prices()
    names = dict(zip(meta["symbol"], meta["name"]))
    try:
        with st.spinner("กำลังคำนวณสูตรหุ้น-พันธบัตร ..."):
            al = allocation_now()
        why = []
        why.append("Valuation " + ("ถูก" if al["z"] > 1 else "แพง" if al["z"] < -1 else "ปกติ") + f" (z {al['z']:+.2f}) → ตั้งต้นหุ้น {al['strategic'] * 100:.0f}%")
        why.append("S&P 500 หลุดค่าเฉลี่ย 10 เดือน → ลดหุ้น" if al["down"] else "แนวโน้มยังเป็นขาขึ้น → ไม่ลด")
        if al["down"] and al["fear"]:
            why.append("กลัวสุดขีดและเริ่มคลาย → เพิ่มกลับบางส่วน")
        a1, a2 = st.columns([1, 1.4])
        a1.markdown(_card("สูตรหุ้น-พันธบัตร 3 ชั้น (พอร์ตสหรัฐฯ)", f"หุ้น {al['w'] * 100:.0f}% · พันธบัตร {100 - al['w'] * 100:.0f}%",
                          " · ".join(why), "#2a78d6"), unsafe_allow_html=True)
        lines.append(f"\n⚖️ สูตรหุ้น-พันธบัตร: หุ้น {al['w'] * 100:.0f}% / พันธบัตร {100 - al['w'] * 100:.0f}%")
    except Exception as e:
        a1, a2 = st.columns([1, 1.4])
        a1.warning(f"คำนวณสูตรหุ้น-พันธบัตรไม่ได้ ({type(e).__name__})")
    hold = {}
    for sym in px.columns:
        p = px[sym].dropna()
        if len(p) < 210:
            continue
        if sym in US_IDX:
            stt = signal_states(s, p, "B").iloc[-1]
            hold[sym] = (STATES[stt["state"]][0] if stt["state"] in STATES else "-", stt["pos"])
        else:
            up = p.iloc[-1] > p.rolling(200).mean().iloc[-1]
            hold[sym] = ("🔵 ถือ (เหนือเส้น 200 วัน)" if up else "🔴 ถือเงินสด (ใต้เส้น 200 วัน)", 1.0 if up else 0.0)
    rows = [{"ดัชนี": names.get(k, k), "กฎ": "B (MA200 + อารมณ์)" if k in US_IDX else "MA200",
             "สถานะ": hold[k][0], "ถือ %": hold[k][1] * 100} for k in ["^GSPC", "^IXIC", "^DJI", "^RUT", "^SET.BK"] if k in hold]
    a2.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"ถือ %": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
    a2.caption("ถือ % = สัดส่วนของเงินส่วนที่ลงในดัชนีนั้น ตามกฎ B (backtest 2008–2026 ในแท็บอารมณ์ตลาด) · "
               "ดัชนีนอกสหรัฐฯ ใช้เส้น 200 วันอย่างเดียว")
    lines.append("📌 กฎ B: " + " · ".join(f"{r['ดัชนี']} {r['ถือ %']:.0f}%" for r in rows))

    # ---------- 3. ดัชนี ----------
    st.markdown("#### 🧭 3. ดัชนีไหนยังไปต่อ อันไหนลง")
    idx_syms = [s_ for s_ in meta.loc[meta["group"].isin(INDEX_GROUPS), "symbol"] if s_ in px] + [s_ for s_ in EXTRA if s_ in px]
    trs = []
    for sym in idx_syms:
        p = px[sym].dropna()
        if len(p) < 30:
            continue
        stt, col, ma50, ma200 = trend_status(p)
        last = p.iloc[-1]
        r = {"ดัชนี / สินทรัพย์": names.get(sym, sym), "ล่าสุด": last, "ณ": p.index[-1].strftime("%d/%m")}
        for lab, n in [("1 วัน %", 1), ("1 สัปดาห์ %", 7), ("1 เดือน %", 30)]:
            ref = _ago(p, n)
            r[lab] = (last / ref - 1) * 100 if pd.notna(ref) else np.nan
        r["เทียบเส้น 50 วัน %"] = (last / ma50 - 1) * 100
        r["เทียบเส้น 200 วัน %"] = (last / ma200 - 1) * 100 if pd.notna(ma200) else np.nan
        r["สถานะ"] = stt
        trs.append(r)
    tr = pd.DataFrame(trs)
    pct = {c_: st.column_config.NumberColumn(format="%+.1f%%") for c_ in tr.columns if c_.endswith("%")}
    st.dataframe(tr.style.map(lambda v: "color:#1b7837" if isinstance(v, float) and v > 0 else
                              "color:#b2182b" if isinstance(v, float) and v < 0 else "",
                              subset=[c_ for c_ in tr.columns if c_.endswith("%")]),
                 hide_index=True, use_container_width=True, height=min(38 + 35 * len(tr), 760),
                 column_config={"ล่าสุด": st.column_config.NumberColumn(format="%,.2f"), **pct})
    st.caption("🟢 ขาขึ้น = เหนือเส้น 50 และ 200 วัน · 🟡 พักตัว = ใต้เส้น 50 แต่ยังเหนือ 200 · "
               "🟠 เด้งในขาลง = เหนือเส้น 50 แต่ใต้ 200 · 🔴 ขาลง = ใต้ทั้งสองเส้น")
    up = tr[tr["สถานะ"].str.startswith("🟢")]["ดัชนี / สินทรัพย์"].tolist()
    dn = tr[tr["สถานะ"].str.startswith("🔴")]["ดัชนี / สินทรัพย์"].tolist()
    lines.append("\n🟢 ยังเป็นขาขึ้น: " + (", ".join(up) or "-"))
    lines.append("🔴 ขาลง: " + (", ".join(dn) or "-"))

    # ---------- 4. หุ้น ----------
    st.markdown("#### ✅ 4. หุ้นผ่านเกณฑ์วันนี้ (โมเมนตัม 12-1 Top 20 ที่ไม่อยู่ในกล่อง ✕)")
    st.caption(f"เรียงตามโมเมนตัม แสดง {TOP_N} อันดับแรก · ตัดขาดทุน = ราคาปิด −8% หรือ ราคาปิด − 3×ATR(14) ค่าที่สูงกว่า "
               "(จุดเริ่มต้นของกฎขาย X7) · เป้าหมาย = 2 เท่าของระยะตัดขาดทุน (อ้างอิงเท่านั้น กฎ X7 ใช้เส้น trailing ขายจริง) · "
               "🆕 = เพิ่งเข้ากลุ่มไม่เกิน 3 วัน")
    cfg = {"ราคาปิด": st.column_config.NumberColumn(format="%,.2f"), "ตัดขาดทุน": st.column_config.NumberColumn(format="%,.2f"),
           "เป้าหมาย": st.column_config.NumberColumn(format="%,.2f"),
           "สูงสุด 52 สัปดาห์": st.column_config.NumberColumn(format="%,.2f"),
           **{k: st.column_config.NumberColumn(format="%+.1f%%") for k in
              ["วันนี้ %", "โมเมนตัม 12-1 %", "ตัดขาดทุน %", "เป้าหมาย %", "ห่างจุดสูงสุด %"]},
           "ใหม่": st.column_config.CheckboxColumn("🆕")}
    show = ["หุ้น", "ใหม่", "กลุ่ม", "กล่อง", "ราคาปิด", "วันนี้ %", "ตัดขาดทุน", "ตัดขาดทุน %", "เป้าหมาย", "เป้าหมาย %",
            "โมเมนตัม 12-1 %", "ติดกลุ่ม (วัน)", "คำแนะนำกล่อง", "ห่างจุดสูงสุด %"]
    for mk, name, idx in [("TH", "🇹🇭 SET100", "^SET.BK"), ("US", "🇺🇸 S&P 500", "^GSPC")]:
        with st.spinner(f"กำลังคำนวณจุดตัดขาดทุนของ {name} ..."):
            pk, day = picks(mk)
        st.markdown(f"**{name}**" + (f" · ราคาปิด {day:%d/%m/%Y}" if day is not None else ""))
        ip = px[idx].dropna() if idx in px else pd.Series(dtype=float)
        if len(ip) >= 200:
            ma = ip.rolling(200).mean().iloc[-1]
            gap = (ip.iloc[-1] / ma - 1) * 100
            if ip.iloc[-1] < ma:
                st.warning(f"🚦 ตัวกรองตลาด: {names.get(idx, idx)} อยู่ใต้เส้น 200 วัน ({gap:+.1f}%) · "
                           "ใน backtest วิธีโมเมนตัม + ไม่ ✕ ได้ผลดีที่สุดเมื่อซื้อเฉพาะตอนดัชนีอยู่เหนือเส้นนี้ "
                           "ช่วงนี้รายชื่อด้านล่างจึงเหมาะกับการเฝ้าดูมากกว่าเข้าซื้อ")
                lines.append(f"\n🚦 {names.get(idx, idx)} ใต้เส้น 200 วัน ({gap:+.1f}%) → รายชื่อ {name} ไว้เฝ้าดู")
            else:
                st.success(f"🚦 ตัวกรองตลาด: {names.get(idx, idx)} อยู่เหนือเส้น 200 วัน ({gap:+.1f}%) · ผ่านเงื่อนไขตลาด")
        if pk is None or pk.empty:
            st.caption("ยังไม่มีผลคำนวณล่าสุด (ดูแท็บสัญญาณเทคนิคแทน)")
            continue
        st.dataframe(pk[show], hide_index=True, use_container_width=True, column_config=cfg,
                     height=38 + 35 * len(pk))
        lines.append(f"\n✅ {name} (ราคาปิด {day:%d/%m}):")
        for r in pk.to_dict("records"):
            lines.append(f"• {r['หุ้น']}{' 🆕' if r['ใหม่'] else ''} ปิด {r['ราคาปิด']:,.2f} · ตัดขาดทุน {r['ตัดขาดทุน']:,.2f} "
                         f"({r['ตัดขาดทุน %']:+.1f}%) · เป้า {r['เป้าหมาย']:,.2f} ({r['เป้าหมาย %']:+.1f}%)")

    # ---------- สรุปแบบข้อความ ----------
    lines.append("\n" + DISCLAIMER)
    txt = "\n".join(lines)
    with st.expander("📋 สรุปแบบข้อความ (กดไอคอนมุมขวาบนเพื่อคัดลอก ส่งต่อทาง LINE ได้)"):
        st.code(txt, language=None)
        st.download_button("⬇️ ดาวน์โหลดเป็นไฟล์ .txt", txt.encode("utf-8"),
                           file_name=f"daily_report_{pd.Timestamp.now(tz='Asia/Bangkok'):%Y%m%d}.txt", mime="text/plain")
