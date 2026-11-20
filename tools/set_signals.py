"""สัญญาณเทคนิค SET100 – ตามกติกาในเอกสาร "SOMMIT DASHBOARD: สรุปสัญญาณและเงื่อนไข"
4 ชั้น: (1) กล่องสี EMA 10/50/200 + RSI  (2) ทิศทาง 29 วัน  (3) สัญญาณหนุน/เตือน -> คะแนนกฎ + ซื้อ/ถือ/รอ
(4) Market Lens: คะแนน Signal Spectrum -100..+100, กรอบม่วง, ★ หุ้นน่าจับตา

จุดที่เอกสารไม่ได้นิยามชัด และวิธีที่เลือกใช้ (ดูใน expander ของแท็บด้วย):
- RSI★ = RSI(14) > 50 และ RSI วันนี้ >= 95% ของเมื่อวาน (ลดลงไม่เกิน 5% แบบสัดส่วน)
- ↩ เขียว/เหลือง ต้องมี RSI★ ถ้าไม่มี -> "ไม่เข้ากล่อง"
- ยอด MACD = จุดสูงสุดเฉพาะที่ใน 22 วัน (นับวันล่าสุดเป็นยอดได้ถ้ายังขึ้นอยู่)
- ช่องขาลง = 70 วันที่จบก่อนวันนี้ 10–20 วัน (ลองทุกตำแหน่งแบบ SOMMIT), ขอบบน = เส้น regression + ส่วนเบี่ยงสูงสุด,
  ยอดต่ำลง = แบ่ง 4 ช่วง ต่ำลงอย่างน้อย 2 ใน 3 ครั้ง, ทะลุ = ราคาปิดเหนือขอบบน (ที่ลากต่อ) หลังจบช่อง
- รูปทรงรายเดือน, RSI, ยอด MACD และกล่อง ✓ เหลือง ปรับตามโค้ดจริงของ SOMMIT (ดู RULES_MD)
"""
import datetime as dt
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "..", "files")

# key: (ป้ายในตาราง, ความหมาย, ลำดับเรียง, คะแนน spectrum)
BOX = {
    "G": ("🟢 ✓", "แข็งแรงเต็มที่ (เหนือ EMA 10/50/200 + RSI★)", 1, 40),
    "B": ("🔵 ✓", "แข็งแรง แต่ RSI ยังไม่ยืนยัน", 2, 40),
    "Y": ("🟡 ✓", "พักตัวในขาขึ้น (ต่ำกว่า EMA10 แต่เหนือ 50/200)", 3, 5),
    "EYE": ("👀", "เด้งระยะสั้นในแนวโน้มยาวขาขึ้น (ต่ำกว่า EMA50)", 4, 5),
    "RG": ("🟢 ↩", "ฟื้นตัว ใกล้ทดสอบ EMA200", 5, 25),
    "RY": ("🟡 ↩", "เด้งระยะสั้นในขาลง", 6, 5),
    "X": ("🔴 ✕", "อ่อนแรง / ขาลง", 7, -35),
    "-": ("–", "ไม่เข้ากล่อง", 8, 0),
}
LEVELS = {2: "บวก", 1: "กลาง-บวก", 0: "กลาง", -1: "ระวัง", -2: "ลบ"}
# key: (สัญลักษณ์, ชื่อ, คะแนนกฎ, คะแนน spectrum, ลำดับเรียง ▲ ก่อน)
DIRS = {
    "UP": ("▲", "ขาขึ้น", 1, 5, 0), "PB": ("↗", "ขึ้นแล้วย่อตัว", 1, 8, 1), "V": ("V", "กลับตัวขึ้น", 1, 15, 2),
    "SW": ("↔", "Sideways", 0, 0, 3), "LA": ("Λ", "กลับตัวลง", -1, -12, 4), "DN": ("▼", "ขาลง", -1, -20, 5),
}
MONTHLY = {"UP": ("ขาขึ้น", 15), "RU": ("กลับตัวขึ้น", 10), "SW": ("Sideway", 0), "RD": ("กลับตัวลง", -10),
           "DN": ("ขาลง", -15)}
VERDICTS = ["ซื้อ", "ถือ", "รอ", "ไม่ซื้อ", "ขาย / ลดสถานะ"]
VERDICT_ICON = {"ซื้อ": "🟢 ซื้อ", "ถือ": "🔵 ถือ", "รอ": "⚪ รอ", "ไม่ซื้อ": "🟠 ไม่ซื้อ", "ขาย / ลดสถานะ": "🔴 ขาย / ลดสถานะ"}


# ======================= ตัวชี้วัด =======================
def indicators(df: pd.DataFrame) -> dict:
    """df: Open/High/Low/Close/Volume -> dict ของ numpy array (คำนวณแบบไม่มองอนาคต)"""
    c = df["Close"]
    out = {"o": df["Open"], "h": df["High"], "l": df["Low"], "c": c, "v": df["Volume"].fillna(0)}
    for n in (10, 50, 100, 200):
        out[f"e{n}"] = c.ewm(span=n, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd"], out["sig"] = macd, macd.ewm(span=9, adjust=False).mean()
    a = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    idx = pd.DatetimeIndex(df.index)
    a["month"] = np.asarray(idx.year * 12 + idx.month)
    a["dates"] = idx
    return a


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last",
                                  "Volume": "sum"}).dropna(subset=["Close"])


# ======================= ชั้น 1: กล่องสี + ระดับ =======================
def rsi_at(c, i, win=260):
    """RSI(14) แบบ SOMMIT: Wilder เริ่มนับใหม่จาก 260 วันล่าสุด · คืน (RSI วันนี้, RSI เมื่อวาน)"""
    v = c[max(0, i - win + 1):i + 1]
    if len(v) < 16:
        return np.nan, np.nan
    d = np.diff(v)
    g = d[:14].clip(min=0).sum() / 14
    lo = -d[:14].clip(max=0).sum() / 14

    def R():
        return 100.0 if lo == 0 else 100 - 100 / (1 + g / lo)
    prev = np.nan
    for x in d[14:]:
        prev = R()
        g = (g * 13 + max(x, 0)) / 14
        lo = (lo * 13 + max(-x, 0)) / 14
    return R(), prev


def box_level(a, i):
    c = a["c"][i]
    a10, a50, a200 = c > a["e10"][i], c > a["e50"][i], c > a["e200"][i]
    r, rp = rsi_at(a["c"], i)
    star = bool(r > 50 and r >= rp * 0.95)
    if a10 and a50 and a200:
        box = "G" if star else "B"
    elif not a10 and a50 and a200:
        box = "-" if star else "Y"  # โค้ด SOMMIT: ✓ เหลือง ต้อง "ไม่" ผ่าน RSI★ (เอกสารไม่ได้เขียนไว้)
    elif a10 and not a50 and a200:
        box = "EYE"
    elif a10 and a50 and not a200:
        box = "RG" if star else "-"
    elif a10 and not a50 and not a200:
        box = "RY" if star else "-"
    elif not a10 and (not a50 or not a200):
        box = "X"
    else:
        box = "-"
    if a10 and a50 and a200:
        lvl = 2
    elif (not a10 and a50 and a200) or (a10 and a50 and not a200):
        lvl = 1
    elif (a10 and not a50 and a200) or (not a10 and a50 and not a200):
        lvl = 0
    elif (not a10 and not a50 and a200) or (a10 and not a50 and not a200):
        lvl = -1
    else:
        lvl = -2
    return box, lvl, star


# ======================= ชั้น 2: ทิศทาง 29 วัน =======================
def _fit(y):
    x = np.arange(len(y))
    slope, icpt = np.polyfit(x, y, 1)
    fit = slope * x + icpt
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - fit) ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
    return slope, icpt, r2


def direction(c):
    n = len(c)
    if n < 20:
        return "SW", 0.0, 0.0
    _, _, r2 = _fit(np.log(c))
    tot = c[-1] / c[0] - 1
    imax, imin = int(c.argmax()), int(c.argmin())

    def mid(k):
        return 0.2 * (n - 1) <= k <= 0.8 * (n - 1)

    if mid(imax) and c[imax] / c[:imax + 1].min() - 1 >= 0.05 and c[-1] / c[imax] - 1 <= -0.06 and tot < 0.03:
        return "LA", tot, r2
    if mid(imin) and c[imin] / c[:imin + 1].max() - 1 <= -0.05 and c[-1] / c[imin] - 1 >= 0.06 and tot > -0.03:
        return "V", tot, r2
    if tot >= 0.03 and c[-1] / c.max() - 1 <= -0.05:
        return "PB", tot, r2
    if (tot >= 0.04 and r2 >= 0.35) or (tot >= 0.06 and r2 >= 0.15):
        return "UP", tot, r2
    if (tot <= -0.04 and r2 >= 0.35) or (tot <= -0.06 and r2 >= 0.15):
        return "DN", tot, r2
    return "SW", tot, r2


# ======================= ชั้น 3: สัญญาณ =======================
def candles(a, i):
    o, h, l, c, v = (a[k] for k in "ohlcv")
    g, r = c[i] > o[i], c[i] < o[i]
    gp, rp = c[i - 1] > o[i - 1], c[i - 1] < o[i - 1]
    av5 = v[i - 5:i].mean()
    bull, bear = [], []
    if g and rp and o[i] <= c[i - 1] and c[i] >= o[i - 1]:
        bull.append("แท่งเขียวกลืนแท่งแดง")
    if o[i] > h[i - 1] and c[i] > c[i - 1]:
        bull.append("เปิด Gap ขึ้น")
    if g and gp and c[i] > c[i - 1]:
        bull.append("เขียว 2 วันติด")
    if g and v[i] > av5:
        bull.append("แท่งเขียววอลุ่มสูง")
    if r and gp and o[i] >= c[i - 1] and c[i] <= o[i - 1]:
        bear.append("แท่งแดงกลืนแท่งเขียว")
    if o[i] < l[i - 1] and c[i] < c[i - 1]:
        bear.append("เปิด Gap ลง")
    if r and rp and c[i] < c[i - 1]:
        bear.append("แดง 2 วันติด")
    if r and v[i] > av5:
        bear.append("แท่งแดงวอลุ่มสูง")
    return bull, bear


def macd_hh(a, i):
    m = a["macd"][i - 21:i + 1]  # 22 ค่าล่าสุด
    if len(m) < 10:
        return None
    # ยอด = สูงกว่าวันก่อน, ไม่ต่ำกว่าวันถัดไป และสูงสุดในช่วง ±3 วัน (ตามโค้ด SOMMIT)
    pk = [j for j in range(1, len(m)) if m[j] > m[j - 1] and (j == len(m) - 1 or m[j] >= m[j + 1])
          and m[j] == m[max(0, j - 3):j + 4].max()]
    if len(pk) < 2 or m[pk[-1]] <= m[pk[-2]]:
        return None
    if a["macd"][i] > 0 and a["macd"][i] > a["sig"][i]:
        return "strong"
    if a["macd"][i] < 0 and a["macd"][i] < a["sig"][i]:
        return "weak"
    return None


def channel_break(a, i, L=70, B_range=(10, 20)):
    """ลองทุกตำแหน่งช่องที่จบก่อนวันนี้ 10–20 วัน (แบบ SOMMIT) · คืนจำนวนวันตั้งแต่ทะลุ หรือ None"""
    for B in range(B_range[0], B_range[1] + 1):
        r = _channel_break_at(a, i, L, B)
        if r is not None:
            return r
    return None


def _channel_break_at(a, i, L, B):
    c = a["c"]
    s, e = i - B - L + 1, i - B
    if s < 0:
        return None
    w = np.log(c[s:e + 1])
    slope, icpt, r2 = _fit(w)
    if slope >= 0 or r2 < 0.40:
        return None
    hs = [w[g].max() for g in np.array_split(np.arange(L), 4)]
    if sum(hs[k + 1] < hs[k] for k in range(3)) < 2:
        return None
    off = (w - (slope * np.arange(L) + icpt)).max()
    upper = np.exp(slope * np.arange(L, L + B) + icpt + off)
    hit = np.where(c[e + 1:i + 1] > upper)[0]
    if not len(hit):
        return None
    bday = e + 1 + int(hit[0])
    if c[i] < upper[-1] * 0.97 or c[i] <= a["e50"][i]:
        return None
    if c[i] / c[max(s, bday - 20):bday + 1].min() - 1 < 0.05:
        return None
    return i - bday


def smash(a, i):
    c = a["c"]
    w = c[i - 39:i + 1]
    k = int(w.argmax())
    if not 1 <= 39 - k <= 3:
        return False
    j = int(w[:k + 1].argmin())
    return bool(w[k] / w[j] - 1 >= 0.08 and k - j >= 8 and c[i] / w[k] - 1 <= -0.03 and c[i] < a["o"][i])


def monthly_shape(a, i):
    """ราคาปิดสิ้นเดือน 6 เดือนก่อน + ราคาล่าสุด (เดือนนี้ยังไม่จบ) -> ขาขึ้น/กลับตัว/ขาลง/Sideway"""
    mo = a["month"]
    prev = mo[:i + 1] < mo[i]
    if prev.sum() < 7:
        return "SW"
    ends = np.where(prev & (np.r_[mo[1:], -1][:i + 1] != mo[:i + 1]))[0][-6:]
    M = np.r_[a["c"][ends], a["c"][i]]  # 7 จุด = 6 เดือน
    if len(M) < 7:
        return "SW"
    # ตามโค้ด SOMMIT (mPat): แท่งเดือน = เปิดที่ราคาปิดเดือนก่อน · เช็กกลับตัวก่อน แล้วขาลง ขาขึ้น
    r = M[1:] / M[:-1] - 1
    chg6, chg3, pc0 = M[-1] / M[0] - 1, M[-1] / M[-4] - 1, M[-2] / M[0] - 1
    streak, sgn = 0, 0  # จำนวนเดือนติดกันทิศเดียวกันนับจากล่าสุด (เดือนที่ขยับ < 0.3% = หยุด ยกเว้นเดือนล่าสุดให้ข้าม)
    for k in range(len(r) - 1, -1, -1):
        if abs(r[k]) < 0.003:
            if k == len(r) - 1:
                continue
            break
        s = 1 if r[k] > 0 else -1
        if not sgn:
            sgn = s
        if s != sgn:
            break
        streak += 1
    if r[-2] <= -0.01 and r[-1] >= 0.015 and M[-1] >= M[-3] and pc0 < 0:
        return "RU"
    if r[-2] >= 0.01 and r[-1] <= -0.015 and M[-1] <= M[-3] and pc0 > 0:
        return "RD"
    if chg6 <= -0.08 or (chg6 <= -0.03 and sgn < 0 and streak >= 3) or (chg3 <= -0.08 and chg6 <= -0.03):
        return "DN"
    if chg6 >= 0.08 or (chg6 >= 0.03 and sgn > 0 and streak >= 3) or (chg3 >= 0.08 and chg6 >= 0.03):
        return "UP"
    return "SW"


# ======================= รวมทุกชั้น =======================
def evaluate(a, i) -> dict:
    box, lvl, star = box_level(a, i)
    d, tot, r2 = direction(a["c"][max(0, i - 28):i + 1])
    bull, bear = candles(a, i)
    hh = macd_hh(a, i)
    cb = channel_break(a, i)
    v = a["v"]
    vratio = v[i - 4:i + 1].mean() / max(v[i - 19:i + 1].mean(), 1)
    mv = bool(a["macd"][i] > a["sig"][i] and a["macd"][i - 1] <= a["sig"][i - 1]
              and v[i] >= 1.8 * max(v[i - 20:i].mean(), 1))
    sm = smash(a, i)
    ms = monthly_shape(a, i)
    sup, warn = [], []
    if bull:
        sup.append("▲ แท่งเทียนขาขึ้น")
    if hh == "strong":
        sup.append("MACD Higher High")
    if cb is not None:
        sup.append("ทะลุช่องขาลง")
    if d == "V" and vratio >= 1.2:
        sup.append("V + วอลุ่มหนุน")
    if mv:
        sup.append("MV")
    if bear:
        warn.append("▼ แท่งเทียนขาลง")
    if sm:
        warn.append("ขึ้นทำยอดแล้วโดนทุบ")
    score = lvl + DIRS[d][2] + len(sup) - len(warn)
    if lvl == -2 and d in ("DN", "LA") and score <= -3:
        verdict = "ขาย / ลดสถานะ"
    elif score <= -1:
        verdict = "ไม่ซื้อ"
    elif score >= 4 and lvl == 2 and not warn:
        verdict = "ซื้อ"
    elif lvl >= 1 and score >= 2:
        verdict = "ถือ"
    else:
        verdict = "รอ"
    spec = (BOX[box][3] + (10 if bull else 0) - (10 if bear else 0) + DIRS[d][3] + MONTHLY[ms][1]
            + (15 if hh == "strong" else 0) + (10 if cb is not None else 0) - (10 if sm else 0))
    spec = int(np.clip(spec, -100, 100))
    purple = box in ("G", "B") and d == "V" and ms == "UP" and hh == "strong"
    watch = (box == "RG" and bool(bull)) or (d == "V" and box != "EYE" and not (box == "X" and bull))
    c = a["c"][i]
    return dict(box=box, lvl=lvl, rsi_star=star, rsi=a["rsi"][i], dir=d, dir_tot=tot, dir_r2=r2,
                bull=bull, bear=bear, macd_hh=hh, chan_days=cb, vratio=vratio, mv=mv, smash=sm, monthly=ms,
                sup=sup, warn=warn, score=score, verdict=verdict, spec=spec, purple=purple, white=spec >= 75,
                watch=watch, near200=abs(c / a["e200"][i] - 1) < 0.02,
                ema={n: a[f"e{n}"][i] for n in (10, 50, 100, 200)}, price=c,
                chg1=c / a["c"][i - 1] - 1)


def group_flags(a, i) -> dict:
    """เฉพาะเงื่อนไขที่ใช้ตัดสินกลุ่มเด่น (เร็วกว่า evaluate ทั้งชุด) · นิยามเดียวกับ evaluate()"""
    box, _, _ = box_level(a, i)
    d, _, _ = direction(a["c"][max(0, i - 28):i + 1])
    bull, _ = candles(a, i)
    strong = box in ("G", "B")
    return {"strong": strong, "watch": box in ("EYE", "RG"),
            "purple": strong and d == "V" and monthly_shape(a, i) == "UP" and macd_hh(a, i) == "strong",
            "star": (box == "RG" and bool(bull)) or (d == "V" and box != "EYE" and not (box == "X" and bull)),
            "x": box == "X"}


GKEYS = ["pass", "mom", "strong", "watch", "purple", "star"]


def streak_table(h: pd.DataFrame) -> tuple:
    """h = สมาชิกกลุ่มรายวัน -> (ตารางจำนวนวันติดต่อเนื่องถึงวันล่าสุดของแต่ละกลุ่ม, ตาราง "เมื่อวานอยู่ แต่วันนี้หลุด", รายการวันที่)"""
    dates = sorted(h["date"].unique())
    stk, out = {}, {}
    for k in GKEYS:
        piv = h.pivot_table(index="date", columns="yahoo", values=k, aggfunc="max").reindex(dates)
        arr = piv.fillna(False).astype(bool).to_numpy()[::-1]  # แถวแรก = วันล่าสุด
        nf = ~arr
        stk[k] = pd.Series(np.where(nf.any(axis=0), nf.argmax(axis=0), arr.shape[0]), index=piv.columns)
        out[k] = pd.Series(arr[1] & ~arr[0] if len(arr) > 1 else np.zeros(arr.shape[1], bool), index=piv.columns)
    return pd.DataFrame(stk), pd.DataFrame(out), dates


@st.cache_data(ttl=3600, show_spinner=False)
def membership_history(syms: tuple, market: str, asof=None, n: int = 40) -> pd.DataFrame:
    """สมาชิกกลุ่มเด่นของทุกหุ้น ย้อนหลัง n วันทำการถึงวันที่ดู (คำนวณจากราคา ไม่ต้องรอบันทึก)"""
    px = load_prices(syms, period_of(market))
    recs = []
    for s, df in px.items():
        if asof is not None:
            df = df[df.index <= pd.Timestamp(asof)]
        if len(df) < 260:
            continue
        a = indicators(df)
        adj = df["Adj Close"].to_numpy() if "Adj Close" in df else a["c"]
        for i in range(len(df) - n, len(df)):
            f = group_flags(a, i)
            f["yahoo"], f["date"] = s, df.index[i]
            f["mom_v"] = adj[i - 21] / adj[i - 252] - 1
            recs.append(f)
    h = pd.DataFrame(recs)
    if h.empty:
        return h
    keep = h["date"].value_counts()
    keep = sorted(keep[keep >= keep.max() * 0.5].index)[-n:]  # วันทำการของตลาด (ตัดวันของหุ้นที่ข้อมูลหาย)
    h = h[h["date"].isin(keep)].copy()
    h["mom"] = h.groupby("date")["mom_v"].rank(ascending=False, method="first") <= 20
    h["pass"] = h["mom"] & ~h["x"].astype(bool)
    return h


def summary_text(lvl, n_sup, n_warn):
    good = lvl >= 1
    if n_sup and not n_warn:
        return "น้ำหนักไปทางบวกหลายชั้น" if good else "กำลังฟื้น แต่ยังรอให้เส้น EMA ยืนยัน"
    if n_warn and not n_sup:
        return "เส้น EMA ยังดี แต่ให้ระวังการย่อตัว" if good else "น้ำหนักไปทางลบ"
    if n_sup and n_warn:
        return "สัญญาณสวนกัน ยังไม่ชัด"
    return "ยังไม่มีสัญญาณหนุนหรือเตือน ดูระดับ EMA เป็นหลัก"


# ======================= ข้อมูล =======================
@st.cache_data(ttl=3600, show_spinner="กำลังดึงราคาย้อนหลังของหุ้นทั้งชุด (ครั้งแรกอาจใช้เวลาราว 1–2 นาที) ...")
def load_prices(syms: tuple, period: str = "10y") -> dict:
    # ราคาจริง (ไม่ปรับปันผล ปรับเฉพาะ split) ให้ตรงกับ set.or.th ที่ SOMMIT ใช้ · เก็บ Adj Close ไว้คิดผลตอบแทนใน backtest
    d = yf.download(list(syms), period=period, interval="1d", group_by="ticker", auto_adjust=False,
                    progress=False, threads=True)
    out = {}
    for s in syms:
        try:
            x = d[s].dropna(subset=["Close"])
        except KeyError:
            continue
        x.index = pd.DatetimeIndex(x.index.tz_localize(None) if x.index.tz is not None else x.index)
        if len(x) >= 230:
            out[s] = x
    return out


MY_CSV = os.path.join(FILES, "my_list.csv")  # รายชื่อของฉัน: หุ้น/ETF นอกดัชนีที่อยากให้สแกนสัญญาณด้วย


def load_my_list() -> pd.DataFrame:
    try:
        return pd.read_csv(MY_CSV, comment="#", encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=["yahoo", "name", "kind"])


def load_sectors(market: str = "TH") -> pd.DataFrame:
    if market == "MY":
        m = load_my_list()
        return pd.DataFrame({"yahoo": m["yahoo"], "sector": m["kind"], "industry": m["name"]})
    if market == "US":  # GICS sector + ชื่อบริษัทจาก files/us_indices.csv
        p = os.path.join(FILES, "us_indices.csv")
        if not os.path.exists(p):
            return pd.DataFrame(columns=["yahoo", "sector", "industry"])
        u = pd.read_csv(p, comment="#")
        return u.loc[u["sp500"] == 1, ["yahoo", "sector", "name"]].rename(columns={"name": "industry"})
    p = os.path.join(FILES, "set100_sectors.csv")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["yahoo", "sector", "industry", "shares"])
    return pd.read_csv(p)[["yahoo", "sector", "industry", "shares"]]


SECTOR_TH = {"Financial Services": "การเงิน", "Consumer Cyclical": "สินค้าฟุ่มเฟือย/บริการ", "Industrials": "อุตสาหกรรม",
             "Utilities": "สาธารณูปโภค", "Consumer Defensive": "สินค้าจำเป็น", "Real Estate": "อสังหาริมทรัพย์",
             "Energy": "พลังงาน", "Healthcare": "การแพทย์", "Basic Materials": "วัตถุดิบ",
             "Communication Services": "สื่อสาร", "Technology": "เทคโนโลยี",
             # GICS (ชื่อใน us_indices.csv)
             "Information Technology": "เทคโนโลยี", "Health Care": "การแพทย์", "Financials": "การเงิน",
             "Consumer Discretionary": "สินค้าฟุ่มเฟือย", "Consumer Staples": "สินค้าจำเป็น", "Materials": "วัตถุดิบ"}


def period_of(market: str) -> str:
    return "10y" if market == "TH" else "5y"  # US และรายชื่อของฉัน = 5 ปี


# ======================= บันทึกรายชื่อหุ้นเด่นรายวัน (files/signal_log/<ตลาด>_<ปี>.csv) =======================
LOG_DIR = os.path.join(FILES, "signal_log")
LOG_KEY = {"TH": "set100", "US": "sp500", "MY": "mylist"}
LOG_COLS = ["date", "yahoo", "symbol", "groups", "verdict", "score", "spec", "box", "dir", "mom", "mom_rank", "price"]


def snapshot_rows(t: pd.DataFrame) -> pd.DataFrame:
    """หุ้นที่ติดกลุ่มเด่นอย่างน้อย 1 กลุ่ม หรือได้ผลอ่าน "ซื้อ" ณ วันทำการล่าสุดของตลาด"""
    t = t[~t["stale"]]
    t = t[t["date"] == t["date"].max()]
    flags = pd.DataFrame({k: rule(t).fillna(False).astype(bool) for k, _, _, rule in ALL_GROUPS})
    grp = flags.apply(lambda r: "|".join(k for k in flags.columns if r[k]), axis=1)
    keep = (grp != "") | (t["verdict"] == "ซื้อ")
    out = t[keep].assign(groups=grp[keep])[LOG_COLS].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["mom"] = out["mom"].round(4)
    out["price"] = out["price"].round(4)
    return out.sort_values(["score", "spec"], ascending=False)


def save_snapshot(t: pd.DataFrame, market: str) -> tuple:
    rows = snapshot_rows(t)
    if rows.empty:
        return None, 0
    day = rows["date"].iloc[0]
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{LOG_KEY[market]}_{day[:4]}.csv")
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=LOG_COLS)
    new = pd.concat([old[old["date"] != day], rows], ignore_index=True).sort_values(["date", "score"], ascending=[True, False])
    new.to_csv(path, index=False, encoding="utf-8")
    return day, len(rows)


def _log_files(market: str) -> tuple:
    if not os.path.isdir(LOG_DIR):
        return ()
    fs = sorted(f for f in os.listdir(LOG_DIR) if f.startswith(LOG_KEY[market] + "_") and f.endswith(".csv"))
    return tuple((f, os.path.getmtime(os.path.join(LOG_DIR, f))) for f in fs)


@st.cache_data(show_spinner=False)
def _read_log(files: tuple) -> pd.DataFrame:
    parts = [pd.read_csv(os.path.join(LOG_DIR, f)) for f, _ in files]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=LOG_COLS)


# ======================= ผลคำนวณล่วงหน้า (files/signal_cache) ให้หน้าเว็บเปิดได้ทันที =======================
CACHE_DIR = os.path.join(FILES, "signal_cache")
LIST_COLS = ["bull", "bear", "sup", "warn"]
CACHE_MAX_AGE_DAYS = 4  # เก่ากว่านี้ (เช่น Actions ล้มหลายวัน) จะคำนวณสดแทน


def save_cache(t: pd.DataFrame, h: pd.DataFrame, market: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    t2 = t.drop(columns=["ema"], errors="ignore").copy()
    for c in LIST_COLS:
        t2[c] = t2[c].map(lambda x: "|".join(x) if isinstance(x, (list, tuple)) else "")
    num = t2.select_dtypes("float").columns
    t2[num] = t2[num].round(4)
    t2.to_csv(os.path.join(CACHE_DIR, f"{LOG_KEY[market]}_table.csv"), index=False, encoding="utf-8")
    # ประวัติแบบย่อ: เก็บเฉพาะแถวที่ติดอย่างน้อย 1 กลุ่ม (คอลัมน์ groups = "pass|mom|...") + รายการวันที่ทั้งหมด
    flags = h[GKEYS].astype(bool)
    grp = flags.apply(lambda r: "|".join(k for k in GKEYS if r[k]), axis=1)
    hh = pd.DataFrame({"date": pd.to_datetime(h["date"]).dt.strftime("%Y-%m-%d"), "yahoo": h["yahoo"], "groups": grp})
    hh = hh[hh["groups"] != ""]
    days = pd.DataFrame({"date": sorted(pd.to_datetime(h["date"]).dt.strftime("%Y-%m-%d").unique()), "yahoo": "__day__",
                         "groups": ""})
    pd.concat([days, hh]).to_csv(os.path.join(CACHE_DIR, f"{LOG_KEY[market]}_hist.csv"), index=False, encoding="utf-8")


def _cache_files(market: str) -> tuple:
    ps = [os.path.join(CACHE_DIR, f"{LOG_KEY[market]}_{k}.csv") for k in ("table", "hist")]
    return tuple((p, os.path.getmtime(p)) for p in ps if os.path.exists(p))


@st.cache_data(show_spinner=False)
def _read_cache(files: tuple):
    if len(files) < 2:
        return None, None
    t = pd.read_csv(files[0][0], keep_default_na=True)
    for c in LIST_COLS:
        t[c] = t[c].fillna("").map(lambda s: [x for x in str(s).split("|") if x])
    t["date"] = pd.to_datetime(t["date"])
    t["box"] = t["box"].fillna("-")
    t["box_tf"] = t["box_tf"].fillna("-")
    t["macd_hh"] = t["macd_hh"].where(t["macd_hh"].notna(), None)
    for c in ("purple", "white", "watch", "near200", "rsi_star", "mv", "smash", "stale"):
        t[c] = t[c].astype(str).str.lower().eq("true")
    hh = pd.read_csv(files[1][0])
    days = pd.to_datetime(hh.loc[hh["yahoo"] == "__day__", "date"])
    hh = hh[hh["yahoo"] != "__day__"]
    # กางกลับเป็นตารางเต็ม (ทุกหุ้น × ทุกวัน) เหมือนผลของ membership_history
    grid = pd.MultiIndex.from_product([days, t["yahoo"].unique()], names=["date", "yahoo"]).to_frame(index=False)
    hh = hh.assign(date=pd.to_datetime(hh["date"]))
    h = grid.merge(hh, on=["date", "yahoo"], how="left")
    g = h["groups"].fillna("")
    for k in GKEYS:
        h[k] = g.str.split("|").map(lambda xs, k=k: k in xs)
    return t, h.drop(columns="groups")


UPDATE_NOTE = "ตั้งเวลาไว้ 05:30 น. แต่ GitHub มักเริ่มช้า ปกติอัปเดตวันทำการช่วงสายราว 10:00–11:30 น."


def cache_updated_at():
    """เวลาที่ GitHub Actions อัปเดตผลคำนวณล่วงหน้าครั้งล่าสุด (เวลาไทย) หรือ None"""
    p = os.path.join(CACHE_DIR, "updated_at.txt")
    try:
        t = pd.Timestamp(open(p, encoding="utf-8").read().strip())
        return t.tz_convert("Asia/Bangkok") if t.tz is not None else t.tz_localize("UTC").tz_convert("Asia/Bangkok")
    except Exception:
        return None


def load_cache(market: str):
    """(ตาราง, ประวัติสมาชิกกลุ่ม) ที่ GitHub Actions คำนวณไว้ · คืน (None, None) ถ้าไม่มีหรือเก่าเกินไป"""
    t, h = _read_cache(_cache_files(market))
    if t is None or (pd.Timestamp.today().normalize() - t["date"].max()).days > CACHE_MAX_AGE_DAYS:
        return None, None
    return t, h


def load_log(market: str) -> pd.DataFrame:
    return _read_log(_log_files(market))  # คีย์แคชผูกกับเวลาแก้ไขไฟล์ จึงอ่านใหม่เมื่อมีบันทึกเพิ่ม


HORIZONS = [5, 20, 60]


@st.cache_data(ttl=3600, show_spinner="กำลังวัดผลจริงจากบันทึก ...")
def log_results(files: tuple, market: str, syms: tuple) -> tuple:
    """ผลจริงหลังติดสัญญาณ: ทุกวันที่มีบันทึก × ทุกกลุ่ม -> ผลตอบแทน H วันทำการถัดไป ลบค่าเฉลี่ยหุ้นทุกตัวในตลาด
    คืน (ตารางสรุป, จำนวนวันที่บันทึก, จำนวนวันที่วัดผลได้แล้วของแต่ละช่วง)"""
    log = _read_log(files)
    if log.empty:
        return pd.DataFrame(), 0, {}
    px = load_prices(syms, period_of(market))
    adj = pd.DataFrame({s: (df["Adj Close"] if "Adj Close" in df else df["Close"]) for s, df in px.items()}).sort_index()
    days = sorted(log["date"].unique())
    rows, ready = [], {H: 0 for H in HORIZONS}
    prev = {}
    for d in days:
        ts = pd.Timestamp(d)
        if ts not in adj.index:
            continue
        i = adj.index.get_loc(ts)
        L = log[log["date"] == d]
        cur = {k: set(L.loc[L["groups"].fillna("").str.contains(k), "yahoo"]) for k in GKEYS}
        for H in HORIZONS:
            if i + H >= len(adj):
                continue
            ready[H] += 1
            fwd = adj.iloc[i + H] / adj.iloc[i] - 1
            ex = fwd - fwd.mean()
            for k in GKEYS:
                for y in cur[k]:
                    if pd.notna(ex.get(y)):
                        rows.append((d, k, y, H, ex[y], bool(prev) and y not in prev.get(k, set())))
        prev = cur
    if not rows:
        return pd.DataFrame(), len(days), ready
    r = pd.DataFrame(rows, columns=["date", "group", "yahoo", "H", "ex", "new"])
    out = []
    for k in GKEYS:
        for lab, m in [("ทุกตัวในกลุ่ม", r["group"] == k), ("เพิ่งเข้ากลุ่ม", (r["group"] == k) & r["new"])]:
            row = {"กลุ่ม": GROUP_ICON.get(k, k), "นับแบบ": lab}
            for H in HORIZONS:
                x = r.loc[m & (r["H"] == H), "ex"]
                row[f"{H} วัน: ชนะตลาด"] = x.mean() * 100 if len(x) else np.nan
                row[f"{H} วัน: % ชนะ"] = (x > 0).mean() * 100 if len(x) else np.nan
                row[f"{H} วัน: ครั้ง"] = len(x)
            out.append(row)
    return pd.DataFrame(out), len(days), ready


def _render_log_results(market: str, syms: tuple):
    _render_newbuy_results(market)
    st.divider()
    files = _log_files(market)
    res, n_days, ready = log_results(files, market, syms)
    st.markdown("**📒 ผลจริงหลังติดสัญญาณ (จากบันทึกรายวัน ไม่ใช่ backtest)**")
    st.caption("ทุกวันที่มีบันทึก: ผลตอบแทนของหุ้นในแต่ละกลุ่ม 5 / 20 / 60 วันทำการถัดไป ลบค่าเฉลี่ยหุ้นทุกตัวในตลาดช่วงเดียวกัน · "
               "ใช้รายชื่อที่บันทึกไว้จริงในวันนั้น จึงไม่มีปัญหารายชื่อหุ้นปัจจุบันแบบ backtest · "
               "\"เพิ่งเข้ากลุ่ม\" = ไม่อยู่ในกลุ่มนั้นในวันที่บันทึกก่อนหน้า")
    c = st.columns(4)
    c[0].metric("วันที่มีบันทึก", n_days)
    for col, H in zip(c[1:], HORIZONS):
        col.metric(f"วันที่วัดผล {H} วันได้แล้ว", ready.get(H, 0))
    if res.empty:
        st.info("ยังวัดผลไม่ได้: ต้องมีราคาหลังวันที่บันทึกอย่างน้อย 5 วันทำการ · บันทึกเริ่มสะสมทุกวันทำการ (ช่วงสาย) "
                "แล้วตารางนี้จะเติมเองอัตโนมัติ (ผลที่พอเชื่อได้ควรมีบันทึกอย่างน้อย 2–3 เดือน)")
        return
    cfg = {}
    for H in HORIZONS:
        cfg[f"{H} วัน: ชนะตลาด"] = st.column_config.NumberColumn(format="%+.2f%%")
        cfg[f"{H} วัน: % ชนะ"] = st.column_config.NumberColumn(format="%.0f%%")
    st.dataframe(res, hide_index=True, use_container_width=True, column_config=cfg)
    st.caption("ช่วงที่บันทึกยังสั้น ตัวเลขจะแกว่งมากและเชื่อไม่ได้ · ช่วงถือซ้อนทับกันระหว่างวัน · ยังไม่หักค่าซื้อขาย · ไม่ใช่คำแนะนำการลงทุน")


# ======================= ติดตามผลจริงของ "สัญญาณซื้อใหม่" (ทดสอบไปข้างหน้า) =======================
# GitHub Actions บันทึกทุกวันทำการ -> files/signal_log/newbuy_<set100|sp500>_<ปี>.csv
# 1 ไม้ = วันแรกที่ขึ้นสัญญาณซื้อ (days = 1) · ซื้อที่ราคาเปิดวันทำการถัดไป · ขายตามกฎ X7 (stop −8% + trailing 3×ATR)
NB_COLS = ["date", "yahoo", "symbol", "days", "n_sup", "mom_rank", "best", "price", "score"]
NB_COST = {"TH": 0.003, "US": 0.001, "MY": 0.001}  # ค่าซื้อขายไปกลับโดยประมาณ
NB_INDEX = {"TH": "^SET.BK", "US": "^GSPC", "MY": "^GSPC"}


def newbuy_rows(t: pd.DataFrame, market: str) -> pd.DataFrame:
    """หุ้นที่ขึ้นสัญญาณซื้อมาไม่เกิน NEWBUY_MAX วัน ณ แถวล่าสุดของแต่ละหุ้น"""
    n_all = t["mom_n"].max() if "mom_n" in t else t["mom_rank"].max()
    t = t[~t["stale"] & (t["verdict"] == "ซื้อ")].sort_values("yahoo")
    if t.empty:
        return pd.DataFrame(columns=NB_COLS)
    bs = buy_streaks(tuple((y, f"{d:%Y-%m-%d}") for y, d in zip(t["yahoo"], t["date"])), market)
    out = pd.DataFrame({"date": t["date"].dt.strftime("%Y-%m-%d"), "yahoo": t["yahoo"], "symbol": t["symbol"],
                        "days": t["yahoo"].map(bs).fillna(0).astype(int), "n_sup": t["sup"].map(len),
                        "mom_rank": t["mom_rank"], "price": t["price"].round(4), "score": t["score"]})
    out["best"] = out["n_sup"].ge(2) & out["mom_rank"].le(max(1, round(n_all * 0.2)))
    return out[out["days"].between(1, NEWBUY_MAX)][NB_COLS]


def save_newbuy(rows: pd.DataFrame, market: str) -> int:
    if rows.empty:
        return 0
    os.makedirs(LOG_DIR, exist_ok=True)
    for yr, g in rows.groupby(rows["date"].str[:4]):
        path = os.path.join(LOG_DIR, f"newbuy_{LOG_KEY[market]}_{yr}.csv")
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=NB_COLS)
        key = set(zip(g["date"], g["yahoo"]))
        old = old[[(d, y) not in key for d, y in zip(old["date"], old["yahoo"])]]
        new = pd.concat([old, g], ignore_index=True) if len(old) else g
        new.sort_values(["date", "best", "yahoo"], ascending=[True, False, True]).to_csv(path, index=False, encoding="utf-8")
    return len(rows)


def _nb_files(market: str) -> tuple:
    if not os.path.isdir(LOG_DIR):
        return ()
    fs = sorted(f for f in os.listdir(LOG_DIR) if f.startswith(f"newbuy_{LOG_KEY[market]}_"))
    return tuple((os.path.join(LOG_DIR, f), os.path.getmtime(os.path.join(LOG_DIR, f))) for f in fs)


@st.cache_data(ttl=3600, show_spinner="กำลังคำนวณผลจริงของสัญญาณซื้อ ...")
def newbuy_results(files: tuple, market: str) -> pd.DataFrame:
    """1 แถว = 1 ไม้ (วันแรกของสัญญาณ) พร้อมสถานะ X7 ณ ราคาล่าสุด"""
    from tv_chart import x7_exit
    if not files:
        return pd.DataFrame()
    log = pd.concat([pd.read_csv(f) for f, _ in files], ignore_index=True)
    tr = log[log["days"] == 1].drop_duplicates(["date", "yahoo"]).copy()
    if tr.empty:
        return pd.DataFrame()
    px = load_prices(tuple(sorted(tr["yahoo"].unique())), "1y")
    try:
        ip = pd.read_csv(os.path.join(FILES, "assets_prices.csv"), encoding="utf-8-sig")
        ip = ip[ip["symbol"] == NB_INDEX[market]]
        idx = pd.Series(ip["close"].to_numpy(float), index=pd.to_datetime(ip["date"])).sort_index()
    except Exception:
        idx = pd.Series(dtype=float)
    rows = []
    for r in tr.itertuples():
        df = px.get(r.yahoo)
        rec = {"วันที่สัญญาณ": r.date, "yahoo": r.yahoo, "หุ้น": r.symbol, "✅": bool(r.best), "หนุน": int(r.n_sup),
               "อันดับโมเมนตัม": r.mom_rank, "ราคาวันสัญญาณ": r.price}
        after = df.index[df.index > pd.Timestamp(r.date)] if df is not None else []
        if df is None or not len(after):
            rec["สถานะ"] = "⏳ รอซื้อ (ราคาเปิดวันถัดไป)"
            rows.append(rec)
            continue
        res = x7_exit(df, after[0])
        if not res.get("ok"):
            rec["สถานะ"] = "—"
            rows.append(rec)
            continue
        ex = res["exit"]
        done = ex is not None and ex.get("date") is not None
        p_out, d_out = (ex["price"], ex["date"]) if done else (res["last"], res["last_date"])
        ret = p_out / res["entry"] - 1 - (NB_COST[market] if done else NB_COST[market] / 2)
        mk = np.nan
        if len(idx):
            a, b = idx.asof(res["entry_date"] - pd.Timedelta(days=1)), idx.asof(pd.Timestamp(d_out))
            mk = b / a - 1 if pd.notna(a) and pd.notna(b) else np.nan
        held = df.index[(df.index >= res["entry_date"]) & (df.index <= pd.Timestamp(d_out))]
        rec.update({"ซื้อวันที่": f"{res['entry_date']:%Y-%m-%d}", "ราคาซื้อ": res["entry"],
                    "สถานะ": ("🔴 ขายแล้ว · " + ("stop −8%" if "stop" in ex["why"] else "หลุด trailing")) if done
                    else ("🟡 รอขาย (เกิดสัญญาณขายแล้ว)" if ex is not None else "🟢 ถืออยู่"),
                    "ขาย/ล่าสุด": p_out, "ถึงวันที่": f"{pd.Timestamp(d_out):%Y-%m-%d}", "ถือ (วันทำการ)": int(len(held)),
                    "ผลตอบแทน %": ret * 100, "ดัชนีช่วงเดียวกัน %": mk * 100, "ชนะดัชนี %": (ret - mk) * 100,
                    "เส้น stop": res["stop"], "เส้น trailing": res["trail_now"], "_closed": done})
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("วันที่สัญญาณ", ascending=False)


def _render_newbuy_results(market: str):
    st.markdown("**🟢 ผลจริงของ \"สัญญาณซื้อใหม่\" (ซื้อที่ราคาเปิดวันถัดไป · ขายตามกฎ X7)**")
    st.caption("บันทึกทุกวันทำการตั้งแต่ 18 พ.ย. 2026 · 1 ไม้ = วันแรกที่ขึ้นสัญญาณซื้อ · หักค่าซื้อขายโดยประมาณ · "
               "ไม้ที่ยังถืออยู่คิดจากราคาล่าสุด · ✅ = หนุน ≥ 2 + โมเมนตัม 20% บนสุด "
               "(backtest SET100 ปี 2015–2026: ชนะตลาดเฉลี่ย +2.7%/ไม้ · ไม้ที่ชนะราว 35%)")
    r = newbuy_results(_nb_files(market), market)
    if r.empty or "ผลตอบแทน %" not in r:
        st.info("ยังไม่มีไม้ที่ซื้อได้ · ระบบบันทึกสัญญาณทุกวันทำการ แล้วตารางนี้จะเติมเอง")
        if not r.empty:
            st.dataframe(r[["วันที่สัญญาณ", "หุ้น", "✅", "หนุน", "สถานะ"]], hide_index=True)
        return
    bought = r[r["ผลตอบแทน %"].notna()]
    rows = []
    for lab, g in [("ทุกสัญญาณ", bought), ("✅ หนุน ≥ 2 + โมเมนตัม", bought[bought["✅"]]),
                   ("ไม่มี ✅", bought[~bought["✅"]])]:
        c = g[g["_closed"] == True]  # noqa: E712
        rows.append({"กลุ่ม": lab, "ไม้ทั้งหมด": len(g), "ขายแล้ว": len(c), "ถืออยู่": len(g) - len(c),
                     "ชนะ % (ขายแล้ว)": (c["ผลตอบแทน %"] > 0).mean() * 100 if len(c) else np.nan,
                     "เฉลี่ย/ไม้ %": g["ผลตอบแทน %"].mean() if len(g) else np.nan,
                     "ชนะดัชนี/ไม้ %": g["ชนะดัชนี %"].mean() if len(g) else np.nan,
                     "แย่สุด %": g["ผลตอบแทน %"].min() if len(g) else np.nan,
                     "ดีสุด %": g["ผลตอบแทน %"].max() if len(g) else np.nan})
    pc = {k: st.column_config.NumberColumn(format="%+.2f%%") for k in ["เฉลี่ย/ไม้ %", "ชนะดัชนี/ไม้ %", "แย่สุด %", "ดีสุด %"]}
    pc["ชนะ % (ขายแล้ว)"] = st.column_config.NumberColumn(format="%.0f%%")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config=pc)
    st.caption("ช่วงแรกจำนวนไม้ยังน้อย ตัวเลขแกว่งมาก ควรตัดสินเมื่อมีไม้ที่ขายแล้วอย่างน้อย 30–50 ไม้ · ไม่ใช่คำแนะนำการลงทุน")
    show = ["วันที่สัญญาณ", "หุ้น", "✅", "หนุน", "สถานะ", "ซื้อวันที่", "ราคาซื้อ", "ขาย/ล่าสุด", "ถึงวันที่", "ถือ (วันทำการ)",
            "ผลตอบแทน %", "ชนะดัชนี %", "เส้น stop", "เส้น trailing"]
    cfg = {k: st.column_config.NumberColumn(format="%+.2f%%") for k in ["ผลตอบแทน %", "ชนะดัชนี %"]}
    cfg.update({k: st.column_config.NumberColumn(format="%,.2f") for k in ["ราคาซื้อ", "ขาย/ล่าสุด", "เส้น stop", "เส้น trailing"]})
    with st.expander(f"รายการทุกไม้ ({len(r)} ไม้)"):
        st.dataframe(r[[c for c in show if c in r]], hide_index=True, use_container_width=True, column_config=cfg)


GROUP_ICON = {"pass": "✅ ผ่านเกณฑ์", "mom": "🚀 โมเมนตัม", "strong": "💪 แข็งแกร่ง", "watch": "🌱 น่าติดตาม", "purple": "◆ กรอบม่วง",
              "star": "★ น่าจับตา"}


def _log_history(log: pd.DataFrame, sym: str):
    """ประวัติการติดกลุ่มหุ้นเด่นของหุ้นตัวนี้ จากบันทึกรายวัน"""
    h = log[log["yahoo"] == sym] if len(log) else log
    if not len(h):
        st.caption("📒 หุ้นตัวนี้ยังไม่มีในบันทึกรายชื่อรายวัน")
        return
    with st.expander(f"📒 ประวัติการติดสัญญาณจากบันทึกจริง · {h['date'].nunique()} วัน "
                     f"(ล่าสุด {pd.to_datetime(h['date']).max():%d/%m/%Y})"):
        v = h.sort_values("date", ascending=False)
        st.dataframe(pd.DataFrame({
            "วันที่": v["date"],
            "กลุ่มที่ติด": v["groups"].fillna("").map(lambda g: " · ".join(GROUP_ICON.get(k, k) for k in str(g).split("|") if k)),
            "ผลอ่าน": v["verdict"].map(VERDICT_ICON),
            "คะแนน": v["score"], "Spectrum": v["spec"],
            "โมเมนตัม 12 ด.": v["mom"] * 100, "ราคา": v["price"],
        }), hide_index=True, use_container_width=True, height=min(35 * len(v) + 40, 300), column_config={
            "คะแนน": st.column_config.NumberColumn(format="%+d"),
            "โมเมนตัม 12 ด.": st.column_config.NumberColumn(format="%+.0f%%"),
            "ราคา": st.column_config.NumberColumn(format="%.2f")})


@st.cache_data(ttl=3600, show_spinner=False)
def signal_table(syms: tuple, tf: str = "D", market: str = "TH", asof=None) -> pd.DataFrame:
    """asof = None ใช้ราคาล่าสุด · ใส่วันที่ = คำนวณสัญญาณย้อนหลังจากราคาปิด ณ วันนั้น (หรือวันทำการก่อนหน้า)"""
    px = load_prices(syms, period_of(market))
    rows = []
    for s, df in px.items():
        if asof is not None:
            df = df[df.index <= pd.Timestamp(asof)]
            if len(df) < 230:
                continue
        a = indicators(df)
        r = evaluate(a, len(df) - 1)
        # โมเมนตัม 12-1: ผลตอบแทนรวมปันผล 12 เดือน ไม่นับเดือนล่าสุด (ผล backtest ดีกว่าสัญญาณ SOMMIT ทั้งไทยและสหรัฐ)
        adj = df["Adj Close"].to_numpy() if "Adj Close" in df else a["c"]
        r["mom"] = adj[-22] / adj[-253] - 1 if len(adj) >= 253 else np.nan
        if tf != "D":  # กล่องสีจากแท่งรายสัปดาห์/รายเดือน (คะแนน Spectrum ยังใช้รายวันเสมอ)
            dd = resample(df, "W-FRI" if tf == "W" else "ME")
            if len(dd) >= 3:
                r["box_tf"], r["lvl_tf"], _ = box_level(indicators(dd), len(dd) - 1)
        recent = df.index[df.index >= df.index[-1] - pd.Timedelta(days=150)]
        r["gap_days"] = int(recent.to_series().diff().dt.days.max() or 0)  # ช่องว่างข้อมูลยาวสุดใน ~5 เดือนล่าสุด
        r.setdefault("box_tf", r["box"])
        r.setdefault("lvl_tf", r["lvl"])
        r["yahoo"], r["date"] = s, df.index[-1]
        rows.append(r)
    t = pd.DataFrame(rows)
    # ตัดหุ้นที่ข้อมูล Yahoo ขาดหาย (แท่งล่าสุดเก่าเกิน 7 วัน หรือมีช่วงว่างเกิน 10 วันใน 5 เดือนล่าสุด) เพื่อไม่ให้สัญญาณผิด
    t["stale"] = (t["date"] < t["date"].max() - pd.Timedelta(days=7)) | (t["gap_days"] > 10)
    sec = load_sectors(market)
    t = t.merge(sec, on="yahoo", how="left")
    if market in ("US", "MY"):
        from sector_heatmap import market_caps
        t["mcap"] = t["yahoo"].map(market_caps(tuple(t["yahoo"])))
    else:
        t["mcap"] = t["price"] * t["shares"]
    t["mom_rank"] = t["mom"].where(~t["stale"]).rank(ascending=False, method="first")
    t["mom_n"] = int(t["mom_rank"].max()) if t["mom_rank"].notna().any() else 0
    if market == "MY":  # รายชื่อสั้น -> จัดอันดับเทียบหุ้น S&P 500 (ผลคำนวณล่วงหน้า) ให้ Top 20 / 20% บนสุด มีความหมายเหมือนหน้า S&P 500
        ref, _ = load_cache("US")
        if ref is not None and ref["mom"].notna().sum() > 100:
            rm = np.sort(ref.loc[~ref["stale"], "mom"].dropna().to_numpy())
            t["mom_rank"] = t["mom"].where(~t["stale"]).map(lambda m: np.nan if pd.isna(m) else
                                                           float(len(rm) - np.searchsorted(rm, m, side="right") + 1))
            t["mom_n"] = len(rm)
    t["symbol"] = t["yahoo"].str.replace(".BK", "", regex=False)
    t["sector_th"] = t["sector"].map(SECTOR_TH).fillna(t["sector"]).fillna("อื่นๆ")
    return t


# ======================= ผล backtest (รันแยกจาก scratch ด้วยฟังก์ชันชุดเดียวกัน) =======================
# วิธี: หุ้น SET100 รายชื่อปัจจุบัน 97 ตัว · ม.ค. 2013 – พ.ย. 2026 · ประเมินทุก 5 วันทำการ (54,874 ครั้ง) ด้วย evaluate() ชุดเดียวกับแท็บ
# วัด "ชนะตลาด" = ผลตอบแทนหุ้นนั้น 60 วันทำการถัดไป ลบค่าเฉลี่ยของทุกหุ้นในสัปดาห์เดียวกัน (หน่วย %)
BT_INFO = """
**ผลทดสอบย้อนหลัง: กฎเหล่านี้แยกหุ้นที่ดีกว่าตลาดได้จริงไหม**

หุ้น SET100 รายชื่อปัจจุบัน 97 ตัว · ม.ค. 2013 – พ.ย. 2026 · ประเมินสัปดาห์ละครั้ง ด้วยกติกาชุดเดียวกับแท็บนี้ ·
"ชนะตลาด" = ผลตอบแทน 3 เดือนถัดไปของหุ้น ลบค่าเฉลี่ยของหุ้นทุกตัวในสัปดาห์เดียวกัน

- **ผลอ่าน "ซื้อ" และ Spectrum ≥ +75** ชนะตลาดเฉลี่ยราว **+0.7 – 0.9% ต่อ 3 เดือน** แต่ชนะแค่ **45–46% ของครั้ง** (ฐาน 45%)
  แปลว่าได้เปรียบเล็กน้อยจากหุ้นบางตัวที่วิ่งแรง ไม่ใช่ชนะสม่ำเสมอ และ **ช่วง 2022–2026 แทบหายไป**
- **"ไม่ซื้อ" และ Spectrum ติดลบ** แพ้ตลาดราว −0.5% ต่อ 3 เดือน ค่อนข้างคงที่ทุกช่วง: ใช้ **คัดหุ้นอ่อนออก** ได้ดีกว่าใช้หาหุ้นเด่น
- **◆ กรอบม่วง** บวกในสองช่วงแรก แต่ **ติดลบ −1.7% ในช่วง 2022–2026** จึงไม่ยืนยันข้ออ้างว่าบวกทุกช่วง (ช่วงเวลาที่ต้นฉบับใช้อาจต่างกัน)
- **★ หุ้นน่าจับตา** แพ้ตลาดเล็กน้อย
- ข้อควรระวัง: ใช้รายชื่อ SET100 ปัจจุบัน (หุ้นที่หลุดดัชนีหรือเจ๊งไม่อยู่ในชุด) · ช่วงถือ 3 เดือนซ้อนทับกัน จำนวนครั้งจึงดูมากเกินจริง ·
  ยังไม่หักค่าคอมมิชชัน · ตีความบางเงื่อนไขเอง (ดูด้านล่าง)
"""
BT_TABLE = pd.DataFrame(
    [['ทุกหุ้น (ฐาน)', 54874, 0.0, 45.0, 0.0, 0.0, 0.0, 0.0], ['ผล: ซื้อ', 7472, 0.86, 46.0, 0.27, 1.09, 1.66, -0.04],
     ['ผล: ถือ', 13325, 0.43, 45.0, 0.14, 0.42, 0.92, 0.04], ['ผล: รอ', 8468, 0.04, 44.0, 0.15, -0.12, -0.82, 0.83],
     ['ผล: ไม่ซื้อ', 15069, -0.51, 44.0, -0.29, -0.56, -0.55, -0.46],
     ['ผล: ขาย / ลดสถานะ', 10540, -0.45, 45.0, -0.07, -0.87, -0.77, -0.01],
     ['◆ กรอบม่วง', 711, 0.48, 46.0, 0.51, 1.95, 0.95, -1.65], ['▣ Spectrum ≥ +75', 4443, 0.67, 45.0, 0.29, 1.28, 1.02, -0.23],
     ['★ น่าจับตา', 7607, -0.33, 42.0, 0.05, 0.06, 0.05, -0.87], ['Spectrum ≤ −40', 18849, -0.56, 44.0, -0.19, -0.86, -0.78, -0.2],
     ['Spectrum −40 ถึง −10', 7610, -0.4, 44.0, -0.23, -0.46, -0.21, -0.51],
     ['Spectrum −10 ถึง +10', 4542, -0.55, 43.0, -0.17, -0.41, -0.58, -0.64],
     ['Spectrum +10 ถึง +40', 7340, 0.58, 44.0, 0.3, -0.08, 0.1, 1.53], ['Spectrum ≥ +40', 16533, 0.71, 46.0, 0.24, 1.02, 1.21, 0.02]],
    columns=["กลุ่ม", "จำนวนครั้ง", "ชนะตลาด 3 เดือน", "% ครั้งที่ชนะ", "ชนะตลาด 1 เดือน", "ช่วง 2013–2017",
             "ช่วง 2018–2021", "ช่วง 2022–2026"])
BT_CFG = {c: st.column_config.NumberColumn(format="%+.2f%%") for c in
          ["ชนะตลาด 3 เดือน", "ชนะตลาด 1 เดือน", "ช่วง 2013–2017", "ช่วง 2018–2021", "ช่วง 2022–2026"]}
BT_CFG["% ครั้งที่ชนะ"] = st.column_config.NumberColumn(format="%.0f%%")


# ======================= UI =======================
def _spectrum_fig(t: pd.DataFrame):
    t = t.copy()
    med = t["mcap"].median()
    t["mc"] = t["mcap"].fillna(med if pd.notna(med) else 1.0)
    t["w"] = np.sqrt(t["mc"])
    t["w"] = t["w"] / t["w"].max()
    rows = []
    for sec, g in t.groupby("sector_th"):
        rows.append((sec, float(np.average(g["spec"], weights=g["mc"])), g))
    rows.sort(key=lambda r: r[1])  # แถวล่างสุดก่อน -> คะแนนสูงสุดอยู่บน
    X, base, Y, col, lc, lw, txt, hov, cd = [], [], [], [], [], [], [], [], []
    for sec, avg, g in rows:
        g = g.sort_values(["spec", "mc"])
        pos = 0.0
        for _, r in g.iterrows():
            X.append(r["w"])
            base.append(pos)
            pos += r["w"]
            Y.append(sec)
            col.append(sample_colorscale("RdYlGn", [(np.clip(r["spec"], -70, 70) + 70) / 140])[0])
            if r["purple"]:
                lc.append("#7b2cbf"), lw.append(4)
            elif r["white"]:
                lc.append("#111"), lw.append(3)
            else:
                lc.append("rgba(255,255,255,.9)"), lw.append(1)
            txt.append(("◆" if r["purple"] else "") + r["symbol"] + ("★" if r["watch"] else ""))
            hov.append(f"<b>{r['symbol']}</b> · {r.get('industry') or ''}<br>Spectrum {r['spec']:+d} · "
                       f"{BOX[r['box']][0]} {DIRS[r['dir']][0]} · {r['verdict']}")
            cd.append(r["yahoo"])
    f = go.Figure(go.Bar(x=X, base=base, y=Y, orientation="h", marker=dict(color=col, line=dict(color=lc, width=lw)),
                         text=txt, textposition="inside", insidetextanchor="middle", textfont=dict(size=11),
                         hovertext=hov, hoverinfo="text", customdata=cd, width=0.85))
    xmax = max(b + x for b, x in zip(base, X)) if X else 1
    for sec, avg, g in rows:
        f.add_annotation(x=xmax * 1.01, y=sec, text=f"<b>{avg:+.0f}</b>", showarrow=False, xanchor="left",
                         font=dict(color="#1b7837" if avg >= 0 else "#b2182b"))
    f.update_layout(height=60 + 46 * len(rows), margin=dict(l=10, r=40, t=10, b=10), showlegend=False,
                    xaxis=dict(visible=False, range=[0, xmax * 1.07]), bargap=0.1, uniformtext=dict(minsize=7, mode="hide"))
    return f


def _detail(t: pd.DataFrame, px: dict, sym: str, asof=None):
    r = t[t["yahoo"] == sym].iloc[0]
    if asof is not None:  # ดูย้อนหลัง: ตัดกราฟถึงวันที่เลือก
        px = {sym: px[sym][px[sym].index <= pd.Timestamp(asof)]}
    st.markdown(f"##### {r['symbol']} · {r['sector_th']} · {r.get('industry') or ''}")
    c = st.columns(5)
    c[0].metric("ผลอ่าน", VERDICT_ICON[r["verdict"]], f"คะแนนกฎ {r['score']:+d}", delta_color="off")
    c[1].metric("กล่องสี", BOX[r["box"]][0], BOX[r["box"]][1], delta_color="off")
    c[2].metric("ระดับ EMA", LEVELS[r["lvl"]], "⚠️ ห่าง EMA200 < 2%" if r["near200"] else None, delta_color="off")
    c[3].metric("ทิศทาง 29 วัน", f"{DIRS[r['dir']][0]} {DIRS[r['dir']][1]}",
                f"{r['dir_tot']:+.1%} · R² {r['dir_r2']:.2f}", delta_color="off")
    c[4].metric("Spectrum", f"{r['spec']:+d}", ("◆ กรอบม่วง " if r["purple"] else "") + ("กรอบเข้ม ≥75" if r["white"] else ""),
                delta_color="off")
    st.info(f"**สรุปรวม:** {summary_text(r['lvl'], len(r['sup']), len(r['warn']))}")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**สัญญาณหนุน**")
        items = []
        if r["bull"]:
            items.append("▲ แท่งเทียนขาขึ้น: " + ", ".join(r["bull"]))
        if r["macd_hh"] == "strong":
            items.append("MACD Higher High (เหนือศูนย์และเหนือ Signal)")
        elif r["macd_hh"] == "weak":
            items.append("MACD Higher High ระดับ *อ่อน* (MACD ต่ำกว่าศูนย์และ Signal · ไม่นับคะแนน)")
        if pd.notna(r["chan_days"]):
            items.append(f"ทะลุช่องขาลง ({int(r['chan_days'])} วันก่อน)")
        if "V + วอลุ่มหนุน" in r["sup"]:
            items.append(f"V + วอลุ่มหนุน (วอลุ่ม 5 วัน = {r['vratio']:.2f} เท่าของ 20 วัน)")
        if r["mv"]:
            items.append("MV: MACD ตัดขึ้นวันนี้ + วอลุ่ม ≥ 1.8 เท่า (ต้นฉบับระบุว่ายังไม่ทดสอบย้อนหลัง)")
        st.markdown("\n".join(f"- {x}" for x in items) or "- ไม่มี")
    with s2:
        st.markdown("**สัญญาณเตือน**")
        items = []
        if r["bear"]:
            items.append("▼ แท่งเทียนขาลง: " + ", ".join(r["bear"]))
        if r["smash"]:
            items.append("ขึ้นไปทำยอดแล้วโดนทุบ")
        st.markdown("\n".join(f"- {x}" for x in items) or "- ไม่มี")
    st.caption(f"คะแนนกฎ = ระดับ EMA {r['lvl']:+d} + ทิศทาง {DIRS[r['dir']][2]:+d} + หนุน {len(r['sup'])} − เตือน "
               f"{len(r['warn'])} = **{r['score']:+d}** · Spectrum = กล่อง {BOX[r['box']][3]:+d} · แท่งเทียน "
               f"{10 if r['bull'] else (-10 if r['bear'] else 0):+d} · ทิศทาง {DIRS[r['dir']][3]:+d} · รายเดือน "
               f"{MONTHLY[r['monthly']][0]} {MONTHLY[r['monthly']][1]:+d} · MACD {15 if r['macd_hh'] == 'strong' else 0:+d} · "
               f"ช่อง/ทุบ {(10 if pd.notna(r['chan_days']) else 0) - (10 if r['smash'] else 0):+d} · RSI {r['rsi']:.0f}")
    # กราฟสไตล์ TradingView (Lightweight Charts) + EMA 10/50/200 + ปักวันที่เกิดสัญญาณ
    # ส่งข้อมูล 400 แท่งเพื่อให้เส้น EMA ต่อเนื่อง แล้วให้กราฟเปิดที่ 180 แท่งล่าสุด (ซูมออกดูย้อนหลังได้)
    from tv_chart import lw_chart, signal_events, x7_panel, DEFAULT_MARKERS
    full = px[sym].iloc[-400:]
    ev = signal_events(sym, start=full.index[0], end=full.index[-1])
    b = ev[ev["key"].isin(["buy", "A"])] if len(ev) else ev
    xl, xm = x7_panel(full, f"x7_{sym}", b["date"].max() if len(b) else None)
    lw_chart(full, ma=(10, 50, 200), ma_kind="EMA", events=ev, keys=DEFAULT_MARKERS, height=420, visible_bars=180,
             extra_lines=xl, extra_marks=xm)
    st.caption("ซูมด้วยล้อเมาส์ · ลากเพื่อเลื่อน · 💪 เข้ากลุ่มแข็งแกร่ง · ซื้อ/ขาย = ผลอ่านเปลี่ยน · "
               "A/C = สูตรที่ทดสอบแล้วชนะตลาดใน 20 วัน")


HL_CSS = """
<style>
[class*="st-key-hl_"] button{min-height:0;padding:2px 10px;border-radius:999px;font-size:.85rem;line-height:1.5}
[class*="st-key-hl_"] button p{font-size:.85rem;font-variant-numeric:tabular-nums}
[class*="st-key-hl_strong_"] button{background:rgba(27,120,55,.12);border-color:rgba(27,120,55,.55)}
[class*="st-key-hl_watch_"] button{background:rgba(212,160,23,.14);border-color:rgba(212,160,23,.6)}
[class*="st-key-hl_purple_"] button{background:rgba(123,44,191,.12);border-color:rgba(123,44,191,.6)}
[class*="st-key-hl_star_"] button{background:rgba(42,120,214,.12);border-color:rgba(42,120,214,.55)}
[class*="st-key-hl_mom_"] button{background:rgba(230,85,13,.12);border-color:rgba(230,85,13,.6)}
[class*="st-key-hl_pass_"] button{background:rgba(27,120,55,.22);border-color:rgba(27,120,55,.9);font-weight:600}
</style>
"""
# กลุ่มเด่นตามเอกสาร SOMMIT: (คีย์, หัวข้อ, คำอธิบาย, ตัวกรอง)
HL_GROUPS = [
    ("strong", "💪 หุ้นแข็งแกร่ง", "กล่อง 🟢✓ / 🔵✓ = ราคาเหนือ EMA 10, 50 และ 200 · เรียงตามคะแนนกฎ",
     lambda t: t["box"].isin(["G", "B"])),
    ("watch", "🌱 หุ้นน่าติดตาม (กำลังฟื้น)", "กล่อง 👀 เด้งในแนวโน้มยาวขาขึ้น / 🟢↩ ฟื้นตัว ใกล้ทดสอบ EMA200",
     lambda t: t["box"].isin(["EYE", "RG"])),
    ("purple", "◆ กรอบม่วง", "แข็งแรง + ทิศทาง V + รายเดือนขาขึ้น + MACD Higher High (ผลทดสอบย้อนหลังไม่สม่ำเสมอ)",
     lambda t: t["purple"]),
    ("star", "★ หุ้นน่าจับตา", "↩ เขียว + แท่งเทียนขาขึ้น หรือ ทิศทาง V (ไม่ใช่ 👀 และไม่ใช่ ✕ ที่มีแท่งขาขึ้น)",
     lambda t: t["watch"]),
]


MOM_GROUP = ("mom", "🚀 โมเมนตัมสูงสุด 12 เดือน (Top 20)",
             "ผลตอบแทนรวมปันผล 12 เดือน ไม่นับเดือนล่าสุด · ผล backtest ดีกว่าสัญญาณ SOMMIT ทั้งหุ้นไทยและ S&P 500",
             lambda t: t["mom_rank"] <= 20)
# สูตรที่ backtest ได้ Sharpe ดีที่สุดทั้งไทยและสหรัฐ: เลือกด้วยโมเมนตัม แล้วใช้กฎ SOMMIT ตัดหุ้นที่ราคาหลุดเส้น EMA (กล่อง ✕) ออก
PASS_GROUP = ("pass", "✅ ผ่านเกณฑ์: โมเมนตัม Top 20 ที่ไม่อยู่ในกล่อง ✕",
              "สูตรที่ทดสอบย้อนหลังได้ Sharpe ดีที่สุด (SET100 1.07 เมื่อรวมตัวกรองตลาด · S&P 500 1.12) · ตัวเลข = ผลตอบแทน 12 เดือน",
              lambda t: (t["mom_rank"] <= 20) & (t["box"] != "X"))
ALL_GROUPS = [PASS_GROUP, MOM_GROUP] + HL_GROUPS
NEWBUY_MAX = 3  # กรอบสัญญาณซื้อใหม่: ขึ้นสัญญาณ "ซื้อ" ต่อเนื่องมาไม่เกินกี่วันทำการ
NEWBUY_GROUP = ("buy", f"🟢 สัญญาณซื้อใหม่ (ไม่เกิน {NEWBUY_MAX} วัน)",
                "ผลอ่านเพิ่งเปลี่ยนเป็น \"ซื้อ\" (จุดเขียวบนกราฟ) · ✅ = มีสัญญาณหนุน ≥ 2 ข้อ และโมเมนตัม 12 เดือนอยู่ 20% บนสุด "
                "(แบบที่ backtest SET100 ปี 2015–2026 ได้ผลดีที่สุด เมื่อขายตามกฎ X7) · S&P 500 สัญญาณซื้อยังแพ้ตลาด",
                lambda t: (t["verdict"] == "ซื้อ") & t["stk_buy"].between(1, NEWBUY_MAX))


@st.cache_data(ttl=3600, show_spinner=False)
def buy_streaks(items: tuple, market: str, maxd: int = 10) -> dict:
    """items = ((yahoo, วันที่ของแถวนั้น), ...) เฉพาะหุ้นที่ได้ "ซื้อ" · คืนจำนวนวันทำการที่ได้ "ซื้อ" ต่อเนื่องถึงวันนั้น
    ใช้วันที่ของแต่ละแถว เพราะบางหุ้นแถวล่าสุดอาจช้ากว่าตัวอื่น 1 วัน (Yahoo ส่งแท่งล่าสุดไม่ครบตอนคำนวณ)"""
    out = {}
    if not items:
        return out
    day_of = dict(items)
    for s, df in load_prices(tuple(day_of), period_of(market)).items():
        df = df[df.index <= pd.Timestamp(day_of[s])]
        if len(df) < 270:
            continue
        a = indicators(df)
        n, i = 0, len(df) - 1
        while n < maxd and evaluate(a, i - n)["verdict"] == "ซื้อ":
            n += 1
        out[s] = n
    return out
HL_LIMIT = 40  # จำนวนชิปสูงสุดต่อกล่อง (S&P 500 มีหุ้นเข้ากลุ่มได้เป็นร้อยตัว)


GROUP_SHORT = {"pass": "✅", "mom": "🚀", "strong": "💪", "watch": "🌱", "purple": "◆", "star": "★"}


def _days(n, cap):
    return f"{int(n)}+ วัน" if n >= cap else f"{int(n)} วัน"


def _highlights(t: pd.DataFrame, p: str, groups, members: dict = None, title: str = "⭐ หุ้นเด่นวันนี้",
                dropped: pd.DataFrame = None, cap: int = 40):
    """แผงหุ้นเด่นด้านบนแท็บ: กดชื่อหุ้นเพื่อดูเหตุผลและกราฟในช่องเหตุผลรายตัว
    members = {กลุ่ม: set(yahoo)} จากบันทึกรายวัน (ถ้ามี) แทนการคำนวณกฎใหม่
    คอลัมน์ stk_<กลุ่ม> = จำนวนวันที่ติดกลุ่มต่อเนื่อง · dropped = หุ้นที่เมื่อวานอยู่ในกลุ่มแต่วันนี้หลุด"""
    if members is not None:
        groups = [(k, ttl, d, rule if k == "buy" else (lambda t, k=k: t["yahoo"].isin(members.get(k, set()))))
                  for k, ttl, d, rule in groups]
    st.markdown(HL_CSS, unsafe_allow_html=True)
    st.markdown(f"#### {title} <span style='font-size:.8rem;color:gray;font-weight:400'>"
                "กดชื่อหุ้นเพื่อดูเหตุผลและกราฟด้านล่าง · ตัวเลข = คะแนนกฎ (🚀 = ผลตอบแทน 12 เดือน) · ไม่ใช่คำแนะนำซื้อขาย</span>",
                unsafe_allow_html=True)
    ss = st.session_state
    cols = st.columns(2)
    for n, (key, title, desc, rule) in enumerate(groups):
        g = t[rule(t)]
        if key == "buy":
            g = g.assign(_best=g["best_buy"]).sort_values(["_best", "stk_buy", "score"], ascending=[False, True, False])
        else:
            g = g.sort_values("mom_rank") if key in ("mom", "pass") else g.sort_values(["score", "spec"], ascending=False)
        with cols[n % 2].container(border=True):
            st.markdown(f"**{title}** · {len(g)} ตัว")
            st.caption(desc)
            if g.empty:
                st.caption("— ไม่มีวันนี้ —")
                continue
            sk = f"stk_{key}"
            if sk in g:
                new = g[g[sk] == 1]["symbol"].tolist()
                st.caption(("🆕 **เพิ่งเข้าวันนี้:** " + ", ".join(new)) if new else "🆕 เพิ่งเข้าวันนี้: –")
            with st.container(horizontal=True, key=f"hl_{key}_{p}", gap="small"):
                for _, r in g.head(HL_LIMIT).iterrows():
                    n = r.get(sk, 0) or 0
                    tag = ("🆕 " if n == 1 else "")
                    days = f" · {int(n)}{'+' if n >= cap else ''}ว" if n >= 1 else ""
                    if key == "buy":
                        lab = (f"{'✅ ' if r['best_buy'] else ''}{tag}{r['symbol']} · {int(n)}ว · หนุน {len(r['sup'])}")
                    elif key in ("mom", "pass"):
                        lab = f"{tag}{r['symbol']} {r['mom']:+.0%}{days}"
                    else:
                        lab = f"{tag}{r['symbol']} {r['score']:+d}{days}" + (" 🟢" if r["verdict"] == "ซื้อ" else "")
                    if st.button(lab, key=f"{p}_hlb_{key}_{r['symbol']}",
                                 help=f"{BOX[r['box']][0]} {BOX[r['box']][1]} · {DIRS[r['dir']][0]} {DIRS[r['dir']][1]} · "
                                      f"ผลอ่าน {r['verdict']} · Spectrum {r['spec']:+d} · % วันนี้ {r['chg1']:+.2%} · "
                                      f"โมเมนตัม 12 ด. {r['mom']:+.0%}"):
                        ss[f"{p}_pick"] = r["yahoo"]
                        st.rerun()
            if len(g) > HL_LIMIT:
                st.caption(f"แสดง {HL_LIMIT} อันดับแรกจาก {len(g)} ตัว · ดูทั้งหมดได้ด้วยตัวกรองของตารางด้านล่าง")
            if dropped is not None and key in dropped:
                out = [s.replace(".BK", "") for s in dropped.index[dropped[key]]]
                if out:
                    st.caption("⬇️ **หลุดจากกลุ่มวันนี้:** " + ", ".join(out[:30]) + (f" และอีก {len(out) - 30} ตัว" if len(out) > 30 else ""))
    st.caption("🆕 = เพิ่งเข้ากลุ่มวันนี้ · \"5ว\" = อยู่ในกลุ่มต่อเนื่อง 5 วันทำการ · 🟢 หลังชื่อ = ผลอ่าน \"ซื้อ\" · "
               "ผลทดสอบย้อนหลังแนะนำให้ใช้กลุ่มเหล่านี้เป็นตัวคัดกรองเบื้องต้น ไม่ใช่สัญญาณซื้อทันที")


def _streak_panel(h: pd.DataFrame, sym: str, cap: int):
    """สถานะการติดกลุ่มของหุ้นตัวที่เลือก: ติดต่อเนื่องกี่วัน ตั้งแต่เมื่อไร + ไทม์ไลน์ย้อนหลัง"""
    x = h[h["yahoo"] == sym].sort_values("date")
    if x.empty:
        return
    dates = list(x["date"])
    names = {"pass": "✅ ผ่านเกณฑ์", "mom": "🚀 โมเมนตัม Top 20", "strong": "💪 แข็งแกร่ง", "watch": "🌱 น่าติดตาม", "purple": "◆ กรอบม่วง",
             "star": "★ น่าจับตา"}
    lines = []
    for k in GKEYS:
        v = x[k].astype(bool).to_numpy()
        if v[-1]:
            n = len(v) - (np.where(~v)[0][-1] + 1 if (~v).any() else 0)
            since = dates[-n]
            lines.append(f"**{names[k]}**: ติดต่อเนื่อง **{_days(n, cap)}** (ตั้งแต่ {since:%d %b %Y})"
                         + (" · 🆕 เพิ่งเข้าวันนี้" if n == 1 else ""))
        elif v.any():
            last = dates[np.where(v)[0][-1]]
            lines.append(f"{names[k]}: ไม่ติดแล้ว · ติดครั้งล่าสุด {last:%d %b %Y}")
    st.markdown("**🗓️ ความต่อเนื่องของสัญญาณ** (" + f"{len(dates)} วันทำการล่าสุด)  \n"
                + ("  \n".join(lines) if lines else "ไม่เคยติดกลุ่มเด่นในช่วงนี้"))
    z = [x[k].astype(int).tolist() for k in GKEYS]
    f = go.Figure(go.Heatmap(z=z, x=dates, y=[names[k] for k in GKEYS], colorscale=[[0, "#eef0f3"], [1, "#2e8b57"]],
                             showscale=False, xgap=1, ygap=2, zmin=0, zmax=1,
                             hovertemplate="%{y}<br>%{x|%d %b %Y}<extra></extra>"))
    f.update_layout(height=190, margin=dict(l=10, r=10, t=5, b=5), yaxis=dict(autorange="reversed"))
    st.plotly_chart(f, use_container_width=True)


def _scroll_to(anchor_id: str):
    """เลื่อนหน้าไปที่ anchor (ใช้หลังคลิกหุ้นในตาราง/Spectrum ที่อยู่ด้านล่าง)"""
    import streamlit.components.v1 as components
    components.html(f"<script>setTimeout(()=>{{const e=window.parent.document.getElementById('{anchor_id}');"
                    f"if(e)e.scrollIntoView({{behavior:'smooth',block:'start'}})}},300)</script>", height=0)


def _picker_table(df: pd.DataFrame, key: str, p: str, **kw):
    """ตารางที่คลิกแถวแล้วเลือกหุ้นนั้นในช่องเหตุผลรายตัว (อยู่ในแท็บเดิม ไม่ย้ายไปแท็บกราฟ)"""
    ss = st.session_state
    nonce = ss.get(f"{p}_nonce", 0)
    cfg = dict(kw.pop("column_config", {}) or {})
    cfg["yahoo"] = None
    ev = st.dataframe(df, key=f"{key}_{nonce}", on_select="rerun", selection_mode="single-row",
                      hide_index=True, column_config=cfg, **kw)
    rows = ev.selection.rows if ev is not None else []
    if rows:
        ss[f"{p}_pick"] = df.iloc[rows[0]]["yahoo"]
        ss[f"{p}_scroll"] = True
        ss[f"{p}_nonce"] = nonce + 1
        st.rerun()


FILTERS = ["ทั้งหมด", "✅ ผ่านเกณฑ์ (โมเมนตัม + ไม่ ✕)", "🆕 เพิ่งเข้ากลุ่มเด่น (ไม่เกิน 3 วัน)", "🟢 สัญญาณซื้อใหม่ (ไม่เกิน 3 วัน)", "🚀 Top 20 โมเมนตัม", "💪 หุ้นแข็งแกร่ง", "🌱 หุ้นน่าติดตาม (กำลังฟื้น)", "◆ กรอบม่วง",
           "★ หุ้นน่าจับตา", "ซื้อ / ถือ", "บวกวันนี้", "ลบวันนี้"]
SORTS = ["คำแนะนำการลงทุน (กล่องสี)", "สถานะทางเทคนิค (คะแนนกฎ มาก→น้อย)", "ทิศทาง (▲ ก่อน)", "Spectrum (มาก→น้อย)",
         "โมเมนตัม 12 เดือน (มาก→น้อย)"]


MARKETS = {"TH": "🇹🇭 SET100", "US": "🇺🇸 S&P 500", "MY": "⭐ รายชื่อของฉัน"}


def render_signals_hub(set100: pd.DataFrame, us: pd.DataFrame):
    """แท็บเดียวรวม 2 ตลาด · เลือกตลาดก่อน แล้วค่อยคำนวณเฉพาะตลาดนั้น (ไม่โหลดหุ้น 600 ตัวพร้อมกัน)"""
    mk = st.segmented_control("ตลาด", list(MARKETS), format_func=MARKETS.get, default="TH", key="sig_market",
                              label_visibility="collapsed")
    mk = mk or "TH"
    if mk == "TH":
        render_signals_tab(set100, market="TH")
    elif mk == "MY":
        my = load_my_list()
        if my.empty:
            st.info("ยังไม่มีรายชื่อ · เพิ่มได้ในไฟล์ files/my_list.csv")
            return
        render_signals_tab(my, market="MY")
    else:
        render_signals_tab(us[us["sp500"] == 1], market="US")


def _date_picker(p: str, market: str):
    """📅 ดูสัญญาณ ณ วันที่ · คืน None เมื่อเป็นวันนี้ (ใช้ราคาล่าสุด)"""
    ss = st.session_state
    today = dt.date.today()
    lo = today - dt.timedelta(days=365 * (5 if market == "TH" else 3))
    log = load_log(market)
    logged = sorted(pd.to_datetime(log["date"]).dt.date.unique()) if len(log) else []
    if not isinstance(ss.get(f"{p}_asof"), dt.date):
        ss[f"{p}_asof"] = today
    c1, c2 = st.columns([1.3, 3])
    d = c1.date_input("📅 ดูสัญญาณ ณ วันที่", min_value=lo, max_value=today, key=f"{p}_asof", format="DD/MM/YYYY",
                      help="เลือกวันในอดีตเพื่อดูว่าวันนั้นหุ้นตัวไหนติดสัญญาณ · วันหยุดจะใช้ราคาปิดวันทำการก่อนหน้า")
    if logged:
        c2.caption(f"📒 มีบันทึกรายชื่อจริง {len(logged)} วัน ({logged[0]:%d/%m/%Y} – {logged[-1]:%d/%m/%Y}) · "
                   "วันอื่นจะคำนวณย้อนหลังจากราคาในอดีต")
    else:
        c2.caption("📒 ยังไม่มีบันทึกรายชื่อจริง (GitHub Actions จะเริ่มบันทึกทุกวันทำการ) · วันในอดีตจะคำนวณย้อนหลังจากราคา")
    return None if d >= today else d, log


def render_signals_tab(universe: pd.DataFrame, clickable_table=None, market: str = "TH"):
    """สัญญาณเทคนิคของตลาดเดียว · market="TH" (SET100) หรือ "US" (S&P 500) · คีย์ session แยกกันตามตลาด"""
    p = {"TH": "sig", "US": "usig", "MY": "msig"}[market]
    ss = st.session_state
    if f"{p}_pick" in ss:  # หุ้นที่คลิกมาจากแผงหุ้นเด่น/ตาราง/Spectrum -> ตั้งค่าก่อนสร้าง selectbox
        ss[f"{p}_detail"] = ss.pop(f"{p}_pick")
    syms = tuple(universe["yahoo"])
    star = dict(zip(universe["yahoo"], universe["set50"])) if "set50" in universe else {}
    asof, log = _date_picker(p, market)
    o1, o2, o3 = st.columns([1.2, 1.6, 1.4])
    tf = o1.radio("กล่องสีคำนวณจากแท่ง", ["D", "W", "M"], horizontal=True, key=f"{p}_tf",
                  format_func={"D": "รายวัน", "W": "1W", "M": "1M"}.get,
                  help="1W/1M เปลี่ยนเฉพาะกล่องสีในตาราง · คะแนนกฎและ Spectrum คำนวณจากแท่งรายวันเสมอ")
    sort = o2.selectbox("เรียงตาม", SORTS, key=f"{p}_sort", index=0 if market == "TH" else 4)
    flt = o3.selectbox("กรอง", FILTERS, key=f"{p}_flt")
    # ใช้ผลที่ GitHub Actions คำนวณไว้ทุกวันทำการ (เปิดได้ทันที) เมื่อดูวันล่าสุดด้วยแท่งรายวัน · ปิดสวิตช์ = คำนวณสดจากราคาล่าสุด
    h_cached, from_cache = None, False
    if asof is None and tf == "D":
        fast = st.toggle("⚡ ใช้ผลที่คำนวณไว้ล่วงหน้า (ราคาปิดล่าสุด · เปิดเร็ว)", value=True, key=f"{p}_fast",
                         help="ปิดสวิตช์เพื่อคำนวณสดจากราคาล่าสุดของ Yahoo (รวมราคาระหว่างวัน) · ใช้เวลานานกว่า")
        if fast:
            tc, h_cached = load_cache(market)
            if tc is not None:
                t, from_cache = tc, True
    if not from_cache:
        h_cached = None
        with st.spinner("กำลังคำนวณสัญญาณ ..."):
            t = signal_table(syms, tf, market, asof)
    if t.empty:
        st.warning("ไม่มีข้อมูลราคาพอสำหรับวันที่เลือก")
        return
    stale = t[t["stale"]]
    if len(stale):
        last = t["date"].max()
        st.warning("ไม่แสดง " + ", ".join(
            f"{r.symbol} (" + (f"ไม่มีราคาตั้งแต่ {r.date:%d %b %Y}" if r.date < last - pd.Timedelta(days=7)
                               else f"ข้อมูลขาดช่วง {r.gap_days} วัน") + ")" for r in stale.itertuples())
            + " เพราะราคาย้อนหลังจาก Yahoo ไม่ครบ สัญญาณจะคลาดเคลื่อน")
        t = t[~t["stale"]]
    day = t["date"].max()
    src = "ราคาจริง ไม่ปรับปันผล เหมือน set.or.th" if market == "TH" else "ราคาจริง ไม่ปรับปันผล"
    # บันทึกจริงของวันนั้น (ถ้ามี) -> ใช้รายชื่อหุ้นเด่นจากบันทึก ไม่คำนวณกฎใหม่
    L = log[log["date"] == f"{day:%Y-%m-%d}"] if len(log) else log
    members = None
    if len(L):
        members = {}
        for y, g in zip(L["yahoo"], L["groups"].fillna("")):
            for k in str(g).split("|"):
                if k:
                    members.setdefault(k, set()).add(y)
    if asof is None:
        if from_cache:
            up = cache_updated_at()
            how = ("⚡ ผลคำนวณล่วงหน้าจากราคาปิด · " + (f"อัปเดตล่าสุด {up:%d/%m/%Y %H:%M} น." if up is not None
                                                         else "อัปเดตทุกวันทำการ") + " (ปกติช่วงสายราว 10:00–11:30 น.)")
        else:
            how = f"คำนวณสดจาก Yahoo Finance ({src})"
        st.caption(f"ข้อมูลถึง {day:%d %b %Y} · {len(t)} หุ้น · {how} · "
                   "คลิกชื่อหุ้น แถวในตาราง หรือช่องใน Spectrum เพื่อดูเหตุผลและกราฟในช่อง 🔍 เหตุผลรายตัว")
    elif members is not None:
        st.success(f"📒 **ย้อนดูวันที่ {day:%d %b %Y}** · แผงหุ้นเด่นแสดงจาก **บันทึกจริงของวันนั้น** · "
                   "ตาราง กราฟ และเหตุผลรายตัว คำนวณย้อนหลังจากราคา ณ วันนั้น")
    else:
        st.info(f"🧮 **ย้อนดูวันที่ {day:%d %b %Y}** (คำนวณย้อนหลังจากราคาปิด ณ วันนั้น · ไม่มีบันทึกจริงของวันนี้) · "
                "อาจต่างจากที่เคยแสดงจริงเล็กน้อย เพราะใช้รายชื่อหุ้นในดัชนีปัจจุบัน")
    if market == "MY":
        short = [y for y in syms if y not in set(t["yahoo"])]
        st.info("⭐ **รายชื่อของฉัน** (แก้ได้ในไฟล์ files/my_list.csv) · อันดับโมเมนตัม = อันดับเมื่อเทียบหุ้น S&P 500 ทั้งหมด · "
                "ETF ทวีคูณ/กลับทิศ (เช่น SQQQ, SVIX, RAM) และดัชนี ^VIX เคลื่อนไหวต่างจากหุ้นปกติ "
                "กฎสัญญาณและผล backtest ทำมาจากหุ้นรายตัว จึงใช้กับกลุ่มนี้ได้จำกัด"
                + (f" · ยังคำนวณไม่ได้ (ประวัติราคาไม่ถึง 230 วัน): {', '.join(short)}" if short else ""))
    if market == "US":
        st.warning("⚠️ ผล backtest ปี 2014–2026: กฎแบบ SOMMIT **ใช้กับหุ้น S&P 500 ไม่ได้ผล** (หุ้นที่ได้ \"ซื้อ\" แพ้ตลาด, "
                   "\"ขาย\" กลับชนะตลาด) · ตัวที่ได้ผลดีกว่าคือ 🚀 โมเมนตัม 12 เดือน · รายละเอียดในแท็บย่อย 🧪 ทดสอบย้อนหลัง")

    vc = t["verdict"].value_counts()
    m = st.columns(7)
    for col, v in zip(m, VERDICTS):
        col.metric(VERDICT_ICON[v], int(vc.get(v, 0)))
    m[5].metric("◆ กรอบม่วง", int(t["purple"].sum()))
    m[6].metric("★ น่าจับตา", int(t["watch"].sum()))

    # ----- ความต่อเนื่อง: ติดกลุ่มมากี่วัน / เพิ่งเข้า / หลุด (คำนวณย้อนหลัง n วันทำการ) -----
    cap = {"TH": 60, "US": 40, "MY": 40}[market]
    if h_cached is not None:
        h = h_cached
    else:
        with st.spinner("กำลังนับวันที่หุ้นติดกลุ่มต่อเนื่อง ..."):
            h = membership_history(syms, market, asof, cap)
    dropped = None
    if len(h):
        stk, dropped, _ = streak_table(h)
        t = t.merge(stk.add_prefix("stk_"), left_on="yahoo", right_index=True, how="left")
        for k in GKEYS:
            t[f"stk_{k}"] = t[f"stk_{k}"].fillna(0).astype(int)
    else:
        for k in GKEYS:
            t[f"stk_{k}"] = 0

    # สัญญาณซื้อใหม่: นับวันที่ได้ "ซื้อ" ต่อเนื่อง จากราคาจริง (เฉพาะหุ้นที่วันนี้ได้ "ซื้อ")
    bt_ = t[t["verdict"] == "ซื้อ"].sort_values("yahoo")
    with st.spinner("กำลังนับวันของสัญญาณซื้อ ..."):
        bs = buy_streaks(tuple((y, f"{d:%Y-%m-%d}") for y, d in zip(bt_["yahoo"], bt_["date"])), market)
    t["stk_buy"] = t["yahoo"].map(bs).fillna(0).astype(int)
    nmom = t["mom_n"].max() if "mom_n" in t else t["mom_rank"].max()
    t["best_buy"] = (t["sup"].map(len) >= 2) & (t["mom_rank"] <= max(1, round(nmom * 0.2)))
    groups = [NEWBUY_GROUP] + ALL_GROUPS
    _highlights(t, p, groups, members,
                "⭐ หุ้นเด่นวันนี้" if asof is None else f"⭐ หุ้นเด่น ณ {day:%d %b %Y}", dropped, cap)

    # ----- เหตุผลรายตัว + กราฟ (ต่อจากหุ้นเด่น) -----
    st.markdown(f"<div id='{p}-detail'></div>", unsafe_allow_html=True)
    st.markdown("#### 🔍 เหตุผลรายตัว" + ("" if asof is None else f" ณ {day:%d %b %Y}"))
    opts = sorted(t["yahoo"])
    names = dict(zip(t["yahoo"], t["industry"])) if market in ("US", "MY") else {}
    if ss.get(f"{p}_detail") not in opts:
        hl = t.sort_values("mom_rank") if market in ("US", "MY") else \
            t[t["box"].isin(["G", "B"])].sort_values(["score", "spec"], ascending=False)
        ss[f"{p}_detail"] = hl["yahoo"].iloc[0] if len(hl) else opts[0]
    d1, d2 = st.columns([2, 1])
    sym = d1.selectbox("เลือกหุ้น (หรือคลิกชื่อหุ้นจากแผงด้านบน / ตาราง / Spectrum)", opts, key=f"{p}_detail",
                       format_func=lambda s: s.replace(".BK", "") + (f" — {names[s]}" if s in names else ""))
    d2.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    if d2.button("📈 เปิดในแท็บกราฟราคา (เต็มจอ)", key=f"{p}_open_chart"):
        ss["goto_sym"] = sym
        st.rerun()
    with st.container(border=True):
        r0 = t[t["yahoo"] == sym].iloc[0]
        if pd.notna(r0["mom"]):
            st.caption(f"🚀 โมเมนตัม 12 เดือน: {r0['mom']:+.1%} · อันดับ {int(r0['mom_rank'])} จาก {int(t['mom_rank'].max())}")
        # ใช้ผลคำนวณล่วงหน้า -> ดึงราคาเฉพาะหุ้นตัวที่เลือก (ไม่ต้องโหลดทั้งตลาด)
        _detail(t, load_prices((sym,) if from_cache else syms, period_of(market)), sym, asof)
        if len(h):
            _streak_panel(h, sym, cap)
        _log_history(log, sym)
    if ss.pop(f"{p}_scroll", False):
        _scroll_to(f"{p}-detail")

    tab_a, tab_b, tab_c, tab_d = st.tabs(["📋 ตารางสัญญาณรายตัว", "🌈 Market Lens: Signal Spectrum",
                                          "🧪 ทดสอบย้อนหลัง & วิธีคิด", "📒 ผลจริงจากบันทึก"],
                                         key=f"{p}_subtabs", on_change="rerun")
    with tab_d:
        if getattr(tab_d, "open", True):  # โหลดราคาทั้งตลาดเฉพาะตอนเปิดแท็บนี้
            _render_log_results(market, syms)
    with tab_a:
        v = t.copy()
        if flt == "บวกวันนี้":
            v = v[v["chg1"] > 0]
        elif flt == "ลบวันนี้":
            v = v[v["chg1"] < 0]
        elif flt == "◆ กรอบม่วง":
            v = v[v["purple"]]
        elif flt == "★ หุ้นน่าจับตา":
            v = v[v["watch"]]
        elif flt == "ซื้อ / ถือ":
            v = v[v["verdict"].isin(["ซื้อ", "ถือ"])]
        elif flt.startswith("💪"):
            v = v[v["box"].isin(["G", "B"])]
        elif flt.startswith("🌱"):
            v = v[v["box"].isin(["EYE", "RG"])]
        elif flt.startswith("🚀"):
            v = v[v["mom_rank"] <= 20]
        elif flt.startswith("✅"):
            v = v[(v["mom_rank"] <= 20) & (v["box"] != "X")]
        elif flt.startswith("🟢"):
            v = v[(v["verdict"] == "ซื้อ") & v["stk_buy"].between(1, NEWBUY_MAX)]
        elif flt.startswith("🆕"):
            s = v[[f"stk_{k}" for k in GKEYS]]
            v = v[((s >= 1) & (s <= 3)).any(axis=1)]
        if sort.startswith("คำแนะนำ"):
            v = v.assign(_k=v["box_tf"].map(lambda b: BOX[b][2])).sort_values(["_k", "symbol"])
        elif sort.startswith("สถานะ"):
            v = v.sort_values(["score", "spec"], ascending=False)
        elif sort.startswith("ทิศทาง"):
            v = v.assign(_k=v["dir"].map(lambda d: DIRS[d][4])).sort_values(["_k", "dir_tot"], ascending=[True, False])
        elif sort.startswith("โมเมนตัม"):
            v = v.sort_values("mom_rank")
        else:
            v = v.sort_values("spec", ascending=False)
        tbl = pd.DataFrame({
            "yahoo": v["yahoo"],
            "หุ้น": v["symbol"] + v["yahoo"].map(lambda s: " ★" if star.get(s) == 1 else ""),
            "กลุ่ม": v["sector_th"],
            "ราคา": v["price"],
            "% วันนี้": v["chg1"] * 100,
            "โมเมนตัม 12 ด.": v["mom"] * 100,
            "กล่องสี": v["box_tf"].map(lambda b: BOX[b][0]),
            "ระดับ EMA": v["lvl_tf"].map(LEVELS) + v["near200"].map(lambda x: " ⚠️" if x else ""),
            "ทิศทาง": v["dir"].map(lambda d: f"{DIRS[d][0]} {DIRS[d][1]}"),
            "สัญญาณหนุน": v["sup"].map(lambda x: " · ".join(x)),
            "สัญญาณเตือน": v["warn"].map(lambda x: " · ".join(x)),
            "คะแนน": v["score"],
            "ผลอ่าน": v["verdict"].map(VERDICT_ICON),
            "Spectrum": v["spec"],
            "พิเศษ": (((v["mom_rank"] <= 20) & (v["box"] != "X")).map(lambda x: "✅ " if x else "")
                      + v["mom_rank"].map(lambda x: "🚀 " if x <= 20 else "") + v["purple"].map(lambda x: "◆ " if x else "")
                      + v["white"].map(lambda x: "▣ " if x else "") + v["watch"].map(lambda x: "★" if x else "")),
            "ติดกลุ่มต่อเนื่อง": [" ".join(("🆕" if r[f"stk_{k}"] == 1 else "") + f"{GROUP_SHORT[k]}{r[f'stk_{k}']}ว"
                                           for k in GKEYS if r[f"stk_{k}"] > 0) for _, r in v.iterrows()],
        })
        if market in ("US", "MY"):
            tbl.insert(2, "บริษัท", v["industry"])
        _picker_table(tbl, f"{p}_tbl", p, use_container_width=True, height=560, column_config={
            "ราคา": st.column_config.NumberColumn(format="%.2f"),
            "% วันนี้": st.column_config.NumberColumn(format="%+.2f%%"),
            "โมเมนตัม 12 ด.": st.column_config.NumberColumn(format="%+.0f%%"),
            "คะแนน": st.column_config.NumberColumn(format="%+d"),
            "Spectrum": st.column_config.ProgressColumn(min_value=-100, max_value=100, format="%+d"),
        })
        st.caption("คลิกแถวเพื่อดูเหตุผลและกราฟในช่อง 🔍 เหตุผลรายตัวด้านบน · "
                   + ("★ หลังชื่อ = SET50 · " if market == "TH" else "")
                   + "⚠️ = ราคาห่าง EMA200 ไม่ถึง 2% · พิเศษ: 🚀 Top 20 โมเมนตัม · ◆ กรอบม่วง · ▣ Spectrum ≥ +75 · ★ หุ้นน่าจับตา")

    with tab_b:
        st.caption("แต่ละแถว = กลุ่มอุตสาหกรรม เรียงจากคะแนนเฉลี่ยถ่วง market cap สูงสุด (ตัวเลขท้ายแถว) · "
                   "ในแถวเรียงจากคะแนนต่ำ (ซ้าย แดง) ไปสูง (ขวา เขียว) · ความกว้าง ∝ √market cap · สีเข้มสุดที่ ±70 · "
                   "กรอบม่วงหนา ◆ = กรอบม่วง · กรอบดำ = Spectrum ≥ +75 (ต้นฉบับใช้กรอบขาว) · ★ = หุ้นน่าจับตา")
        nonce = ss.get(f"{p}_spec_nonce", 0)
        ev = st.plotly_chart(_spectrum_fig(t), use_container_width=True, on_select="rerun",
                             selection_mode="points", key=f"{p}_spec_{nonce}")
        pts = ev.selection.points if ev is not None and ev.selection else []
        if pts and pts[0].get("customdata"):
            cdv = pts[0]["customdata"]
            ss[f"{p}_pick"] = cdv[0] if isinstance(cdv, (list, tuple)) else cdv
            ss[f"{p}_scroll"] = True
            ss[f"{p}_spec_nonce"] = nonce + 1
            st.rerun()
        top = t.sort_values("spec", ascending=False)
        a1, a2 = st.columns(2)
        a1.markdown("**◆ กรอบม่วง:** " + (", ".join(top[top["purple"]]["symbol"]) or "ไม่มีวันนี้"))
        a1.markdown("**▣ Spectrum ≥ +75:** " + (", ".join(top[top["white"]]["symbol"]) or "ไม่มีวันนี้"))
        a2.markdown("**★ หุ้นน่าจับตา:** " + (", ".join(top[top["watch"]]["symbol"]) or "ไม่มีวันนี้"))

    with tab_c:
        if market in ("US", "MY"):
            st.markdown(BT_US)
        else:
            if BT_INFO:
                st.markdown(BT_INFO)
            if not BT_TABLE.empty:
                st.dataframe(BT_TABLE, hide_index=True, use_container_width=True, column_config=BT_CFG)
                st.caption("ช่วงปี = ชนะตลาด 3 เดือนเฉลี่ยในช่วงนั้น · ⚠️ ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำซื้อขาย")
        with st.expander("📘 กติกาทั้ง 4 ชั้น และจุดที่ตีความเพิ่ม"):
            st.markdown(RULES_MD)


# ผล backtest S&P 500 (รันออฟไลน์ด้วย evaluate() ชุดเดียวกัน · หุ้นในดัชนีปัจจุบัน 498 ตัว · ประเมินทุกสิ้นเดือน ม.ค. 2014 – ต.ค. 2026)
BT_US = """
**ผลทดสอบย้อนหลังกับหุ้น S&P 500: กฎแบบ SOMMIT ใช้ได้ไหม**

หุ้นในดัชนี S&P 500 ปัจจุบัน 498 ตัว · ประเมินทุกสิ้นเดือน ม.ค. 2014 – ต.ค. 2026 · ปรับพอร์ตทุกเดือน · หักค่าซื้อขายฝั่งละ 0.1%

| กลยุทธ์ | ผลตอบแทน/ปี | ขาดทุนสูงสุด | Sharpe | 2014–17 | 2018–21 | 2022–26 |
|---|---|---|---|---|---|---|
| SPY (ถ่วงตามขนาด) | 13.1% | −24% | 0.93 | 14.4% | 14.4% | 12.0% |
| ถือทุกตัวเท่ากัน (ฐานเทียบ) | 16.2% | −24% | 1.05 | 18.6% | 17.6% | 13.8% |
| ตัด "ไม่ซื้อ/ขาย" ออก | 14.3% | −22% | 1.00 | 16.3% | 16.1% | 11.8% |
| 💪 หุ้นแข็งแกร่ง (กล่อง ✓) | 12.7% | −18% | 0.94 | 13.5% | 14.8% | 11.1% |
| Top 50 Spectrum | 10.6% | −20% | 0.75 | 9.0% | 13.7% | 10.0% |
| ◆ กรอบม่วงทุกตัว | 10.1% | −26% | 0.56 | 12.4% | 6.3% | 12.4% |
| **Top 50 โมเมนตัม 12 เดือน** | **22.1%** | −21% | 1.06 | 17.5% | 19.4% | 28.2% |
| โมเมนตัม + ไม่เอากล่อง ✕ | 21.5% | −21% | **1.12** | 17.4% | 20.9% | 24.9% |

| กลุ่ม (ชนะค่าเฉลี่ยทุกหุ้นใน 3 เดือนถัดไป) | ชนะตลาด 3 เดือน | % ครั้งที่ชนะ (ฐาน 48%) |
|---|---|---|
| ผลอ่าน "ซื้อ" | −0.23% | 46% |
| ผลอ่าน "ขาย / ลดสถานะ" | +0.58% | 50% |
| Spectrum ≥ +40 | −0.17% | 47% |
| Spectrum ≤ −40 | +0.21% | 49% |
| 🌱 หุ้นน่าติดตาม | −0.49% | 46% |

- **สัญญาณ SOMMIT กลับทางในหุ้นใหญ่สหรัฐ:** หุ้นที่ได้ "ขาย" กลับชนะตลาด หุ้นที่ได้ "ซื้อ" แพ้ตลาด (หุ้นใหญ่สหรัฐมักเด้งกลับในระยะสั้น)
- **ทุกพอร์ตที่ใช้สัญญาณ SOMMIT แพ้การถือทุกตัวเท่ากัน** ข้อดีเดียวคือกลุ่มหุ้นแข็งแกร่งขาดทุนสูงสุดน้อยกว่า
- **🚀 โมเมนตัม 12 เดือนดีที่สุด** เหมือนผลกับหุ้นไทย · ตัวกรองตลาดขาขึ้นใช้ไม่ได้กับสหรัฐ (ผลตอบแทนหายเกือบครึ่ง)
- ⚠️ ใช้รายชื่อหุ้นในดัชนีปัจจุบัน หุ้นที่ขึ้นแรงจนถูกเพิ่มเข้าดัชนีภายหลังอยู่ในชุดตั้งแต่ปี 2014 ตัวเลขโมเมนตัมจึงสูงเกินจริง ·
  ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน
"""


RULES_MD = """
**ชั้น 1 กล่องสี (EMA 10/50/200 + RSI★)** · RSI★ = RSI(14) > 50 และลดลงไม่เกิน 5% จากเมื่อวาน
- 🟢✓ เหนือทุกเส้น + RSI★ · 🔵✓ เหนือทุกเส้น · 🟡✓ ต่ำกว่า EMA10 แต่เหนือ 50/200 · 👀 เหนือ 10/200 ต่ำกว่า 50
- 🟢↩ เหนือ 10/50 ต่ำกว่า 200 + RSI★ · 🟡↩ เหนือ 10 ต่ำกว่า 50/200 + RSI★ · 🔴✕ ต่ำกว่า 10 และ (50 หรือ 200)
- ระดับ: บวก +2 · กลาง-บวก +1 · กลาง 0 · ระวัง −1 · ลบ −2

**ชั้น 2 ทิศทาง 29 วัน** (regression ของ log ราคา) เช็กตามลำดับ Λ → V → ↗ → ▲ → ▼ → ↔

**ชั้น 3 สัญญาณ** · หนุน: แท่งเทียนขาขึ้น, MACD Higher High, ทะลุช่องขาลง, V + วอลุ่ม, MV · เตือน: แท่งเทียนขาลง, ขึ้นทำยอดแล้วโดนทุบ
- คะแนนกฎ = ระดับ EMA + ทิศทาง (▲↗V +1, Λ▼ −1) + จำนวนหนุน − จำนวนเตือน
- ผลอ่าน (เช็กตามลำดับ): ขาย = ระดับลบ + ทิศ ▼/Λ + คะแนน ≤ −3 · ไม่ซื้อ = คะแนน ≤ −1 · ซื้อ = คะแนน ≥ 4 + ระดับบวก + ไม่มีเตือน ·
  ถือ = ระดับ ≥ กลาง-บวก + คะแนน ≥ 2 · ที่เหลือ = รอ

**ชั้น 4 Signal Spectrum** = กล่อง (+40/+25/+5/−35) + แท่งเทียน (±10) + ทิศทาง (V +15, ↗ +8, ▲ +5, Λ −12, ▼ −20)
+ รูปทรงรายเดือน (±15/±10) + MACD HH +15 + ทะลุช่อง +10 / โดนทุบ −10 · ตัดให้อยู่ใน ±100
- ◆ กรอบม่วง = กล่อง 🟢✓/🔵✓ + ทิศทาง V + รายเดือนขาขึ้น + MACD HH · ★ น่าจับตา = (🟢↩ + แท่งขาขึ้น) หรือ (V ที่ไม่ใช่ 👀 และไม่ใช่ ✕ ที่มีแท่งขาขึ้น)

**จุดที่เอกสารไม่ได้นิยาม และวิธีที่ใช้**
- ยอด MACD = จุดสูงสุดเฉพาะที่ใน 22 วัน (วันล่าสุดนับเป็นยอดได้ถ้ายังขึ้นอยู่) · ระดับ "อ่อน" แสดงไว้แต่ไม่นับคะแนน
- ช่องขาลง = 70 วัน ที่จบก่อนวันนี้ 10–20 วัน (ลองทุกตำแหน่ง ผ่านตำแหน่งใดก็นับ) · ขอบบน = เส้นแนวโน้ม + ส่วนเบี่ยงสูงสุด ·
  ยอดต่ำลง = แบ่ง 4 ช่วง ต่ำลง ≥ 2 ใน 3 ครั้ง · ทะลุ = ราคาปิดเหนือขอบบนที่ลากต่อ หลังจบช่อง ·
  เทียบรายชื่อ SOMMIT วันที่ 12 พ.ย. 2569: ตรงกัน 20 จาก 23 ตัว
- รูปทรงรายเดือน (ตามโค้ด SOMMIT): เช็กกลับตัวก่อน · "ก่อนหน้านั้นเป็นช่วงลง" = 5 เดือนแรกลงสุทธิ · เดือนปัจจุบันใช้ราคาล่าสุด

**ปรับให้ตรงโค้ดจริงของ SOMMIT (เอกสารไม่ได้เขียนไว้)**
- 🟡✓ พักตัวในขาขึ้น ต้อง *ไม่* ผ่าน RSI★ (ถ้าผ่าน = ไม่เข้ากล่อง)
- ยอด MACD ต้องสูงสุดในช่วง ±3 วัน · RSI(14) คิดใหม่จาก 260 วันล่าสุด
- ใช้ราคาจริงไม่ปรับปันผล (เหมือน set.or.th) · backtest คิดผลตอบแทนจากราคาปรับปันผล
- เทียบทั้ง 100 หุ้น วันที่ 13 พ.ย. 2569: ผลอ่านตรง 93 · กล่องสี 93 · ทิศทาง 99 · รายเดือน 99 · สัญญาณหนุน/เตือน 98 ·
  Spectrum ห่างไม่เกิน 10 คะแนน 97 ตัว (ที่ต่างคือหุ้นที่ราคาใกล้เส้น EMA มาก และ BANPU ที่ Yahoo หยุดอัปเดตราคา)
- กลุ่มในแถว Spectrum ใช้ sector ของ Yahoo (11 กลุ่ม) แทนกลุ่มธุรกิจย่อยของ SET
- ไม่มีในแท็บนี้ (ไม่มีข้อมูลฟรี): ราคาเป้าหมายรายโบรก / Auto Scan, Value View (EPS IAA), ข่าว, CLAUDE EPS
"""
