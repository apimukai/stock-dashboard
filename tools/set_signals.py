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
@st.cache_data(ttl=3600, show_spinner="กำลังดึงราคาย้อนหลัง SET100 (ครั้งแรกอาจใช้เวลาราว 1 นาที) ...")
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


def load_sectors() -> pd.DataFrame:
    p = os.path.join(FILES, "set100_sectors.csv")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["yahoo", "sector", "industry", "shares"])
    return pd.read_csv(p)[["yahoo", "sector", "industry", "shares"]]


SECTOR_TH = {"Financial Services": "การเงิน", "Consumer Cyclical": "สินค้าฟุ่มเฟือย/บริการ", "Industrials": "อุตสาหกรรม",
             "Utilities": "สาธารณูปโภค", "Consumer Defensive": "สินค้าจำเป็น", "Real Estate": "อสังหาริมทรัพย์",
             "Energy": "พลังงาน", "Healthcare": "การแพทย์", "Basic Materials": "วัตถุดิบ",
             "Communication Services": "สื่อสาร", "Technology": "เทคโนโลยี"}


@st.cache_data(ttl=3600, show_spinner=False)
def signal_table(syms: tuple, tf: str = "D") -> pd.DataFrame:
    px = load_prices(syms)
    rows = []
    for s, df in px.items():
        a = indicators(df)
        r = evaluate(a, len(df) - 1)
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
    sec = load_sectors()
    t = t.merge(sec, on="yahoo", how="left")
    t["mcap"] = t["price"] * t["shares"]
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


def _detail(t: pd.DataFrame, px: dict, sym: str):
    r = t[t["yahoo"] == sym].iloc[0]
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
    df = px[sym].iloc[-180:]
    f = go.Figure(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name=r["symbol"], showlegend=False))
    full = px[sym]["Close"]
    for n, colr in [(10, "#f28e2b"), (50, "#2a78d6"), (200, "#7b2cbf")]:
        e = full.ewm(span=n, adjust=False).mean().iloc[-180:]
        f.add_trace(go.Scatter(x=e.index, y=e, name=f"EMA{n}", line=dict(color=colr, width=1.3)))
    f.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", y=1.05))
    st.plotly_chart(f, use_container_width=True)


HL_CSS = """
<style>
[class*="st-key-hl_"] button{min-height:0;padding:2px 10px;border-radius:999px;font-size:.85rem;line-height:1.5}
[class*="st-key-hl_"] button p{font-size:.85rem;font-variant-numeric:tabular-nums}
.st-key-hl_strong button{background:rgba(27,120,55,.12);border-color:rgba(27,120,55,.55)}
.st-key-hl_watch button{background:rgba(212,160,23,.14);border-color:rgba(212,160,23,.6)}
.st-key-hl_purple button{background:rgba(123,44,191,.12);border-color:rgba(123,44,191,.6)}
.st-key-hl_star button{background:rgba(42,120,214,.12);border-color:rgba(42,120,214,.55)}
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


def _highlights(t: pd.DataFrame):
    """แผงหุ้นเด่นด้านบนแท็บ: กดชื่อหุ้นเพื่อดูเหตุผลและกราฟในช่องเหตุผลรายตัว"""
    st.markdown(HL_CSS, unsafe_allow_html=True)
    st.markdown("#### ⭐ หุ้นเด่นวันนี้ <span style='font-size:.8rem;color:gray;font-weight:400'>"
                "กดชื่อหุ้นเพื่อดูเหตุผลและกราฟด้านล่าง · ตัวเลข = คะแนนกฎ · ไม่ใช่คำแนะนำซื้อขาย</span>",
                unsafe_allow_html=True)
    ss = st.session_state
    cols = st.columns(2)
    for n, (key, title, desc, rule) in enumerate(HL_GROUPS):
        g = t[rule(t)].sort_values(["score", "spec"], ascending=False)
        with cols[n % 2].container(border=True):
            st.markdown(f"**{title}** · {len(g)} ตัว")
            st.caption(desc)
            if g.empty:
                st.caption("— ไม่มีวันนี้ —")
                continue
            with st.container(horizontal=True, key=f"hl_{key}", gap="small"):
                for _, r in g.iterrows():
                    lab = f"{r['symbol']} {r['score']:+d}" + (" 🟢" if r["verdict"] == "ซื้อ" else "")
                    if st.button(lab, key=f"hlb_{key}_{r['symbol']}",
                                 help=f"{BOX[r['box']][0]} {BOX[r['box']][1]} · {DIRS[r['dir']][0]} {DIRS[r['dir']][1]} · "
                                      f"ผลอ่าน {r['verdict']} · Spectrum {r['spec']:+d} · % วันนี้ {r['chg1']:+.2%}"):
                        ss["sig_pick"] = r["yahoo"]
                        st.rerun()
    st.caption("🟢 หลังชื่อ = ผลอ่าน \"ซื้อ\" · ผลทดสอบย้อนหลังแนะนำให้ใช้กลุ่มเหล่านี้เป็นตัวคัดกรองเบื้องต้น ไม่ใช่สัญญาณซื้อทันที")


def _scroll_to(anchor_id: str):
    """เลื่อนหน้าไปที่ anchor (ใช้หลังคลิกหุ้นในตาราง/Spectrum ที่อยู่ด้านล่าง)"""
    import streamlit.components.v1 as components
    components.html(f"<script>setTimeout(()=>{{const e=window.parent.document.getElementById('{anchor_id}');"
                    f"if(e)e.scrollIntoView({{behavior:'smooth',block:'start'}})}},300)</script>", height=0)


def _picker_table(df: pd.DataFrame, key: str, **kw):
    """ตารางที่คลิกแถวแล้วเลือกหุ้นนั้นในช่องเหตุผลรายตัว (อยู่ในแท็บเดิม ไม่ย้ายไปแท็บกราฟ)"""
    ss = st.session_state
    nonce = ss.get("sig_nonce", 0)
    cfg = dict(kw.pop("column_config", {}) or {})
    cfg["yahoo"] = None
    ev = st.dataframe(df, key=f"{key}_{nonce}", on_select="rerun", selection_mode="single-row",
                      hide_index=True, column_config=cfg, **kw)
    rows = ev.selection.rows if ev is not None else []
    if rows:
        ss["sig_pick"] = df.iloc[rows[0]]["yahoo"]
        ss["sig_scroll"] = True
        ss["sig_nonce"] = nonce + 1
        st.rerun()


def render_signals_tab(set100: pd.DataFrame, clickable_table=None):
    ss = st.session_state
    if "sig_pick" in ss:  # หุ้นที่คลิกมาจากแผงหุ้นเด่น/ตาราง/Spectrum -> ตั้งค่าก่อนสร้าง selectbox
        ss["sig_detail"] = ss.pop("sig_pick")
    syms = tuple(set100["yahoo"])
    star = dict(zip(set100["yahoo"], set100["set50"])) if "set50" in set100 else {}
    o1, o2, o3 = st.columns([1.2, 1.6, 1.4])
    tf = o1.radio("กล่องสีคำนวณจากแท่ง", ["D", "W", "M"], horizontal=True, key="sig_tf",
                  format_func={"D": "รายวัน", "W": "1W", "M": "1M"}.get,
                  help="1W/1M เปลี่ยนเฉพาะกล่องสีในตาราง · คะแนนกฎและ Spectrum คำนวณจากแท่งรายวันเสมอ")
    sort = o2.selectbox("เรียงตาม", ["คำแนะนำการลงทุน (กล่องสี)", "สถานะทางเทคนิค (คะแนนกฎ มาก→น้อย)",
                                     "ทิศทาง (▲ ก่อน)", "Spectrum (มาก→น้อย)"], key="sig_sort")
    flt = o3.selectbox("กรอง", ["ทั้งหมด", "💪 หุ้นแข็งแกร่ง", "🌱 หุ้นน่าติดตาม (กำลังฟื้น)", "◆ กรอบม่วง",
                                "★ หุ้นน่าจับตา", "ซื้อ / ถือ", "บวกวันนี้", "ลบวันนี้"],
                       key="sig_flt")
    t = signal_table(syms, tf)
    if t.empty:
        st.warning("ดึงราคาไม่สำเร็จ ลองใหม่อีกครั้ง")
        return
    stale = t[t["stale"]]
    if len(stale):
        st.warning("ไม่แสดง " + ", ".join(f"{r.symbol} (ข้อมูลขาดช่วง {r.gap_days} วัน)" for r in stale.itertuples())
                   + " เพราะราคาย้อนหลังจาก Yahoo ไม่ครบ สัญญาณจะคลาดเคลื่อน")
        t = t[~t["stale"]]
    asof = t["date"].max()
    st.caption(f"ข้อมูลถึง {asof:%d %b %Y} · {len(t)} หุ้น · คำนวณจาก Yahoo Finance (ราคาจริง ไม่ปรับปันผล เหมือน set.or.th) · "
               "คลิกชื่อหุ้น แถวในตาราง หรือช่องใน Spectrum เพื่อดูเหตุผลและกราฟในช่อง 🔍 เหตุผลรายตัว")

    vc = t["verdict"].value_counts()
    m = st.columns(7)
    for col, v in zip(m, VERDICTS):
        col.metric(VERDICT_ICON[v], int(vc.get(v, 0)))
    m[5].metric("◆ กรอบม่วง", int(t["purple"].sum()))
    m[6].metric("★ น่าจับตา", int(t["watch"].sum()))

    _highlights(t)

    # ----- เหตุผลรายตัว + กราฟ (ต่อจากหุ้นเด่นวันนี้) -----
    st.markdown("<div id='sig-detail'></div>", unsafe_allow_html=True)
    st.markdown("#### 🔍 เหตุผลรายตัว")
    opts = sorted(t["yahoo"])
    if ss.get("sig_detail") not in opts:
        hl = t[t["box"].isin(["G", "B"])].sort_values(["score", "spec"], ascending=False)
        ss["sig_detail"] = hl["yahoo"].iloc[0] if len(hl) else opts[0]
    d1, d2 = st.columns([2, 1])
    sym = d1.selectbox("เลือกหุ้น (หรือคลิกชื่อหุ้นจากแผงด้านบน / ตาราง / Spectrum)", opts, key="sig_detail",
                       format_func=lambda s: s.replace(".BK", ""))
    d2.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    if d2.button("📈 เปิดในแท็บกราฟราคา (เต็มจอ)", key="sig_open_chart"):
        ss["goto_sym"] = sym
        st.rerun()
    with st.container(border=True):
        _detail(t, load_prices(syms), sym)
    if ss.pop("sig_scroll", False):
        _scroll_to("sig-detail")

    tab_a, tab_b, tab_c = st.tabs(["📋 ตารางสัญญาณรายตัว", "🌈 Market Lens: Signal Spectrum", "🧪 ทดสอบย้อนหลัง & วิธีคิด"])
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
        if sort.startswith("คำแนะนำ"):
            v = v.assign(_k=v["box_tf"].map(lambda b: BOX[b][2])).sort_values(["_k", "symbol"])
        elif sort.startswith("สถานะ"):
            v = v.sort_values(["score", "spec"], ascending=False)
        elif sort.startswith("ทิศทาง"):
            v = v.assign(_k=v["dir"].map(lambda d: DIRS[d][4])).sort_values(["_k", "dir_tot"], ascending=[True, False])
        else:
            v = v.sort_values("spec", ascending=False)
        tbl = pd.DataFrame({
            "yahoo": v["yahoo"],
            "หุ้น": v["symbol"] + v["yahoo"].map(lambda s: " ★" if star.get(s) == 1 else ""),
            "กลุ่ม": v["sector_th"],
            "ราคา": v["price"],
            "% วันนี้": v["chg1"] * 100,
            "กล่องสี": v["box_tf"].map(lambda b: BOX[b][0]),
            "ระดับ EMA": v["lvl_tf"].map(LEVELS) + v["near200"].map(lambda x: " ⚠️" if x else ""),
            "ทิศทาง": v["dir"].map(lambda d: f"{DIRS[d][0]} {DIRS[d][1]}"),
            "สัญญาณหนุน": v["sup"].map(lambda x: " · ".join(x)),
            "สัญญาณเตือน": v["warn"].map(lambda x: " · ".join(x)),
            "คะแนน": v["score"],
            "ผลอ่าน": v["verdict"].map(VERDICT_ICON),
            "Spectrum": v["spec"],
            "พิเศษ": (v["purple"].map(lambda x: "◆ " if x else "") + v["white"].map(lambda x: "▣ " if x else "")
                      + v["watch"].map(lambda x: "★" if x else "")),
        })
        _picker_table(tbl, "sig_tbl", use_container_width=True, height=560, column_config={
            "ราคา": st.column_config.NumberColumn(format="%.2f"),
            "% วันนี้": st.column_config.NumberColumn(format="%+.2f%%"),
            "คะแนน": st.column_config.NumberColumn(format="%+d"),
            "Spectrum": st.column_config.ProgressColumn(min_value=-100, max_value=100, format="%+d"),
        })
        st.caption("คลิกแถวเพื่อดูเหตุผลและกราฟในช่อง 🔍 เหตุผลรายตัวด้านบน · ★ หลังชื่อ = SET50 · "
                   "⚠️ = ราคาห่าง EMA200 ไม่ถึง 2% · พิเศษ: ◆ กรอบม่วง · ▣ Spectrum ≥ +75 · ★ หุ้นน่าจับตา")

    with tab_b:
        st.caption("แต่ละแถว = กลุ่มอุตสาหกรรม (Yahoo) เรียงจากคะแนนเฉลี่ยถ่วง market cap สูงสุด (ตัวเลขท้ายแถว) · "
                   "ในแถวเรียงจากคะแนนต่ำ (ซ้าย แดง) ไปสูง (ขวา เขียว) · ความกว้าง ∝ √market cap · สีเข้มสุดที่ ±70 · "
                   "กรอบม่วงหนา ◆ = กรอบม่วง · กรอบดำ = Spectrum ≥ +75 (ต้นฉบับใช้กรอบขาว) · ★ = หุ้นน่าจับตา")
        nonce = ss.get("tbl_nonce", 0)
        ev = st.plotly_chart(_spectrum_fig(t), use_container_width=True, on_select="rerun",
                             selection_mode="points", key=f"sig_spec_{nonce}")
        pts = ev.selection.points if ev is not None and ev.selection else []
        if pts and pts[0].get("customdata"):
            cdv = pts[0]["customdata"]
            ss["sig_pick"] = cdv[0] if isinstance(cdv, (list, tuple)) else cdv
            ss["sig_scroll"] = True
            ss["tbl_nonce"] = nonce + 1
            st.rerun()
        top = t.sort_values("spec", ascending=False)
        a1, a2 = st.columns(2)
        a1.markdown("**◆ กรอบม่วง:** " + (", ".join(top[top["purple"]]["symbol"]) or "ไม่มีวันนี้"))
        a1.markdown("**▣ Spectrum ≥ +75:** " + (", ".join(top[top["white"]]["symbol"]) or "ไม่มีวันนี้"))
        a2.markdown("**★ หุ้นน่าจับตา:** " + (", ".join(top[top["watch"]]["symbol"]) or "ไม่มีวันนี้"))

    with tab_c:
        if BT_INFO:
            st.markdown(BT_INFO)
        if not BT_TABLE.empty:
            st.dataframe(BT_TABLE, hide_index=True, use_container_width=True, column_config=BT_CFG)
            st.caption("ช่วงปี = ชนะตลาด 3 เดือนเฉลี่ยในช่วงนั้น · ⚠️ ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำซื้อขาย")
        with st.expander("📘 กติกาทั้ง 4 ชั้น และจุดที่ตีความเพิ่ม"):
            st.markdown(RULES_MD)


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
