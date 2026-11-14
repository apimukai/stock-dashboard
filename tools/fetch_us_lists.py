"""ดึงรายชื่อหุ้นในดัชนีสหรัฐฯ (Dow 30, Nasdaq-100, S&P 500) จาก Wikipedia -> files/us_indices.csv
ใช้: python fetch_us_lists.py
ควรรันใหม่เป็นระยะ (เช่น ทุกไตรมาส) เพราะรายชื่อในดัชนีเปลี่ยนได้
"""
import io
import os
import urllib.request
import datetime as dt

import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files", "us_indices.csv")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}


def tables(url):
    html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read().decode("utf-8")
    return pd.read_html(io.StringIO(html))


def pick(tbls, need):
    for t in tbls:
        cols = [str(c) for c in t.columns]
        if all(any(n.lower() in c.lower() for c in cols) for n in need):
            return t
    raise RuntimeError(f"ไม่พบตารางที่มีคอลัมน์ {need}")


def col(t, name):
    return next(c for c in t.columns if name.lower() in str(c).lower())


sp = pick(tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"), ["Symbol", "Security", "GICS Sector"])
sp = pd.DataFrame({"symbol": sp[col(sp, "Symbol")], "name": sp[col(sp, "Security")],
                   "sector": sp[col(sp, "GICS Sector")]})

# Wikipedia ไม่มีตารางรายชื่อ Nasdaq-100 / Dow แล้ว -> ใช้ slickcharts (มีน้ำหนักในดัชนีด้วย)
nd = pick(tables("https://www.slickcharts.com/nasdaq100"), ["Symbol", "Company"])
nd = pd.DataFrame({"symbol": nd[col(nd, "Symbol")], "name": nd[col(nd, "Company")], "sector": None})

dj = pick(tables("https://www.slickcharts.com/dowjones"), ["Symbol", "Company"])
dj = pd.DataFrame({"symbol": dj[col(dj, "Symbol")], "name": dj[col(dj, "Company")], "sector": None})

for d in (sp, nd, dj):
    d["symbol"] = d["symbol"].astype(str).str.strip().str.replace(r"^(NYSE|NASDAQ|Nasdaq):\s*", "", regex=True)

allsym = pd.concat([sp, nd, dj]).drop_duplicates("symbol")  # S&P 500 ก่อน: ใช้ชื่อ/sector แบบ GICS
allsym["sector"] = allsym["sector"].fillna("Other")
allsym["yahoo"] =allsym["symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
allsym["dow30"] = allsym["symbol"].isin(dj["symbol"]).astype(int)
allsym["ndx100"] = allsym["symbol"].isin(nd["symbol"]).astype(int)
allsym["sp500"] = allsym["symbol"].isin(sp["symbol"]).astype(int)
allsym = allsym.sort_values("symbol")[["symbol", "yahoo", "name", "sector", "dow30", "ndx100", "sp500"]]

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    f.write(f"# US index constituents (S&P 500: Wikipedia, Nasdaq-100/Dow: slickcharts), fetched {dt.date.today()}\n")
    allsym.to_csv(f, index=False)
print(f"Dow30={allsym.dow30.sum()} Nasdaq100={allsym.ndx100.sum()} SP500={allsym.sp500.sum()} "
      f"total={len(allsym)} -> {OUT}")
