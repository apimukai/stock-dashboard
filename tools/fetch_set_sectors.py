"""ดึงกลุ่มอุตสาหกรรม (sector/industry) และจำนวนหุ้นของหุ้น SET100 จาก Yahoo Finance -> files/set100_sectors.csv
ใช้กับแท็บสัญญาณเทคนิค (จัดแถวใน Market Lens และขนาดช่องตาม market cap)
ข้อมูลนี้เปลี่ยนไม่บ่อย รันใหม่เมื่อรายชื่อ SET100 เปลี่ยน (ม.ค./ก.ค.): python tools/fetch_set_sectors.py
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "..", "files")


def one(sym):
    try:
        info = yf.Ticker(sym).info
    except Exception:
        info = {}
    return {"yahoo": sym, "sector": info.get("sector"), "industry": info.get("industry"),
            "shares": info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")}


def main():
    base = pd.read_csv(os.path.join(FILES, "set100.csv"), comment="#")
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, base["yahoo"]))
    df = base[["symbol", "yahoo"]].merge(pd.DataFrame(rows), on="yahoo", how="left")
    ok = df["sector"].notna().sum()
    print(f"มี sector {ok}/{len(df)} ตัว")
    if ok < len(df) * 0.8:
        sys.exit("ข้อมูลไม่ครบ ไม่เขียนไฟล์")
    df.to_csv(os.path.join(FILES, "set100_sectors.csv"), index=False)


if __name__ == "__main__":
    main()
