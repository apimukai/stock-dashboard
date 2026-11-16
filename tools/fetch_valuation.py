"""ดึง P/E และอัตราปันผลของ S&P 500 รายเดือนจาก multpl.com -> files/sp500_valuation_monthly.csv
ใช้เป็นข้อมูลสำรองของแท็บ "สัดส่วนหุ้น-พันธบัตร" เมื่อเว็บ multpl เข้าไม่ได้
รัน: python tools/fetch_valuation.py   (GitHub Actions รันให้ทุกวัน)
ถ้าดึงไม่สำเร็จ สคริปต์จะหยุดก่อนเขียนไฟล์ ไฟล์เดิมไม่เสีย
"""
import io
import os
import sys

import pandas as pd
import requests

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files", "sp500_valuation_monthly.csv")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36"}


def multpl(slug: str) -> pd.Series:
    html = requests.get(f"https://www.multpl.com/{slug}/table/by-month", headers=UA, timeout=40).text
    t = pd.read_html(io.StringIO(html))[0]
    t.columns = ["date", "v"]
    t["date"] = pd.to_datetime(t["date"])
    t["v"] = pd.to_numeric(t["v"].astype(str).str.replace("†", "").str.replace("%", "").str.strip(), errors="coerce")
    return t.dropna().set_index("date")["v"].sort_index()


if __name__ == "__main__":
    pe, dy = multpl("s-p-500-pe-ratio"), multpl("s-p-500-dividend-yield")
    if len(pe) < 1000 or len(dy) < 1000:
        sys.exit("ข้อมูลไม่ครบ ไม่เขียนไฟล์")
    out = pd.DataFrame({"pe": pe, "div_yield_pct": dy}).sort_index()
    out.index.name = "date"
    out.to_csv(OUT, float_format="%.4f")
    print(f"pe {pe.index[-1].date()} = {pe.iloc[-1]} · dy {dy.index[-1].date()} = {dy.iloc[-1]} -> {OUT}")
