"""บันทึกรายชื่อหุ้นที่ติดสัญญาณเด่นประจำวัน -> files/signal_log/<set100|sp500>_<ปี>.csv
เก็บเฉพาะหุ้นที่ติดกลุ่ม 🚀 โมเมนตัม / 💪 แข็งแกร่ง / 🌱 น่าติดตาม / ◆ กรอบม่วง / ★ น่าจับตา หรือได้ผลอ่าน "ซื้อ"
ใช้กฎชุดเดียวกับแท็บสัญญาณเทคนิค (tools/set_signals.py) · รันซ้ำวันเดียวกันจะเขียนทับข้อมูลวันนั้น
GitHub Actions รันทุกวันทำการหลังตลาดสหรัฐฯ ปิด · รันเองได้: python tools/snapshot_signals.py
"""
import datetime as dt
import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)  # ใช้ st.cache_data นอก Streamlit ได้ แต่จะมีคำเตือนเยอะ

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402
import set_signals as S  # noqa: E402


def universe(market: str) -> tuple:
    if market == "TH":
        df = pd.read_csv(os.path.join(S.FILES, "set100.csv"), comment="#", encoding="utf-8-sig")
        return tuple(df["yahoo"])
    df = pd.read_csv(os.path.join(S.FILES, "us_indices.csv"), comment="#")
    return tuple(df.loc[df["sp500"] == 1, "yahoo"])


CLOSE = {"TH": ("Asia/Bangkok", 17, 0), "US": ("America/New_York", 16, 30)}  # เวลาที่ถือว่าตลาดปิดแล้ว


def last_closed_day(market: str):
    """ถ้าตอนนี้ตลาดยังเปิดอยู่ (หรือยังไม่ถึงเวลาปิด) ให้ใช้ถึงเมื่อวาน เพื่อไม่บันทึกราคาระหว่างวัน"""
    from zoneinfo import ZoneInfo
    tz, hh, mm = CLOSE[market]
    now = dt.datetime.now(ZoneInfo(tz))
    if (now.hour, now.minute) < (hh, mm):
        return now.date() - dt.timedelta(days=1)
    return None  # ปิดแล้ว ใช้ราคาล่าสุดได้เลย


def main():
    ok = 0
    for market in ("TH", "US"):
        try:
            t = S.signal_table(universe(market), "D", market, last_closed_day(market))
            day, n = S.save_snapshot(t, market)
            print(f"{S.LOG_KEY[market]}: {day} บันทึก {n} หุ้น")
            ok += 1
        except Exception as e:  # ตลาดหนึ่งพังไม่ให้กระทบอีกตลาด
            print(f"{S.LOG_KEY[market]}: ผิดพลาด {type(e).__name__}: {e}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
