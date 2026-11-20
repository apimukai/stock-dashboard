"""บันทึกรายงานประจำวัน -> files/daily_report/<YYYY-MM-DD>.json (วันที่ตามเวลาไทย) ให้ย้อนดูในแท็บ "รายงานประจำวัน"
ใช้ข้อมูลชุดเดียวกับหน้าเว็บ (tools/daily_report.py) · ต้องรันหลัง snapshot_signals.py เพื่อให้รายชื่อหุ้นเป็นของวันนี้
GitHub Actions รันทุกวันทำการ · รันเองได้: python tools/snapshot_report.py · รันซ้ำวันเดียวกันจะเขียนทับ
"""
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import daily_report as R  # noqa: E402

# streamlit ตั้งค่า logger ตอน import จึงต้องปิดคำเตือนหลัง import (ใช้ st.cache_data นอกหน้าเว็บได้ แต่เตือนเยอะ)
for _n in list(logging.root.manager.loggerDict):
    if _n.startswith("streamlit"):
        logging.getLogger(_n).setLevel(logging.ERROR)

if __name__ == "__main__":
    rep = R.build_report()
    path = R.save_report(rep)
    print(f"saved {path}")
    print(rep["text"])
