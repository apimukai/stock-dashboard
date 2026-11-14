# Stock Dashboard (Streamlit + Yahoo Finance)

Dashboard ติดตามหุ้นไทย (SET100/SET50) หุ้นสหรัฐฯ (Dow 30, Nasdaq-100, S&P 500) และดัชนีทั่วโลก
พร้อมข้อมูลพื้นฐาน มุมมองนักวิเคราะห์ และ Heatmap รายกลุ่มอุตสาหกรรมสหรัฐฯ

> ข้อมูลจาก Yahoo Finance ผ่าน yfinance · อาจดีเลย์ตามตลาด · เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน

## รันบนเครื่องตัวเอง

```bash
pip install -r requirements.txt
streamlit run tools/stock_dashboard.py
```

หรือดับเบิลคลิก `tools/run_dashboard.bat`

## ขึ้นเว็บด้วย Streamlit Community Cloud (ฟรี)

1. เข้า https://share.streamlit.io แล้ว login ด้วยบัญชี GitHub
2. กด **Create app** → **Deploy a public app from GitHub**
3. เลือก repo นี้ · Branch: `main` · Main file path: `tools/stock_dashboard.py`
4. **Advanced settings** → Python version: `3.12`
5. กด **Deploy** รอประมาณ 2–5 นาที จะได้ลิงก์ `https://<ชื่อแอป>.streamlit.app`

หลังจากนั้นทุกครั้งที่มีการ push ขึ้น GitHub (รวมถึงตอนที่ GitHub Actions อัปเดตข้อมูล) เว็บจะอัปเดตเอง

## การอัปเดตข้อมูล

| ข้อมูล | อัปเดตอย่างไร |
|---|---|
| ราคาหุ้น กราฟ ข้อมูลพื้นฐาน นักวิเคราะห์ | ดึงสดจาก Yahoo ทุกครั้งที่เปิดเว็บ (อัตโนมัติ) |
| `files/assets_prices.csv` (ประวัติ SET Index ฯลฯ) | GitHub Actions รันทุกวันจันทร์–ศุกร์ 05:30 น. (เวลาไทย) |
| `files/us_indices.csv` (รายชื่อ S&P 500, Nasdaq-100, Dow) | GitHub Actions รันทุกวันที่ 1 ของเดือน |
| `files/set100.csv` (รายชื่อ SET100/SET50) | **แก้เอง** ปีละ 2 ครั้ง (ม.ค./ก.ค.) ตามประกาศของ set.or.th |
| `files/world_indices.csv` (ดัชนีทั่วโลก) | แก้เองเมื่ออยากเพิ่ม/ลดดัชนี |

สั่งอัปเดตข้อมูลทันทีได้ที่แท็บ **Actions** ใน GitHub → **Update market data** → **Run workflow**

## โครงสร้างไฟล์

```
tools/
  stock_dashboard.py      หน้าหลัก (ไฟล์ที่ Streamlit รัน)
  stock_profile.py        ข้อมูลพื้นฐาน/นักวิเคราะห์รายหุ้น
  index_fundamentals.py   ข้อมูลพื้นฐานของดัชนี (จาก ETF ที่อิงดัชนี)
  world_markets.py        แผนที่โลก/ดัชนีรายประเทศ
  sector_heatmap.py       Heatmap กลุ่มอุตสาหกรรมสหรัฐฯ
  fetch_assets.py         สคริปต์อัปเดต files/assets_prices.csv
  fetch_us_lists.py       สคริปต์อัปเดต files/us_indices.csv
files/                    ไฟล์ข้อมูล CSV
.github/workflows/        GitHub Actions อัปเดตข้อมูลอัตโนมัติ
```
