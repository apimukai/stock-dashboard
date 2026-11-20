"""ปรับหน้าตาสำหรับจอมือถือ (กว้างไม่เกิน 640 px) · จอคอมไม่เปลี่ยน
เบราว์เซอร์ตัดสินจากความกว้างจอเอง (CSS media query) ไม่ต้องตรวจว่าเป็นอุปกรณ์อะไร
- หัวเว็บเล็กลง · ราคา Watchlist เรียง 2 คอลัมน์
- แถบแท็บตัดบรรทัดเป็นปุ่มหลายแถว เห็นครบทุกแท็บโดยไม่ต้องปัด
- การ์ดหุ้น 2 ใบต่อแถว
- ปุ่มลอย "↑ เมนู" มุมล่างขวา กดแล้วกลับไปที่แถบแท็บ
"""
import streamlit as st
import streamlit.components.v1 as components

MOBILE_MAX = 640  # px

CSS = f"""
<style>
#mob-top {{ display: none; }}
@media (max-width: {MOBILE_MAX}px) {{
  /* หัวเว็บ + ระยะขอบ */
  h1 {{ font-size: 1.05rem !important; line-height: 1.3 !important; padding: .1rem 0 !important; }}
  [data-testid="stMainBlockContainer"], .block-container {{
    padding-top: 2.6rem !important; padding-left: .7rem !important; padding-right: .7rem !important; }}

  /* ราคา Watchlist: 2 คอลัมน์ ตัวเล็กลง */
  /* ยกเลิกกล่องคอลัมน์ ให้ราคาแต่ละตัวเรียงเป็นตาราง 2 ช่องต่อแถวเท่ากันหมด */
  .st-key-quoteboard [data-testid="stHorizontalBlock"] {{
    display: grid !important; grid-template-columns: 1fr 1fr !important; gap: .15rem .6rem !important; }}
  .st-key-quoteboard [data-testid="stColumn"],
  .st-key-quoteboard [data-testid="stColumn"] > div,
  .st-key-quoteboard [data-testid="stColumn"] [data-testid="stVerticalBlock"] {{ display: contents !important; }}
  .st-key-quoteboard [data-testid="stElementContainer"] {{ width: auto !important; margin: 0 !important; }}
  .st-key-quoteboard [data-testid="stMetricValue"] {{ font-size: 1.15rem !important; }}
  .st-key-quoteboard [data-testid="stMetricLabel"] p {{ font-size: .72rem !important; }}
  .st-key-quoteboard [data-testid="stMetricDelta"] {{ font-size: .72rem !important; }}

  /* แถบแท็บ: ปุ่มหลายแถว เห็นครบ ไม่ต้องปัด */
  [role="tablist"] {{ flex-wrap: wrap !important; gap: 5px !important; overflow: visible !important;
                      scroll-margin-top: 64px; /* กดปุ่ม ↑ เมนู แล้วไม่ให้แถวแรกโดนแถบหัวเว็บบัง */ }}
  [role="tablist"] [role="tab"] {{
    flex: 0 0 auto !important; height: auto !important; min-height: 34px; margin: 0 !important; padding: 4px 9px !important;
    border: 1px solid rgba(128,128,128,.35) !important; border-radius: 9px !important; }}
  [role="tablist"] [role="tab"] p {{ font-size: .8rem !important; }}
  [role="tablist"] [role="tab"][aria-selected="true"] {{
    background: rgba(255,75,75,.12) !important; border-color: rgba(255,75,75,.7) !important; }}
  [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display: none !important; }}

  /* การ์ดหุ้น 2 ใบต่อแถว */
  .st-key-cardgrid button {{ width: calc(50vw - 24px) !important; min-height: 84px !important; padding: 6px 8px !important; }}
  .st-key-cardgrid button p {{ font-size: .82rem !important; }}
  .st-key-cardgrid button p strong:nth-of-type(2) {{ font-size: 1.2rem !important; }}

  /* ปุ่มลอยกลับไปที่เมนู */
  #mob-top {{ display: flex; }}
}}
</style>
"""

# ใส่ปุ่มลอยลงในหน้าเว็บหลัก (กรอบ components อยู่ origin เดียวกัน จึงแก้หน้าแม่ได้) · ทำครั้งเดียวต่อหน้า
TOP_BTN = """
<script>
(() => {
  const d = window.parent.document;
  if (d.getElementById('mob-top')) return;
  const b = d.createElement('button');
  b.id = 'mob-top';
  b.textContent = '↑ เมนู';
  b.title = 'กลับไปที่แถบแท็บ';
  Object.assign(b.style, {position: 'fixed', right: '14px', bottom: '18px', zIndex: 99999, padding: '10px 15px',
    borderRadius: '22px', border: 'none', background: '#ff4b4b', color: '#fff', fontSize: '14px', fontWeight: '600',
    boxShadow: '0 2px 10px rgba(0,0,0,.28)', alignItems: 'center', cursor: 'pointer'});
  b.onclick = () => {
    const t = d.querySelector('[role="tablist"]');
    if (t) t.scrollIntoView({behavior: 'auto', block: 'start'});  // กระโดดทันที (หน้ายาวหลายพัน px)
  };
  d.body.appendChild(b);
})();
</script>
"""


def apply_mobile_ui():
    st.markdown(CSS, unsafe_allow_html=True)
    components.html(TOP_BTN, height=0)
