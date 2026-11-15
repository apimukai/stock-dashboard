"""ดัชนีหุ้นรายประเทศ: แผนที่โลก + ผลตอบแทนสกุลท้องถิ่น/USD + ค่าเงิน (ใช้เป็นตัวแทน flow เงินต่างชาติ)
ใช้ใน stock_dashboard.py · รายชื่อดัชนีอยู่ที่ files/world_indices.csv
"""
import os
import datetime as dt
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
WORLD_CSV = os.path.join(HERE, "..", "files", "world_indices.csv")
PRICES_CSV = os.path.join(HERE, "..", "files", "assets_prices.csv")
PERIODS = ["1D", "5D", "1M", "YTD", "1Y"]
# แดงเข้ม -> เทาอ่อน (0%) -> เขียวเข้ม
SCALE = [[0, "#67000d"], [0.25, "#e34a33"], [0.5, "#eeeeee"], [0.75, "#41ab5d"], [1, "#00441b"]]


def load_world() -> pd.DataFrame:
    # ผูก cache กับเวลาแก้ไขไฟล์ แก้ CSV แล้วเห็นผลทันที
    return _load_world(os.path.getmtime(WORLD_CSV))


@st.cache_data
def _load_world(mtime: float) -> pd.DataFrame:
    return pd.read_csv(WORLD_CSV, comment="#", encoding="utf-8-sig")


def _naive(s: pd.Series) -> pd.Series:
    s = s.dropna()
    idx = s.index.tz_localize(None) if getattr(s.index, "tz", None) is not None else s.index
    s.index = pd.DatetimeIndex(idx).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _local_file(sym):
    """ประวัติจาก files/assets_prices.csv (Yahoo มีประวัติ ^SET.BK แค่วันล่าสุด)"""
    try:
        p = pd.read_csv(PRICES_CSV, encoding="utf-8-sig")
        p = p[p["symbol"] == sym]
        return pd.Series(p["close"].astype(float).values, index=pd.to_datetime(p["date"]))
    except Exception:
        return pd.Series(dtype=float)


def _base(c: pd.Series, p: str):
    """วันที่และราคาฐานของแต่ละช่วงเวลา"""
    last_d = c.index[-1]
    if p == "1D":
        return (c.index[-2], c.iloc[-2]) if len(c) > 1 else (None, None)
    if p == "5D":
        return (c.index[-6], c.iloc[-6]) if len(c) > 5 else (None, None)
    target = {"1M": last_d - pd.Timedelta(days=30), "1Y": last_d - pd.Timedelta(days=365),
              "YTD": pd.Timestamp(last_d.year, 1, 1) - pd.Timedelta(days=1)}[p]
    b = c[c.index <= target]
    return (b.index[-1], b.iloc[-1]) if len(b) else (None, None)


def get_world() -> pd.DataFrame:
    return _get_world(os.path.getmtime(WORLD_CSV))


@st.cache_data(ttl=120, show_spinner="กำลังดึงดัชนีทั่วโลก ...")
def _get_world(mtime: float) -> pd.DataFrame:
    w = load_world()
    fx_syms = sorted({f"{c}=X" for c in w["currency"] if c != "USD"})
    d = yf.download(list(w["yahoo"]) + fx_syms, period="2y", interval="1d", progress=False,
                    threads=True, auto_adjust=False)["Close"]
    rows = []
    for r in w.itertuples():
        c = _naive(d[r.yahoo]) if r.yahoo in d else pd.Series(dtype=float)
        if len(c) < 30:  # ประวัติไม่พอ -> เติมจากไฟล์ในโปรเจกต์
            loc = _naive(_local_file(r.yahoo))
            c = pd.concat([loc[~loc.index.isin(c.index)], c]).sort_index()
        row = r._asdict()
        if len(c) < 2:
            rows.append(row)
            continue
        fx = _naive(d[f"{r.currency}=X"]) if r.currency != "USD" and f"{r.currency}=X" in d else None
        row.update({"ราคา": c.iloc[-1], "เปลี่ยนแปลง": c.iloc[-1] - c.iloc[-2], "วันที่": c.index[-1].date()})
        for p in PERIODS:
            bd, bv = _base(c, p)
            if bd is None:
                continue
            r_px = c.iloc[-1] / bv - 1
            # ค่าเงินท้องถิ่นเทียบ USD: + = แข็งค่า (FX = จำนวนเงินท้องถิ่นต่อ 1 USD)
            fxc = 0.0
            if fx is not None and len(fx):
                fb, fl = fx[fx.index <= bd], fx[fx.index <= c.index[-1]]  # ใช้ค่าเงินวันเดียวกับราคาหุ้น
                if len(fb) and len(fl):
                    fxc = fb.iloc[-1] / fl.iloc[-1] - 1
            if r.kind == "etf":  # ราคา ETF เป็น USD อยู่แล้ว
                usd, local = r_px, (1 + r_px) / (1 + fxc) - 1
            elif r.kind == "basket":  # ดัชนีหลายประเทศ: ราคาเป็นสกุล currency -> แปลงเป็น USD, ไม่มีสกุลท้องถิ่นเดียว
                usd = (1 + r_px) * (1 + fxc) - 1
                row[f"{p} usd"] = usd * 100
                continue
            else:
                local, usd = r_px, (1 + r_px) * (1 + fxc) - 1
            row[f"{p} local"], row[f"{p} usd"], row[f"{p} fx"] = local * 100, usd * 100, fxc * 100
        rows.append(row)
    out = pd.DataFrame(rows)
    out["1D %"] = out["เปลี่ยนแปลง"] / (out["ราคา"] - out["เปลี่ยนแปลง"]) * 100  # ให้ตรงกับราคาบนการ์ด
    for c in [f"{p} {k}" for p in PERIODS for k in ("local", "usd", "fx")]:
        if c not in out:
            out[c] = float("nan")
    return out


# ---------------- UI ----------------
_ICONS = {  # ไอคอนสำหรับดัชนีภูมิภาคที่ไม่มีธงประเทศ (SVG แบบ data URI)
    "latam": "data:image/svg+xml;utf8," + quote(
        '<svg xmlns="http://www.w3.org/2000/svg" width="60" height="40" viewBox="0 0 60 40">'
        '<rect width="60" height="40" rx="4" fill="#0b7a3e"/>'
        '<text x="30" y="27" font-family="Arial" font-size="15" font-weight="700" fill="#ffd700" '
        'text-anchor="middle">LatAm</text></svg>'),
}


def flag_url(code: str, w: int = 40) -> str:
    return _ICONS.get(code) or f"https://flagcdn.com/w{w}/{code}.png"


def render_world(df: pd.DataFrame):
    c1, c2, c3 = st.columns([1.3, 1.4, 1])
    p = c1.radio("ช่วงเวลา", PERIODS, index=3, horizontal=True, key="w_period")
    basis = c2.radio("วัดผลตอบแทนเป็น", ["USD (มุมมองนักลงทุนต่างชาติ)", "สกุลเงินท้องถิ่น"],
                     index=1, horizontal=True, key="w_basis")  # ค่าเริ่มต้น: สกุลเงินท้องถิ่น
    scope = c3.selectbox("โฟกัส", ["ทั้งโลก", "เอเชีย", "ยุโรป"], key="w_scope")
    col = f"{p} usd" if basis.startswith("USD") else f"{p} local"
    d = df.dropna(subset=[col]).copy()
    if d.empty:
        st.warning("ไม่มีข้อมูล")
        return
    lim = max(1.0, float(d[col].abs().max()))
    d = d.assign(val_=d[col], local_=d[f"{p} local"], usd_=d[f"{p} usd"], fx_=d[f"{p} fx"])

    # ----- แถบธงสรุปผลตอบแทน (เรียงจากดีไปแย่) -----
    css = """<style>
    .wf-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:4px 0 10px}
    .wf{border-radius:10px;padding:8px 10px;border:1px solid rgba(128,128,128,.25);display:flex;gap:10px;align-items:center}
    .wf img{width:40px;height:27px;object-fit:cover;border-radius:3px;box-shadow:0 0 0 1px rgba(0,0,0,.15)}
    .wf .n{font-size:.8rem;opacity:.75;line-height:1.2}.wf .v{font-size:1.25rem;font-weight:700;font-variant-numeric:tabular-nums}
    </style>"""
    cells = []
    for r in d.sort_values(col, ascending=False).itertuples():
        v = r.val_
        a = min(abs(v) / lim, 1) * 0.35 + 0.06
        bg = f"rgba(44,160,44,{a:.2f})" if v > 0 else f"rgba(214,39,40,{a:.2f})" if v < 0 else "transparent"
        cells.append(f'<div class="wf" style="background:{bg}"><img src="{flag_url(r.flag)}" alt="{r.country}">'
                     f'<div><div class="n">{r.country}<br>{r.index_name}</div><div class="v">{v:+.2f}%</div></div></div>')
    st.markdown(css + '<div class="wf-row">' + "".join(cells) + "</div>", unsafe_allow_html=True)

    # ----- แผนที่โลก (ยุโรป = ระบายทุกประเทศยูโรโซน) -----
    rows = []
    for r in d.itertuples():
        if pd.isna(r.map_iso3) or not str(r.map_iso3).strip():
            continue  # ดัชนีย่อย (เช่น Nasdaq, Dow) ไม่ระบายบนแผนที่ สหรัฐฯ ใช้ S&P 500
        for iso in str(r.map_iso3).split(";"):
            rows.append({"iso": iso, "val": r.val_, "country": r.country, "index_name": r.index_name,
                         "local": r.local_, "usd": r.usd_, "fx": r.fx_,
                         "label_iso": iso == str(r.map_iso3).split(";")[0]})  # ป้ายที่ประเทศแรกของกลุ่ม
    mp = pd.DataFrame(rows)
    def _f(v):
        return "–" if pd.isna(v) else f"{v:+.2f}%"

    hover = [f"<b>{c}</b><br>{ix}<br>Local {_f(lo)} · USD {_f(us)}<br>Currency vs USD {_f(fx)}"
             for c, ix, lo, us, fx in zip(mp["country"], mp["index_name"], mp["local"], mp["usd"], mp["fx"])]
    fig = go.Figure(go.Choropleth(
        locations=mp["iso"], z=mp["val"], zmin=-lim, zmax=lim, zmid=0, colorscale=SCALE,
        text=hover, hovertemplate="%{text}<extra></extra>", marker_line_color="white", marker_line_width=0.5,
        colorbar=dict(title="%", ticksuffix="%", len=0.75, thickness=14)))
    hk = mp[mp["iso"] == "HKG"]
    if len(hk):  # ฮ่องกงเล็กมากบนแผนที่ -> จุดสี
        fig.add_trace(go.Scattergeo(locations=hk["iso"], locationmode="ISO-3", mode="markers",
                                    marker=dict(size=12, color=hk["val"], colorscale=SCALE, cmin=-lim, cmax=lim,
                                                line=dict(color="white", width=1.5)),
                                    text=[h for h, i in zip(hover, mp["iso"]) if i == "HKG"],
                                    hovertemplate="%{text}<extra></extra>", showlegend=False))
    lab = mp[mp["label_iso"]]
    fig.add_trace(go.Scattergeo(locations=lab["iso"], locationmode="ISO-3", mode="text",
                                text=[f"{c}<br>{v:+.1f}%" for c, v in zip(lab["country"].str.replace(" (Eurozone)", ""),
                                                                           lab["val"])],
                                textfont=dict(size=10, color="#111"), hoverinfo="skip", showlegend=False))
    geo = {"ทั้งโลก": dict(scope="world", projection_type="natural earth", lataxis_range=[-10, 72],
                           lonaxis_range=[-130, 150]),
           "เอเชีย": dict(scope="asia"), "ยุโรป": dict(scope="europe")}[scope]
    fig.update_geos(**geo, showcountries=True, countrycolor="white", showland=True, landcolor="#dcdcdc",
                    showocean=True, oceancolor="#eef4fa", showframe=False, showcoastlines=False)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"สีเขียวยิ่งเข้ม = บวกมาก · สีแดงยิ่งเข้ม = ลบมาก (สีเต็มที่ ±{lim:.1f}%) · ประเทศสีเทา = ไม่อยู่ในชุดนี้ · "
               "ยุโรปใช้ Euro Stoxx 50 จึงระบายสีทุกประเทศยูโรโซนเท่ากัน · ชี้เมาส์เพื่อดูรายละเอียด")

    # ----- flow -----
    g1, g2 = st.columns([1, 1.2])
    with g1:
        s = d.sort_values(col)
        f = go.Figure(go.Bar(x=s[col], y=s["country"], orientation="h",
                             marker=dict(color=s[col], colorscale=SCALE, cmin=-lim, cmax=lim),
                             text=s[col].map("{:+.1f}%".format), textposition="outside"))
        f.update_layout(title=f"Return {p} ({'USD' if basis.startswith('USD') else 'local currency'})",
                        height=420, margin=dict(l=10, r=40, t=40, b=10))
        st.plotly_chart(f, use_container_width=True)
    with g2:
        st.markdown("**💸 เงินไหลเข้า-ออกที่ไหน (ประมาณจากหุ้น + ค่าเงิน)**")
        f = go.Figure()
        fd = d.dropna(subset=[f"{p} fx", f"{p} local"])  # ดัชนีภูมิภาค (LatAm) ไม่มีค่าเงินเดียว -> ไม่อยู่ในกราฟนี้
        x, y = fd[f"{p} fx"], fd[f"{p} local"]
        for lab_, cond, c in [("เงินไหลเข้าชัด", (x > 0) & (y > 0), "#1b8a3a"),
                              ("เงินไหลออกชัด", (x < 0) & (y < 0), "#d62728"),
                              ("สัญญาณผสม", ~(((x > 0) & (y > 0)) | ((x < 0) & (y < 0))), "#888888")]:
            sub = fd[cond]
            f.add_trace(go.Scatter(x=sub[f"{p} fx"], y=sub[f"{p} local"], mode="markers+text", name=lab_,
                                   text=sub["country"], textposition="top center", textfont=dict(size=11),
                                   marker=dict(size=11, color=c), customdata=sub[f"{p} usd"],
                                   hovertemplate="%{text}<br>Index (local) %{y:+.2f}%<br>Currency %{x:+.2f}%"
                                                 "<br>Total in USD %{customdata:+.2f}%<extra></extra>"))
        f.add_hline(y=0, line_color="gray", line_width=1)
        f.add_vline(x=0, line_color="gray", line_width=1)
        f.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="ค่าเงินเทียบ USD (%) → แข็งค่า", yaxis_title="ดัชนีหุ้น สกุลท้องถิ่น (%)",
                        legend=dict(orientation="h", y=1.08))
        st.plotly_chart(f, use_container_width=True)
        st.caption("มุมขวาบน = หุ้นขึ้นและค่าเงินแข็ง มักเป็นสัญญาณเงินต่างชาติไหลเข้า · มุมซ้ายล่าง = ไหลออก · "
                   "เป็นการประมาณเท่านั้น ข้อมูล fund flow จริงต้องใช้แหล่งข้อมูลเสียเงิน (เช่น EPFR)")

    # ----- ตาราง -----
    d["flag_img"] = d["flag"].map(lambda c: flag_url(c, 40))
    tbl = d[["flag_img", "country", "index_name", "ราคา"] + [f"{q} usd" for q in PERIODS]
            + [f"{p} local", f"{p} fx", "วันที่"]].sort_values(f"{p} usd", ascending=False)
    st.dataframe(tbl, hide_index=True, use_container_width=True,
                 column_config={"flag_img": st.column_config.ImageColumn("", width="small"),
                                "country": "Country", "index_name": "Index",
                                "ราคา": st.column_config.NumberColumn("Last", format="%,.2f"),
                                **{f"{q} usd": st.column_config.NumberColumn(f"{q} (USD)", format="%+.2f%%") for q in PERIODS},
                                f"{p} local": st.column_config.NumberColumn(f"{p} local", format="%+.2f%%"),
                                f"{p} fx": st.column_config.NumberColumn(f"{p} currency", format="%+.2f%%"),
                                "วันที่": "Date"})
    st.caption("Vietnam ใช้ VanEck Vietnam ETF (USD) แทน เพราะ Yahoo ไม่มีประวัติ VN-Index · "
               "EM Latin America ใช้ Xtrackers MSCI EM Latin America ETF (DBX3.DE, EUR) แปลงเป็น USD "
               "(บราซิล เม็กซิโก ชิลี โคลอมเบีย เปรู) ไม่มีสกุลท้องถิ่นเดียว จึงแสดงเฉพาะแบบ USD · "
               "Thailand ใช้ SET Index จาก files/assets_prices.csv + Yahoo · ธงจาก flagcdn.com")
