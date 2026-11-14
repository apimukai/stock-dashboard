"""Fetch daily closes for the dashboard's asset tab from Yahoo Finance's chart API.
Usage: python -I fetch_assets.py <data folder> [start YYYY-MM-DD]
Writes assets_meta.csv (overwrite) and assets_prices.csv (merge: keeps old rows, adds/updates newer ones)."""
import csv, io, json, os, sys, time, urllib.request, datetime as dt
from zoneinfo import ZoneInfo

F = sys.argv[1]
START = sys.argv[2] if len(sys.argv) > 2 else '2024-11-01'

# symbol, Thai name, group, country/region, currency, fx symbol, fx mode (div: local per USD, mul: USD per local), unit
META = [
    ('^GSPC', 'S&P 500', 'หุ้นสหรัฐฯ', 'สหรัฐฯ', 'USD', '', '', 'จุด'),
    ('^IXIC', 'Nasdaq Composite', 'หุ้นสหรัฐฯ', 'สหรัฐฯ', 'USD', '', '', 'จุด'),
    ('^DJI', 'Dow Jones', 'หุ้นสหรัฐฯ', 'สหรัฐฯ', 'USD', '', '', 'จุด'),
    ('^RUT', 'Russell 2000 (หุ้นเล็ก)', 'หุ้นสหรัฐฯ', 'สหรัฐฯ', 'USD', '', '', 'จุด'),
    ('^SET.BK', 'SET Index', 'หุ้นเอเชีย', 'ไทย', 'THB', 'THB=X', 'div', 'จุด'),
    ('^N225', 'Nikkei 225', 'หุ้นเอเชีย', 'ญี่ปุ่น', 'JPY', 'JPY=X', 'div', 'จุด'),
    ('000001.SS', 'Shanghai Composite', 'หุ้นเอเชีย', 'จีน', 'CNY', 'CNY=X', 'div', 'จุด'),
    ('^HSI', 'Hang Seng', 'หุ้นเอเชีย', 'ฮ่องกง', 'HKD', 'HKD=X', 'div', 'จุด'),
    ('^KS11', 'KOSPI', 'หุ้นเอเชีย', 'เกาหลีใต้', 'KRW', 'KRW=X', 'div', 'จุด'),
    ('^TWII', 'Taiwan Weighted', 'หุ้นเอเชีย', 'ไต้หวัน', 'TWD', 'TWD=X', 'div', 'จุด'),
    ('^BSESN', 'Sensex', 'หุ้นเอเชีย', 'อินเดีย', 'INR', 'INR=X', 'div', 'จุด'),
    ('^JKSE', 'Jakarta Composite', 'หุ้นเอเชีย', 'อินโดนีเซีย', 'IDR', 'IDR=X', 'div', 'จุด'),
    ('^STOXX50E', 'Euro Stoxx 50', 'หุ้นยุโรป', 'ยูโรโซน', 'EUR', 'EURUSD=X', 'mul', 'จุด'),
    ('^GDAXI', 'DAX', 'หุ้นยุโรป', 'เยอรมนี', 'EUR', 'EURUSD=X', 'mul', 'จุด'),
    ('^FTSE', 'FTSE 100', 'หุ้นยุโรป', 'อังกฤษ', 'GBP', 'GBPUSD=X', 'mul', 'จุด'),
    ('^BVSP', 'Bovespa', 'หุ้นอเมริกาใต้', 'บราซิล', 'BRL', 'BRL=X', 'div', 'จุด'),
    ('GC=F', 'ทองคำ', 'สินค้าโภคภัณฑ์', 'โลก', 'USD', '', '', 'ดอลลาร์/ออนซ์'),
    ('SI=F', 'เงิน (Silver)', 'สินค้าโภคภัณฑ์', 'โลก', 'USD', '', '', 'ดอลลาร์/ออนซ์'),
    ('CL=F', 'น้ำมันดิบ WTI', 'สินค้าโภคภัณฑ์', 'โลก', 'USD', '', '', 'ดอลลาร์/บาร์เรล'),
    ('BZ=F', 'น้ำมันดิบ Brent', 'สินค้าโภคภัณฑ์', 'โลก', 'USD', '', '', 'ดอลลาร์/บาร์เรล'),
    ('HG=F', 'ทองแดง', 'สินค้าโภคภัณฑ์', 'โลก', 'USD', '', '', 'ดอลลาร์/ปอนด์'),
    ('BTC-USD', 'Bitcoin', 'คริปโต', 'โลก', 'USD', '', '', 'ดอลลาร์'),
    ('ETH-USD', 'Ethereum', 'คริปโต', 'โลก', 'USD', '', '', 'ดอลลาร์'),
    ('TLT', 'พันธบัตรสหรัฐฯ ระยะยาว (TLT)', 'พันธบัตร', 'สหรัฐฯ', 'USD', '', '', 'ดอลลาร์'),
    ('DX-Y.NYB', 'ดัชนีดอลลาร์ (DXY)', 'ค่าเงินและดอกเบี้ย', 'สหรัฐฯ', 'USD', '', '', 'จุด'),
    ('^TNX', 'ยีลด์ US Treasury 10 ปี', 'ค่าเงินและดอกเบี้ย', 'สหรัฐฯ', 'USD', '', '', '%'),
    ('THB=X', 'USD/THB', 'ค่าเงินและดอกเบี้ย', 'ไทย', 'THB', '', '', 'บาท/ดอลลาร์'),
]
FX = sorted({m[5] for m in META if m[5]} - {m[0] for m in META})

def fetch(sym):
    p1 = int(dt.datetime.strptime(START, '%Y-%m-%d').replace(tzinfo=dt.timezone.utc).timestamp())
    p2 = int(time.time()) + 86400
    u = 'https://query1.finance.yahoo.com/v8/finance/chart/%s?period1=%d&period2=%d&interval=1d' % (urllib.request.quote(sym, safe=''), p1, p2)
    req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
    d = json.load(urllib.request.urlopen(req, timeout=30))
    res = d['chart']['result'][0]
    # Convert each bar's timestamp with the exchange's real time zone (handles daylight saving).
    # A fixed gmtoffset shifted summer bars of NY futures and London FX to the previous day.
    tz = ZoneInfo(res['meta'].get('exchangeTimezoneName') or 'UTC')
    closes = res['indicators']['quote'][0]['close']
    out = {}
    for t, c in zip(res.get('timestamp', []), closes):
        if c is None: continue
        local = dt.datetime.fromtimestamp(t, tz)
        if local.weekday() >= 5 and not sym.endswith('-USD'):
            continue  # stray weekend bar (crypto trades 7 days, everything else should not)
        out[local.strftime('%Y-%m-%d')] = round(c, 4)
    return out

path = os.path.join(F, 'assets_prices.csv')
SYMBOLS = {m[0] for m in META} | set(FX)
old = {}
if os.path.exists(path):
    for r in csv.DictReader(open(path, encoding='utf-8-sig')):
        if r['symbol'] in SYMBOLS:  # drop assets removed from META
            old[(r['symbol'], r['date'])] = (r['close'], r.get('source') or 'Yahoo Finance')
# Optional import of a daily history file "<symbol>_history.csv" with columns date,close (e.g. SET index from set.or.th)
for hp in [p for p in os.listdir(F) if p.endswith('_history_import.csv')]:
    sym = hp[:-len('_history_import.csv')].replace('_', '^', 1) if hp.startswith('_') else hp[:-len('_history_import.csv')]
    for line in open(os.path.join(F, hp), encoding='utf-8-sig'):
        p = line.strip().split(',')
        if len(p) >= 2 and p[0][:2] == '20':
            old.setdefault((sym, p[0]), (repr(float(p[1])), p[2] if len(p) > 2 else 'import'))
new, fail = dict(old), []
for sym in [m[0] for m in META] + FX:
    try:
        got = fetch(sym)
        # Yahoo has no history for ^SET.BK and returns only the latest close; keep it but never overwrite set.or.th rows
        for day, c in got.items():
            if sym == '^SET.BK' and (sym, day) in new:
                continue
            new[(sym, day)] = (repr(c), 'Yahoo Finance')
        print(sym, len(got), max(got) if got else '-')
    except Exception as e:
        fail.append(sym); print('FAIL', sym, e)
    time.sleep(0.4)
rows = sorted([[s, d, c, src] for (s, d), (c, src) in new.items()])
buf = io.StringIO(); w = csv.writer(buf, lineterminator='\r\n')
w.writerow(['symbol', 'date', 'close', 'source']); w.writerows(rows)
open(path, 'w', encoding='utf-8-sig', newline='').write(buf.getvalue())
buf = io.StringIO(); w = csv.writer(buf, lineterminator='\r\n')
w.writerow(['symbol', 'name', 'group', 'country', 'currency', 'fx_symbol', 'fx_mode', 'unit'])
w.writerows(META + [(s, s, 'อัตราแลกเปลี่ยน (ใช้คำนวณ)', '', '', '', '', '') for s in FX])
open(os.path.join(F, 'assets_meta.csv'), 'w', encoding='utf-8-sig', newline='').write(buf.getvalue())
print('rows', len(rows), 'failed', fail)
