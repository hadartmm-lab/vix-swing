import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title='VIX Swing Live', page_icon='📉', layout='centered')

st.markdown('''
<style>
.block-container{max-width:850px;padding-top:1.5rem}
.big-signal{font-size:2.2rem;font-weight:800;text-align:center;margin:.3rem 0 0}
.center{text-align:center}.muted{color:#8b949e}.metric-card{border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:14px}
</style>
''', unsafe_allow_html=True)

st.title('VIX Swing Live — 2–3 Days')
st.caption('נתונים אוטומטיים + מסנן סיכון קצר־טווח. VIX הוא מסנן סביבתי, לא טריגר כניסה עצמאי.')

@st.cache_data(ttl=300)
def load_data():
    tickers = ['^VIX','^VIX9D','^VIX3M','^GSPC','^NDX']
    data = yf.download(tickers, period='3mo', interval='1d', auto_adjust=False, progress=False, group_by='ticker', threads=True)
    out = {}
    for t in tickers:
        try:
            s = data[t]['Close'].dropna().astype(float)
        except Exception:
            s = pd.Series(dtype=float)
        out[t] = s
    return out

def ema(series, n):
    return float(series.ewm(span=n, adjust=False).mean().iloc[-1])

def rsi(series, n=14):
    d = series.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = -d.clip(upper=0).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    val = 100 - (100/(1+rs))
    x = val.iloc[-1]
    return float(x) if pd.notna(x) else 50.0

def pct_change(series, n):
    if len(series) <= n: return 0.0
    return float((series.iloc[-1]/series.iloc[-1-n]-1)*100)

try:
    d = load_data()
    vix, v9, v3m = d['^VIX'], d['^VIX9D'], d['^VIX3M']
    spx, ndx = d['^GSPC'], d['^NDX']
    if min(len(vix), len(v9), len(v3m), len(spx), len(ndx)) < 20:
        raise RuntimeError('לא התקבל מספיק מידע מהמקור החי.')
except Exception as e:
    st.error(f'לא הצלחתי למשוך נתוני שוק כרגע: {e}')
    st.stop()

market_name = st.radio('מדד לניתוח', ['Nasdaq 100','S&P 500'], horizontal=True)
market = ndx if market_name == 'Nasdaq 100' else spx

V = float(vix.iloc[-1]); V9 = float(v9.iloc[-1]); V3 = float(v3m.iloc[-1])
E5 = ema(vix,5); E10 = ema(vix,10); R = rsi(vix)
C1 = pct_change(vix,1); C5 = pct_change(vix,5)
M20 = ema(market,20); M = float(market.iloc[-1]); MC2 = pct_change(market,2)

# score: positive = risk-off / equity short bias
s=0.0
s += 1.2 if V>E5 else -1.2
s += 1.0 if E5>E10 else -1.0
if V9 > V*1.03: s += 1.4
elif V9 < V*0.97: s -= 1.0
if C1>=8: s+=1.4
elif C1>=3: s+=0.7
elif C1<=-8: s-=1.4
elif C1<=-3: s-=0.7
if C5>=10: s+=1.2
elif C5>=4: s+=0.7
elif C5<=-10: s-=1.2
elif C5<=-4: s-=0.7
# term structure using spot vs VIX3M proxy
ratio = V / V3 if V3 else 1.0
if ratio >= 1.03: s += 1.35
elif ratio <= 0.97: s -= 1.0
s += -1.0 if M>M20 else 1.0
s += -1.0 if MC2>0.35 else (1.0 if MC2<-0.35 else 0.0)
if R>=60: s+=0.35
elif R<=40: s-=0.35
if V>=30: s+=0.5
elif V>=25: s+=0.25
elif V<15: s-=0.2

if s>=3:
    signal='🔴 חיפוש SHORT'; regime='RISK-OFF'; summary='הפחד והמומנטום הקצר תומכים יותר בלחץ על שוק המניות.'
    action='חפש אישור שורט ב־1H–4H: שבירת תמיכה, lower high או כישלון פריצה. אל תרדוף אחרי נפילה שכבר התרחשה.'
    invalid='התרחיש נחלש אם VIX יורד חזרה מתחת EMA5/EMA10, VIX9D נרגע והמדד חוזר מעל EMA20.'
elif s<=-3:
    signal='🟢 חיפוש LONG'; regime='RISK-ON'; summary='הפחד הקצר נרגע והסביבה תומכת יותר במהלך חיובי של 1–3 ימים.'
    action='חפש אישור לונג ב־1H–4H: שמירת תמיכה, higher low או פריצה עם המשכיות.'
    invalid='התרחיש נחלש אם VIX חוזר מעל EMA5/EMA10, VIX9D מזנק והמדד נשבר מתחת EMA20.'
else:
    signal='🟡 המתנה'; regime='NEUTRAL'; summary='אין כרגע סנכרון מספיק לעסקה קצרה.'
    action='אל תכריח עסקה. חכה לסנכרון טוב יותר בין VIX, VIX9D וכיוון המדד.'
    invalid='—'

strength=abs(s)
quality='חזק מאוד' if strength>=5 else 'חזק' if strength>=4 else 'בינוני־חזק' if strength>=3 else 'בינוני' if strength>=1.5 else 'חלש'
fear='שקט מאוד' if V<15 else 'רגיל' if V<20 else 'זהירות' if V<25 else 'פחד גבוה' if V<30 else 'לחץ קיצוני'
trend='עולה ↑' if (V>E5 and E5>E10) else 'יורד ↓' if (V<E5 and E5<E10) else 'מעורב ↔'
term='Backwardation / לחץ' if ratio>=1.03 else 'Contango / רגוע' if ratio<=0.97 else 'שטוח / ניטרלי'

c1,c2,c3 = st.columns(3)
c1.metric('VIX', f'{V:.2f}', f'{C1:+.2f}%')
c2.metric('VIX9D', f'{V9:.2f}')
c3.metric('VIX3M', f'{V3:.2f}')

st.markdown(f'<div class="big-signal">{signal}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="center muted">{regime} · איכות סט־אפ: <b>{quality}</b> · ציון {s:.2f}</div>', unsafe_allow_html=True)
st.info(summary)

m1,m2 = st.columns(2)
with m1:
    st.metric('EMA5 VIX', f'{E5:.2f}')
    st.metric('EMA10 VIX', f'{E10:.2f}')
    st.metric('RSI VIX', f'{R:.1f}')
with m2:
    st.metric(f'{market_name}', f'{M:,.2f}', f'{MC2:+.2f}% ב־2 ימים')
    st.metric('EMA20 מדד', f'{M20:,.2f}')
    st.metric('VIX / VIX3M', f'{ratio:.3f}')

st.subheader('פירוש מהיר')
rows = pd.DataFrame({
    'בדיקה':['מצב פחד','מגמת VIX','מבנה טווח','המדד מול EMA20','כיוון עדיף'],
    'מצב':[fear,trend,term,'מעל' if M>M20 else 'מתחת',signal.replace('🟢 ','').replace('🔴 ','').replace('🟡 ','')]
})
st.dataframe(rows, hide_index=True, use_container_width=True)

st.subheader('תכנית פעולה')
st.write('**כניסה:**', action)
st.write('**ביטול התרחיש:**', invalid)
st.write('**יציאה:** לעסקת 2–3 ימים, חפש לקחת רווח חלקי ביום 1–2 ולבחון יציאה מלאה ביום 2–3, או מוקדם יותר אם VIX מתהפך נגד העסקה והמדד מאבד מומנטום.')

st.caption(f'עדכון אחרון באפליקציה: {datetime.now().strftime("%d/%m/%Y %H:%M")} · הנתונים מגיעים דרך Yahoo Finance/yfinance ועלולים להיות מעוכבים. לא ייעוץ השקעות.')

if st.button('🔄 רענן נתונים'):
    st.cache_data.clear(); st.rerun()
