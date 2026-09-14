import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="VIX Swing Live", page_icon="📊", layout="centered")
st.markdown("""
<style>
.block-container{max-width:820px;padding-top:2rem}
.signal{padding:22px;border-radius:18px;text-align:center;margin:18px 0}
.strongshort{background:#ffdede}.shortwatch{background:#fff0df}
.wait{background:#fff7cf}.longwatch{background:#e6f5ff}.stronglong{background:#dcf7e7}
.signal h1{margin:0;font-size:42px}.signal p{margin:8px 0 0}
</style>""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def hist(ticker, period="6mo"):
    x = yf.download(ticker, period=period, interval="1d", auto_adjust=False,
                    progress=False, threads=False)
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    return x.dropna()

def close_series(df):
    c = df["Close"]
    if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
    return c.astype(float).dropna()

def rsi(s, n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-(100/(1+up/dn.replace(0,np.nan)))

def get_one(candidates):
    for t in candidates:
        try:
            d=hist(t)
            if len(d)>25: return t, close_series(d)
        except: pass
    return None, None

st.title("VIX Swing Live")
st.caption("מסנן סביבת שוק לעסקאות קצרות של 2–3 ימי מסחר. VIX הוא מסנן — לא טריגר כניסה עצמאי.")

if st.button("🔄 רענן נתונים", use_container_width=True):
    st.cache_data.clear(); st.rerun()

market_name=st.radio("מדד לניתוח",["Nasdaq 100","S&P 500"],horizontal=True)
market_ticker="^NDX" if market_name=="Nasdaq 100" else "^GSPC"

try:
    _,v=get_one(["^VIX"]); _,v9=get_one(["^VIX9D"])
    _,v3=get_one(["^VIX3M"]); _,vv=get_one(["^VVIX"])
    _,m=get_one([market_ticker])
    if any(x is None for x in [v,v9,v3,m]): raise ValueError("missing market series")

    V=float(v.iloc[-1]); V9=float(v9.iloc[-1]); V3=float(v3.iloc[-1])
    VV=float(vv.iloc[-1]) if vv is not None else np.nan
    e5=float(v.ewm(span=5,adjust=False).mean().iloc[-1])
    e10=float(v.ewm(span=10,adjust=False).mean().iloc[-1])
    vrsi=float(rsi(v).iloc[-1])
    c1=(V/float(v.iloc[-2])-1)*100
    c5=(V/float(v.iloc[-6])-1)*100
    me20=float(m.ewm(span=20,adjust=False).mean().iloc[-1])
    M=float(m.iloc[-1]); m2=(M/float(m.iloc[-3])-1)*100; m5=(M/float(m.iloc[-6])-1)*100
    ratio9=V9/V; ratio3=V/V3

    a,b,c=st.columns(3)
    a.metric("VIX",f"{V:.2f}",f"{c1:+.2f}%")
    b.metric("VIX9D",f"{V9:.2f}")
    c.metric("VIX3M",f"{V3:.2f}")
    st.caption(f"VVIX: {VV:.1f} · VIX EMA5: {e5:.2f} · EMA10: {e10:.2f} · RSI: {vrsi:.0f}")

    # Positive = risk-off / favors equity short.
    s=0.0; reasons=[]
    def add(points,text):
        nonlocal_dummy = None
        return points,text

    if V>e5: s+=1.2; reasons.append(("+1.20","VIX מעל EMA5"))
    else: s-=1.2; reasons.append(("-1.20","VIX מתחת EMA5"))
    if e5>e10: s+=1.0; reasons.append(("+1.00","EMA5 מעל EMA10"))
    else: s-=1.0; reasons.append(("-1.00","EMA5 מתחת EMA10"))

    if ratio9>=1.03: s+=1.4; reasons.append(("+1.40","VIX9D בפרמיה מעל VIX"))
    elif ratio9<=0.97: s-=1.0; reasons.append(("-1.00","VIX9D מתחת VIX"))
    else: reasons.append(("0.00","VIX9D/VIX ניטרלי"))

    if c1>=8: s+=1.4; reasons.append(("+1.40","זינוק VIX יומי ≥8%"))
    elif c1>=3: s+=0.7; reasons.append(("+0.70","VIX עולה ≥3% ביום"))
    elif c1<=-8: s-=1.4; reasons.append(("-1.40","VIX יורד ≥8% ביום"))
    elif c1<=-3: s-=0.7; reasons.append(("-0.70","VIX יורד ≥3% ביום"))

    if c5>=10: s+=1.2; reasons.append(("+1.20","VIX עלה ≥10% ב־5 ימים"))
    elif c5>=4: s+=0.7; reasons.append(("+0.70","VIX עלה ≥4% ב־5 ימים"))
    elif c5<=-10: s-=1.2; reasons.append(("-1.20","VIX ירד ≥10% ב־5 ימים"))
    elif c5<=-4: s-=0.7; reasons.append(("-0.70","VIX ירד ≥4% ב־5 ימים"))

    # Term structure: stronger confirmation only when the front end actually inverts.
    if ratio3>=1.02: s+=1.35; reasons.append(("+1.35","VIX מעל VIX3M — לחץ/Backwardation"))
    elif ratio3<=0.94: s-=0.45; reasons.append(("-0.45","Contango ברור — מוריד עוצמת SHORT"))

    if M<me20: s+=1.0; reasons.append(("+1.00",f"{market_name} מתחת EMA20"))
    else: s-=1.0; reasons.append(("-1.00",f"{market_name} מעל EMA20"))
    if m2<=-0.5: s+=1.0; reasons.append(("+1.00","מומנטום 2 ימים שלילי"))
    elif m2>=0.5: s-=1.0; reasons.append(("-1.00","מומנטום 2 ימים חיובי"))
    if m5<=-1.0: s+=0.6; reasons.append(("+0.60","מומנטום 5 ימים שלילי"))
    elif m5>=1.0: s-=0.6; reasons.append(("-0.60","מומנטום 5 ימים חיובי"))

    if vrsi>=60: s+=0.35; reasons.append(("+0.35","RSI VIX תומך בעלייה"))
    elif vrsi<=40: s-=0.35; reasons.append(("-0.35","RSI VIX תומך בירידה"))

    # VVIX is confirmation, not a standalone trigger.
    if not np.isnan(VV):
        if VV>=115: s+=0.55; reasons.append(("+0.55","VVIX גבוה — אי־ודאות גבוהה"))
        elif VV<85: s-=0.25; reasons.append(("-0.25","VVIX נמוך"))

    # Five-state mapping. Strong states require both score AND structural confirmation.
    short_struct = (ratio9>=1.0 and ratio3>=0.98 and V>e5 and M<me20)
    long_struct  = (ratio9<=1.0 and ratio3<=0.96 and V<e5 and M>me20)

    if s>=5.0 and short_struct:
        label="🔴 STRONG SHORT"; css="strongshort"
        note="סביבת Risk-Off מסונכרנת. עדיין נכנסים רק אחרי טריגר מחיר."
    elif s>=2.5:
        label="🟠 SHORT WATCH"; css="shortwatch"
        note="יש נטייה לשורט, אבל חסר סנכרון מלא. חפש אישור — לא כניסה אוטומטית."
    elif s<=-5.0 and long_struct:
        label="🟢 STRONG LONG"; css="stronglong"
        note="סביבת Risk-On מסונכרנת. עדיין נכנסים רק אחרי טריגר מחיר."
    elif s<=-2.5:
        label="🔵 LONG WATCH"; css="longwatch"
        note="יש נטייה ללונג, אבל חסר סנכרון מלא. חפש אישור — לא כניסה אוטומטית."
    else:
        label="🟡 WAIT"; css="wait"
        note="אין כרגע יתרון מספיק ברור לעסקה קצרה."

    st.markdown(f'<div class="signal {css}"><h1>{label}</h1><p>ציון: {s:.2f} · {note}</p></div>',unsafe_allow_html=True)

    st.subheader("מה המערכת רואה עכשיו")
    st.write(f"**VIX9D / VIX:** {ratio9:.3f}")
    st.write(f"**VIX / VIX3M:** {ratio3:.3f}")
    st.write(f"**שינוי VIX ב־5 ימים:** {c5:+.2f}%")
    st.write(f"**{market_name} מול EMA20:** {'מתחת' if M<me20 else 'מעל'}")
    st.write(f"**מומנטום 2 ימים:** {m2:+.2f}%")
    st.write(f"**מומנטום 5 ימים:** {m5:+.2f}%")

    with st.expander("פירוט הציון"):
        for pts,txt in reasons: st.write(f"**{pts}** — {txt}")

    st.subheader("פרוטוקול כניסה ל־2–3 ימים")
    if "SHORT" in label:
        st.markdown("""1. ה־VIX נותן **SHORT bias**, לא כניסה.
2. ב־4H–1H חפש שבירת תמיכה / Lower High / כישלון פריצה.
3. אשר ב־1H וב־15m שהנרות והמומנטום באותו כיוון.
4. אם המדד חוזר מעל EMA20 וה־VIX מאבד EMA5 — הסט־אפ נחלש.
5. בעסקה קצרה: נהל רווח בתוך 1–3 ימים; אל תהפוך אותה אוטומטית לעסקת טווח ארוך.""")
    elif "LONG" in label:
        st.markdown("""1. ה־VIX נותן **LONG bias**, לא כניסה.
2. ב־4H–1H חפש שמירת תמיכה / Higher Low / פריצה איכותית.
3. אשר ב־1H וב־15m שהנרות והמומנטום באותו כיוון.
4. אם המדד נשבר מתחת EMA20 וה־VIX חוזר מעל EMA5 — הסט־אפ נחלש.
5. בעסקה קצרה: נהל רווח בתוך 1–3 ימים; אל תהפוך אותה אוטומטית לעסקת טווח ארוך.""")
    else:
        st.info("WAIT: לא להכריח עסקה. חכה לסנכרון טוב יותר בין מבנה ה־VIX לבין המדד.")

    st.caption("המערכת היא מסנן הסתברותי/טכני ואינה מבטיחה תשואה או כיוון שוק.")
except Exception as e:
    st.error("לא הצלחתי למשוך את כל נתוני השוק כרגע. נסה רענון בעוד רגע.")
    st.caption(str(e))
