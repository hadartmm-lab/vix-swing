import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="VIX Swing", page_icon="🎯", layout="centered")

st.markdown('''
<style>
:root{--bg:#07131f;--card:#0b1f31;--line:#184869;--txt:#f5f7fb;--muted:#b9c8d6}
html,body,[class*="css"]{background:var(--bg);color:var(--txt)}
.stApp{background:linear-gradient(180deg,#07131f 0%,#081725 100%)}
.block-container{max-width:820px;padding-top:.8rem;padding-bottom:2rem}
.hero{background:#0b1f31;border:1px solid var(--line);border-radius:20px;padding:16px 18px;margin-bottom:12px}
.hero-title{font-size:1.7rem;font-weight:900}.muted{color:var(--muted);font-size:.88rem}
.signal{background:#0b1f31;border:1px solid var(--line);border-radius:22px;padding:18px;margin:12px 0;text-align:center}
.signal-title{font-size:2rem;font-weight:950;margin-bottom:4px}.score{font-size:1rem;color:#dfe9f2}
.gauge-wrap{margin:20px 4px 6px;position:relative;padding-top:22px}
.gauge{height:18px;border-radius:999px;background:linear-gradient(90deg,#0f6b45 0%,#1a557c 28%,#746519 50%,#8a4d13 72%,#7b1d26 100%);border:1px solid rgba(255,255,255,.16);position:relative}
.pointer{position:absolute;top:0;transform:translateX(-50%);font-size:1.45rem;line-height:1}
.gauge-labels{display:flex;justify-content:space-between;font-size:.72rem;color:#d4e0ea;margin-top:6px}
.status-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:9px;margin-top:10px}
@media(max-width:640px){.status-grid{grid-template-columns:1fr}}
.status{background:#0a1a28;border:1px solid #173f5d;border-radius:15px;padding:12px 13px;display:flex;justify-content:space-between;align-items:center;gap:12px}
.status-name{font-weight:800;font-size:.92rem}.status-val{font-weight:900;text-align:left;white-space:nowrap}
.green{color:#72e8a7}.red{color:#ff8c92}.orange{color:#ffc06d}.blue{color:#82c9ff}.yellow{color:#ffe47b}.white{color:#fff}
.panel{background:#0b1f31;border:1px solid var(--line);border-radius:18px;padding:14px;margin-top:11px}
.stButton>button{width:100%;border-radius:14px;border:1px solid #1f6fa0;background:#0c2d46;color:white;font-weight:850;padding:.65rem 1rem}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,[data-testid="stRadio"] label,[data-testid="stWidgetLabel"] p{color:#f3f7fb!important}
</style>
''', unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_close(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        c = df["Close"] if "Close" in df.columns.get_level_values(0) else df.iloc[:, 0]
        if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    else:
        c = df["Close"]
    return c.astype(float).dropna()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlc(ticker, period="3mo"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        out = pd.DataFrame(index=df.index)
        for col in ["Open","High","Low","Close"]:
            x = df[col]
            if isinstance(x, pd.DataFrame): x = x.iloc[:,0]
            out[col] = x.astype(float)
        return out.dropna()
    return df[["Open","High","Low","Close"]].astype(float).dropna()

def ema_series(s,n): return s.ewm(span=n,adjust=False).mean()
def pctn(s,n): return float((s.iloc[-1]/s.iloc[-1-n]-1)*100) if s is not None and len(s)>n else np.nan

def rsi_series(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False).mean(); ad=dn.ewm(alpha=1/n,adjust=False).mean()
    rs=au/ad.replace(0,np.nan)
    return 100-100/(1+rs)

def resample_close(s,hours):
    if s is None or len(s)<20 or not isinstance(s.index,pd.DatetimeIndex): return None
    try: return s.resample(f"{hours}h").last().dropna()
    except Exception: return None

def detect_divergence(s,lookback=45,pivot=2):
    if s is None or len(s)<25: return "none"
    x=s.dropna().iloc[-lookback:]; r=rsi_series(x,14)
    lows=[]; highs=[]
    for i in range(pivot,len(x)-pivot):
        w=x.iloc[i-pivot:i+pivot+1]
        if x.iloc[i]==w.min(): lows.append(i)
        if x.iloc[i]==w.max(): highs.append(i)
    if len(lows)>=2:
        a,b=lows[-2],lows[-1]
        if pd.notna(r.iloc[a]) and pd.notna(r.iloc[b]) and x.iloc[b]<x.iloc[a] and r.iloc[b]>r.iloc[a]: return "bullish"
    if len(highs)>=2:
        a,b=highs[-2],highs[-1]
        if pd.notna(r.iloc[a]) and pd.notna(r.iloc[b]) and x.iloc[b]>x.iloc[a] and r.iloc[b]<r.iloc[a]: return "bearish"
    return "none"

def cross_status(s):
    e9=ema_series(s,9); e26=ema_series(s,26)
    cross=None; age=None
    start=max(1,len(s)-8)
    for i in range(start,len(s)):
        if e9.iloc[i-1] <= e26.iloc[i-1] and e9.iloc[i] > e26.iloc[i]: cross="golden"; age=len(s)-1-i
        if e9.iloc[i-1] >= e26.iloc[i-1] and e9.iloc[i] < e26.iloc[i]: cross="death"; age=len(s)-1-i
    gap_now=float((e9.iloc[-1]-e26.iloc[-1])/e26.iloc[-1]*100)
    gap_prev=float((e9.iloc[-4]-e26.iloc[-4])/e26.iloc[-4]*100) if len(s)>=4 else gap_now
    approaching=None
    if gap_now<0 and gap_now>gap_prev: approaching="golden"
    elif gap_now>0 and gap_now<gap_prev: approaching="death"
    # proximity: distance + closing speed. Confirmed/recent cross = 10.
    if cross and age is not None and age<=5:
        proximity=10
    else:
        dist=abs(gap_now)
        dist_score=max(1,min(9,9-int(dist/0.20)))
        closing=abs(gap_now)<abs(gap_prev)
        proximity=min(9,dist_score+(1 if closing else 0))
    return float(e9.iloc[-1]),float(e26.iloc[-1]),cross,age,approaching,int(proximity),gap_now

def support_resistance_signal(df,lookback=22,pivot=2):
    if df is None or len(df)<12: return "neutral",None,None
    x=df.iloc[-(lookback+5):].copy()
    # Exclude last two sessions from pivot construction to avoid using the current candle as its own level.
    ref=x.iloc[:-2]
    highs=[]; lows=[]
    for i in range(pivot,len(ref)-pivot):
        if ref["High"].iloc[i] == ref["High"].iloc[i-pivot:i+pivot+1].max(): highs.append((ref.index[i],float(ref["High"].iloc[i])))
        if ref["Low"].iloc[i] == ref["Low"].iloc[i-pivot:i+pivot+1].min(): lows.append((ref.index[i],float(ref["Low"].iloc[i])))
    resistance=highs[-1][1] if highs else float(ref["High"].max())
    support=lows[-1][1] if lows else float(ref["Low"].min())
    close=float(x["Close"].iloc[-1]); prev=float(x["Close"].iloc[-2]); high=float(x["High"].iloc[-1]); low=float(x["Low"].iloc[-1])
    if prev<=resistance and close>resistance*1.002: return "breakout",support,resistance
    if prev>=support and close<support*0.998: return "breakdown",support,resistance
    if high>=resistance*0.995 and close<resistance*0.998: return "reject_resistance",support,resistance
    if low<=support*1.005 and close>support*1.002: return "bounce_support",support,resistance
    return "neutral",support,resistance

def status_row(name,value,cls="white"):
    return f'<div class="status"><div class="status-name">{name}</div><div class="status-val {cls}">{value}</div></div>'

st.markdown('<div class="hero"><div class="hero-title">🎯 VIX Swing</div><div class="muted">כיוון QQQ / Nasdaq במבט אחד</div></div>',unsafe_allow_html=True)
if st.button("🔄 רענן",use_container_width=True): st.cache_data.clear(); st.rerun()

market_name=st.radio("מדד",["Nasdaq 100","S&P 500"],horizontal=True,label_visibility="collapsed")
market_ticker="^NDX" if market_name=="Nasdaq 100" else "^GSPC"

with st.spinner("מחשב..."):
    v=fetch_close("^VIX",period="6mo")
    v_intra=fetch_close("^VIX",period="60d",interval="60m")
    v9=fetch_close("^VIX9D",period="6mo")
    v3=fetch_close("^VIX3M",period="6mo")
    vv=fetch_close("^VVIX",period="6mo")
    m=fetch_close(market_ticker,period="6mo")
    mo=fetch_ohlc(market_ticker,period="3mo")

if any(x is None or len(x)<30 for x in [v,v9,v3,m]):
    st.error("לא הצלחתי למשוך כרגע את כל הנתונים. נסה רענון."); st.stop()

V=float(v.iloc[-1]); V9=float(v9.iloc[-1]); V3=float(v3.iloc[-1]); VV=float(vv.iloc[-1]) if vv is not None and len(vv) else np.nan
VE9,VE26,CROSS,CROSS_AGE,APPROACH,CROSS_NEAR,GAP=cross_status(v)
C1=pctn(v,1); C5=pctn(v,5); R9=V9/V; R3=V/V3
M=float(m.iloc[-1]); ME26=float(ema_series(m,26).iloc[-1]); M2=pctn(m,2); M5=pctn(m,5)
DIV4=detect_divergence(resample_close(v_intra,4),55,2)
DIV12=detect_divergence(resample_close(v_intra,12),45,2)
SR, SUPPORT, RESISTANCE=support_resistance_signal(mo,22,2)

score=0.0; reasons=[]
def add(p,t):
    global score
    score+=p; reasons.append((p,t))

# VIX EMA structure
add(1.2 if V>VE9 else -1.2, "VIX מעל EMA9" if V>VE9 else "VIX מתחת EMA9")
add(1.0 if VE9>VE26 else -1.0, "EMA9 מעל EMA26" if VE9>VE26 else "EMA9 מתחת EMA26")
if CROSS=="golden" and CROSS_AGE is not None and CROSS_AGE<=5: add(0.6,"Golden Cross טרי ב-VIX")
elif CROSS=="death" and CROSS_AGE is not None and CROSS_AGE<=5: add(-0.6,"Death Cross טרי ב-VIX")

# Short-volatility structure
if R9>=1.03: add(1.4,"VIX9D בפרמיה")
elif R9<=0.97: add(-1.0,"VIX9D נמוך מ-VIX")
if C1>=8: add(1.4,"VIX זינק ≥8% ביום")
elif C1>=3: add(0.7,"VIX עלה ≥3% ביום")
elif C1<=-8: add(-1.4,"VIX ירד ≥8% ביום")
elif C1<=-3: add(-0.7,"VIX ירד ≥3% ביום")
if C5>=10: add(1.2,"VIX עלה ≥10% ב-5 ימים")
elif C5>=4: add(0.7,"VIX עלה ≥4% ב-5 ימים")
elif C5<=-10: add(-1.2,"VIX ירד ≥10% ב-5 ימים")
elif C5<=-4: add(-0.7,"VIX ירד ≥4% ב-5 ימים")

# Term structure
if R3>=1.02: add(1.35,"Backwardation / לחץ")
elif R3<=0.94: add(-1.35,"Contango / רגיעה")

# Nasdaq / S&P structure: EMA26 + short momentum
add(1.0 if M<ME26 else -1.0, f"{market_name} מתחת EMA26" if M<ME26 else f"{market_name} מעל EMA26")
if M2<=-0.5: add(1.0,"מומנטום 2D שלילי")
elif M2>=0.5: add(-1.0,"מומנטום 2D חיובי")
if M5<=-1.0: add(0.6,"מומנטום 5D שלילי")
elif M5>=1.0: add(-0.6,"מומנטום 5D חיובי")

# RSI is NOT scored directly; only divergence is scored.
if DIV4=="bullish": add(0.45,"Bullish RSI Divergence ב-VIX 4H")
elif DIV4=="bearish": add(-0.45,"Bearish RSI Divergence ב-VIX 4H")
if DIV12=="bullish": add(0.75,"Bullish RSI Divergence ב-VIX 12H")
elif DIV12=="bearish": add(-0.75,"Bearish RSI Divergence ב-VIX 12H")
if DIV4==DIV12=="bullish": add(0.35,"סנכרון Divergence ל-SHORT")
elif DIV4==DIV12=="bearish": add(-0.35,"סנכרון Divergence ל-LONG")

# Support / resistance from about one month of daily market structure
if SR=="reject_resistance": add(0.5,"דחייה מהתנגדות")
elif SR=="breakout": add(-0.6,"פריצה מעל התנגדות")
elif SR=="bounce_support": add(-0.5,"תגובה מתמיכה")
elif SR=="breakdown": add(0.6,"שבירה מתחת לתמיכה")

# Modest VIX level adjustment
if V>=30: add(0.5,"VIX ≥30")
elif V>=25: add(0.25,"VIX ≥25")
elif V<15: add(-0.2,"VIX <15")

# Final threshold ±3.5
if score>=5.5: state,icon,cls="STRONG SHORT","🔴","red"
elif score>=3.5: state,icon,cls="SHORT","🟠","orange"
elif score<=-5.5: state,icon,cls="STRONG LONG","🟢","green"
elif score<=-3.5: state,icon,cls="LONG","🔵","blue"
else: state,icon,cls="WAIT","🟡","yellow"

# Gauge: -10 = far left (LONG), +10 = far right (SHORT)
pos=max(2,min(98,(score+10)/20*100))
st.markdown(f'''
<div class="signal">
  <div class="signal-title {cls}">{icon} {state}</div>
  <div class="score">Score <b>{score:+.2f}</b> · כניסה לכיוון רק מ-±3.5</div>
  <div class="gauge-wrap">
    <div class="pointer" style="left:{pos:.1f}%">▼</div>
    <div class="gauge"></div>
    <div class="gauge-labels"><span>STRONG LONG</span><span>WAIT</span><span>STRONG SHORT</span></div>
  </div>
</div>
''',unsafe_allow_html=True)

# Compact statuses
ema_bias="SHORT" if VE9>VE26 else "LONG"
ema_cls="red" if ema_bias=="SHORT" else "green"
if CROSS and CROSS_AGE is not None and CROSS_AGE<=5:
    cross_text=("Golden Cross" if CROSS=="golden" else "Death Cross")+" · 10/10"
    cross_cls="red" if CROSS=="golden" else "green"
elif APPROACH:
    cross_text=("Golden מתקרב" if APPROACH=="golden" else "Death מתקרב")+f" · {CROSS_NEAR}/10"
    cross_cls="orange" if APPROACH=="golden" else "blue"
else:
    cross_text=f"ללא חצייה · {CROSS_NEAR}/10"
    cross_cls="white"

def div_text(d):
    if d=="bullish": return "Bullish → SHORT","red"
    if d=="bearish": return "Bearish → LONG","green"
    return "אין","white"

def sr_text(s):
    return {
        "reject_resistance":("התנגדות → SHORT","red"),
        "breakout":("Breakout → LONG","green"),
        "bounce_support":("תמיכה → LONG","green"),
        "breakdown":("Breakdown → SHORT","red"),
        "neutral":("ניטרלי","white")
    }[s]

d4,d4c=div_text(DIV4); d12,d12c=div_text(DIV12); srt,src=sr_text(SR)
term_text="Risk-Off" if R3>=1.02 else ("Risk-On" if R3<=0.94 else "ניטרלי")
term_cls="red" if R3>=1.02 else ("green" if R3<=0.94 else "white")
market_text="מתחת EMA26" if M<ME26 else "מעל EMA26"; market_cls="red" if M<ME26 else "green"

st.markdown('<div class="status-grid">'+
    status_row("VIX · EMA9/26",ema_bias,ema_cls)+
    status_row("Cross · EMA9/26",cross_text,cross_cls)+
    status_row("Divergence 4H",d4,d4c)+
    status_row("Divergence 12H",d12,d12c)+
    status_row("Support / Resistance",srt,src)+
    status_row(f"{market_name} · EMA26",market_text,market_cls)+
    status_row("Term Structure",term_text,term_cls)+
    status_row("VIX9D / VIX",("לחץ" if R9>=1.03 else ("רגוע" if R9<=0.97 else "מאוזן")),"red" if R9>=1.03 else ("green" if R9<=0.97 else "white"))+
    '</div>',unsafe_allow_html=True)

# Minimal execution reminder
if "SHORT" in state: action="חפש טריגר SHORT ב-1H"
elif "LONG" in state: action="חפש טריגר LONG ב-1H"
else: action="אין עסקה — המתן לסנכרון"
st.markdown(f'<div class="panel" style="text-align:center;font-weight:900">{action}</div>',unsafe_allow_html=True)

with st.expander("פירוט החישוב"):
    st.write(f"VIX {V:.2f} · EMA9 {VE9:.2f} · EMA26 {VE26:.2f} · Cross proximity {CROSS_NEAR}/10")
    st.write(f"VIX 1D {C1:+.2f}% · 5D {C5:+.2f}% · VIX9D/VIX {R9:.3f} · VIX/VIX3M {R3:.3f}")
    st.write(f"{market_name}: {'מעל' if M>ME26 else 'מתחת'} EMA26 · 2D {M2:+.2f}% · 5D {M5:+.2f}%")
    st.write("RSI של VIX אינו מקבל ניקוד ישיר; הוא משמש רק לזיהוי Divergence ב-4H/12H.")
    for pts,txt in sorted(reasons,key=lambda z:abs(z[0]),reverse=True):
        st.write(f"**{pts:+.2f}** — {txt}")

st.caption(f"עודכן {datetime.now().strftime('%H:%M')} · נתונים: Yahoo Finance / yfinance · כלי מחקרי, לא ייעוץ השקעות")
