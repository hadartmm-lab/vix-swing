
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="VIX Swing Live", page_icon="🎮", layout="centered")

st.markdown('''
<style>
:root{
  --bg:#07131f; --card:#0b1f31; --line:#184869; --txt:#f3f7fb; --muted:#d5e0ea;
}
html, body, [class*="css"] {background:var(--bg); color:var(--txt);}
.stApp{background:linear-gradient(180deg,#07131f 0%,#081725 100%);}
.block-container{max-width:840px;padding-top:1rem;padding-bottom:2rem;}
h1,h2,h3{color:var(--txt)!important;}
.small-muted{color:#dce7f0;font-size:.9rem}
.hero{background:linear-gradient(135deg,#0b1d2e,#102941);border:1px solid var(--line);border-radius:22px;padding:18px;margin-bottom:16px;box-shadow:0 10px 30px rgba(0,0,0,.25)}
.signal{border-radius:22px;padding:22px 18px;text-align:center;border:1px solid rgba(255,255,255,.08);margin:12px 0 14px}
.strongshort{background:linear-gradient(135deg,#4a1218,#2a1115)}
.shortwatch{background:linear-gradient(135deg,#4a2b0b,#2a1d0d)}
.wait{background:linear-gradient(135deg,#4a430d,#2a270d)}
.longwatch{background:linear-gradient(135deg,#0b304a,#0d2230)}
.stronglong{background:linear-gradient(135deg,#0b4328,#0d2b20)}
.signal-title{font-size:2.35rem;font-weight:900;line-height:1.05}
.signal-sub{margin-top:8px;color:#e7edf4}
.meter{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-top:14px}
.meter > div{padding:8px 4px;border-radius:12px;text-align:center;font-size:.72rem;border:1px solid rgba(255,255,255,.06);opacity:.6;background:#0a1a28}
.active{opacity:1!important;box-shadow:0 0 18px rgba(255,255,255,.12)}
.metric-card{background:linear-gradient(180deg,#0b1f31,#0a1a28);border:1px solid var(--line);border-radius:18px;padding:14px;min-height:142px;margin-bottom:10px}
.metric-name{color:#ffffff;font-weight:800;font-size:.95rem}
.metric-val{font-size:1.85rem;font-weight:900;margin:6px 0;color:#ffffff}
.metric-desc{color:#dce7f0;font-size:.84rem;line-height:1.4}
.tag{display:inline-block;padding:4px 9px;border-radius:999px;font-size:.75rem;font-weight:800;margin-top:6px}
.tag-green{background:#103d2a;color:#6ff0aa}.tag-blue{background:#10344d;color:#74caff}.tag-yellow{background:#4b4015;color:#ffe077}.tag-orange{background:#4b2d11;color:#ffb66b}.tag-red{background:#4a171b;color:#ff858b}
.panel{background:linear-gradient(180deg,#0b1f31,#0a1a28);border:1px solid var(--line);border-radius:18px;padding:16px;margin:12px 0}
.checkrow{padding:7px 0;border-bottom:1px solid rgba(255,255,255,.06)}
.scorebox{display:flex;justify-content:space-between;gap:12px;align-items:center;background:#091827;border:1px solid var(--line);border-radius:16px;padding:14px}
.scorepill{min-width:90px;text-align:center;padding:10px;border-radius:14px;background:#0f2740;font-size:1.25rem;font-weight:900}
.stButton>button{width:100%;border-radius:14px;border:1px solid #1f6fa0;background:#0c2d46;color:white;font-weight:800;padding:.7rem 1rem}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stRadio"] label,
[data-testid="stWidgetLabel"] p,
.stRadio label,
.stRadio p {color:#f3f7fb !important;}
[data-testid="stCaptionContainer"] p {color:#c9d7e3 !important;}
.checkrow, .scorebox, .panel {color:#f3f7fb !important;}
.hero {color:#ffffff !important;}
.signal-title, .signal-sub {color:#ffffff !important;}

</style>
''', unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_series(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            c = df["Close"]
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
        else:
            c = df.iloc[:, 0]
    else:
        c = df["Close"]
    return c.astype(float).dropna()

def ema(s, n):
    return float(s.ewm(span=n, adjust=False).mean().iloc[-1])

def pctn(s, n):
    return float((s.iloc[-1] / s.iloc[-1-n] - 1) * 100) if s is not None and len(s) > n else np.nan

def calc_rsi_series(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False).mean()
    ad = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - 100/(1+rs)

def calc_rsi(s, n=14):
    r = calc_rsi_series(s, n).dropna()
    return float(r.iloc[-1]) if len(r) else np.nan

def resample_close(s, hours):
    if s is None or len(s) < 20:
        return None
    x = s.copy()
    if not isinstance(x.index, pd.DatetimeIndex):
        return None
    try:
        return x.resample(f"{hours}h").last().dropna()
    except Exception:
        return None

def detect_rsi_divergence(s, lookback=45, pivot=2):
    """Bullish: VIX lower low + RSI higher low. Bearish: VIX higher high + RSI lower high."""
    if s is None or len(s) < 25:
        return "none", "אין מספיק נתונים"
    x = s.dropna().iloc[-lookback:]
    r = calc_rsi_series(x, 14)
    lows, highs = [], []
    for i in range(pivot, len(x)-pivot):
        win = x.iloc[i-pivot:i+pivot+1]
        if x.iloc[i] == win.min(): lows.append(i)
        if x.iloc[i] == win.max(): highs.append(i)
    if len(lows) >= 2:
        a,b = lows[-2],lows[-1]
        if pd.notna(r.iloc[a]) and pd.notna(r.iloc[b]) and x.iloc[b] < x.iloc[a] and r.iloc[b] > r.iloc[a]:
            return "bullish", f"VIX LL {x.iloc[a]:.1f}→{x.iloc[b]:.1f} | RSI HL {r.iloc[a]:.0f}→{r.iloc[b]:.0f}"
    if len(highs) >= 2:
        a,b = highs[-2],highs[-1]
        if pd.notna(r.iloc[a]) and pd.notna(r.iloc[b]) and x.iloc[b] > x.iloc[a] and r.iloc[b] < r.iloc[a]:
            return "bearish", f"VIX HH {x.iloc[a]:.1f}→{x.iloc[b]:.1f} | RSI LH {r.iloc[a]:.0f}→{r.iloc[b]:.0f}"
    return "none", "אין Divergence מאושר כרגע"

def badge(text, cls):
    return f'<span class="tag {cls}">{text}</span>'

st.markdown('<div class="hero"><div style="font-size:2rem;font-weight:900">🎮 VIX Swing Live</div><div class="small-muted">משחק פשוט של Risk-On / Risk-Off לעסקאות קצרות של 2–3 ימי מסחר</div></div>', unsafe_allow_html=True)

if st.button("🔄 רענן נתונים", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

market_name = st.radio("מדד לניתוח", ["Nasdaq 100", "S&P 500"], horizontal=True)
market_ticker = "^NDX" if market_name == "Nasdaq 100" else "^GSPC"

with st.spinner("טוען נתוני שוק..."):
    v = fetch_series("^VIX")
    v_intra = fetch_series("^VIX", period="60d", interval="60m")
    v9 = fetch_series("^VIX9D")
    v3 = fetch_series("^VIX3M")
    vv = fetch_series("^VVIX")
    m = fetch_series(market_ticker)

if any(x is None or len(x) < 25 for x in [v, v9, v3, m]):
    st.error("לא הצלחתי למשוך כרגע את כל הנתונים הדרושים. נסה רענון בעוד רגע.")
    st.stop()

V=float(v.iloc[-1]); V9=float(v9.iloc[-1]); V3=float(v3.iloc[-1])
VV=float(vv.iloc[-1]) if vv is not None and len(vv) else np.nan
VE5=ema(v,5); VE10=ema(v,10); VRSI=calc_rsi(v)
C1=pctn(v,1); C5=pctn(v,5)
M=float(m.iloc[-1]); ME5=ema(m,5); ME20=ema(m,20); M2=pctn(m,2); M5=pctn(m,5)
R9=V9/V; R3=V/V3

# RSI divergence on VIX — 4H and 12H, derived from hourly VIX observations.
v4 = resample_close(v_intra, 4)
v12 = resample_close(v_intra, 12)
DIV4, DIV4_DETAIL = detect_rsi_divergence(v4, lookback=55, pivot=2)
DIV12, DIV12_DETAIL = detect_rsi_divergence(v12, lookback=45, pivot=2)

score=0.0
reasons=[]
def add(points, text):
    global score
    score += points
    reasons.append((points, text))

if V>VE5: add(1.2,"VIX מעל EMA5")
else: add(-1.2,"VIX מתחת EMA5")
if VE5>VE10: add(1.0,"EMA5 של VIX מעל EMA10")
else: add(-1.0,"EMA5 של VIX מתחת EMA10")

if R9>=1.03: add(1.4,"VIX9D בפרמיה — לחץ קצר־טווח")
elif R9<=0.97: add(-1.0,"VIX9D נמוך מ־VIX")
else: reasons.append((0.0,"VIX9D/VIX מאוזן"))

if C1>=8: add(1.4,"זינוק יומי ב־VIX ≥ 8%")
elif C1>=3: add(0.7,"VIX עולה ≥ 3% ביום")
elif C1<=-8: add(-1.4,"ירידה יומית ב־VIX ≥ 8%")
elif C1<=-3: add(-0.7,"VIX יורד ≥ 3% ביום")

if C5>=10: add(1.2,"VIX עלה ≥ 10% ב־5 ימים")
elif C5>=4: add(0.7,"VIX עלה ≥ 4% ב־5 ימים")
elif C5<=-10: add(-1.2,"VIX ירד ≥ 10% ב־5 ימים")
elif C5<=-4: add(-0.7,"VIX ירד ≥ 4% ב־5 ימים")

if R3>=1.02: add(1.35,"VIX מעל VIX3M — לחץ חזק")
elif R3<=0.94: add(-0.45,"Contango ברור — שוק פחות לחוץ")
else: reasons.append((0.0,"VIX/VIX3M באמצע"))

if M<ME20: add(1.0,f"{market_name} מתחת EMA20")
else: add(-1.0,f"{market_name} מעל EMA20")
if M<ME5: add(0.45,f"{market_name} מתחת EMA5")
else: add(-0.45,f"{market_name} מעל EMA5")

if M2<=-0.5: add(1.0,"מומנטום 2 ימים שלילי")
elif M2>=0.5: add(-1.0,"מומנטום 2 ימים חיובי")
if M5<=-1.0: add(0.6,"מומנטום 5 ימים שלילי")
elif M5>=1.0: add(-0.6,"מומנטום 5 ימים חיובי")

if VRSI>=60: add(0.35,"RSI VIX תומך בעלייה")
elif VRSI<=40: add(-0.35,"RSI VIX תומך בירידה")

# RSI Divergence: positive score = VIX likely up = equity SHORT bias.
if DIV4 == "bullish": add(0.45,"Bullish RSI Divergence ב־VIX 4H → מחזק SHORT")
elif DIV4 == "bearish": add(-0.45,"Bearish RSI Divergence ב־VIX 4H → מחזק LONG")
else: reasons.append((0.0,"אין RSI Divergence מאושר ב־VIX 4H"))

if DIV12 == "bullish": add(0.75,"Bullish RSI Divergence ב־VIX 12H → מחזק SHORT")
elif DIV12 == "bearish": add(-0.75,"Bearish RSI Divergence ב־VIX 12H → מחזק LONG")
else: reasons.append((0.0,"אין RSI Divergence מאושר ב־VIX 12H"))

if DIV4 == DIV12 == "bullish":
    add(0.35,"סנכרון Divergence שורי ב־4H+12H → בונוס SHORT")
elif DIV4 == DIV12 == "bearish":
    add(-0.35,"סנכרון Divergence דובי ב־4H+12H → בונוס LONG")

if not np.isnan(VV):
    if VV>=115: add(0.55,"VVIX גבוה — אי־ודאות גבוהה")
    elif VV<85: add(-0.25,"VVIX נמוך")

short_struct=(R9>=1.00 and R3>=0.98 and V>VE5 and M<ME20)
long_struct=(R9<=1.00 and R3<=0.96 and V<VE5 and M>ME20)

if score>=5.0 and short_struct:
    state,icon,css,idx,note="STRONG SHORT","🔴","strongshort",4,"Risk-Off מסונכרן. עדיין צריך טריגר מחיר."
elif score>=2.5:
    state,icon,css,idx,note="SHORT WATCH","🟠","shortwatch",3,"נטייה לשורט, אבל חסר סנכרון מלא."
elif score<=-5.0 and long_struct:
    state,icon,css,idx,note="STRONG LONG","🟢","stronglong",0,"Risk-On מסונכרן. עדיין צריך טריגר מחיר."
elif score<=-2.5:
    state,icon,css,idx,note="LONG WATCH","🔵","longwatch",1,"נטייה ללונג, אבל חסר סנכרון מלא."
else:
    state,icon,css,idx,note="WAIT","🟡","wait",2,"אין יתרון מספיק ברור."

states=["STRONG LONG","LONG WATCH","WAIT","SHORT WATCH","STRONG SHORT"]
classes=["tag-green","tag-blue","tag-yellow","tag-orange","tag-red"]
meter='<div class="meter">'
for i,s in enumerate(states):
    active=" active" if i==idx else ""
    meter += f'<div class="{active}"><span class="tag {classes[i]}">{s}</span></div>'
meter += '</div>'

st.markdown(f'''
<div class="signal {css}">
  <div class="signal-title">{icon} {state}</div>
  <div class="signal-sub">ציון: <b>{score:.2f}</b> · {note}</div>
  {meter}
</div>
''', unsafe_allow_html=True)

def metric_card(name, value, desc, btext, bcls):
    st.markdown(f'''
    <div class="metric-card">
      <div class="metric-name">{name}</div>
      <div class="metric-val">{value}</div>
      {badge(btext,bcls)}
      <div class="metric-desc" style="margin-top:8px">{desc}</div>
    </div>
    ''', unsafe_allow_html=True)

c1,c2,c3=st.columns(3)
with c1:
    metric_card("VIX",f"{V:.2f}","מדד הפחד ל־30 יום. עלייה חדה = יותר לחץ בשוק.",f"{C1:+.2f}% היום","tag-red" if C1>0 else "tag-green")
with c2:
    metric_card("VIX9D",f"{V9:.2f}","ציפיות תנודתיות ל־9 ימים. חשוב במיוחד לעסקאות קצרות.","קצר־טווח","tag-blue")
with c3:
    metric_card("VIX3M",f"{V3:.2f}","ציפיות תנודתיות לכ־3 חודשים. משמש להשוואת מבנה הפחד.","טווח בינוני","tag-blue")

c4,c5,c6=st.columns(3)
with c4:
    vv_state="גבוה" if not np.isnan(VV) and VV>=115 else "רגיל"
    metric_card("VVIX",f"{VV:.1f}" if not np.isnan(VV) else "—","התנודתיות של ה־VIX עצמו. גבוה = שוק אופציות עצבני יותר.",vv_state,"tag-red" if vv_state=="גבוה" else "tag-green")
with c5:
    txt="לחץ קצר עולה" if R9>1.03 else ("רגוע" if R9<0.97 else "מאוזן")
    cls="tag-red" if R9>1.03 else ("tag-green" if R9<0.97 else "tag-blue")
    metric_card("VIX9D / VIX",f"{R9:.3f}","מעל 1 = פחד קצר־טווח חזק יותר. מתחת 1 = לחץ קצר מתון יותר.",txt,cls)
with c6:
    txt="לחץ חזק" if R3>=1.0 else ("רגיל" if R3<0.94 else "מתקרב ללחץ")
    cls="tag-red" if R3>=1.0 else ("tag-green" if R3<0.94 else "tag-orange")
    metric_card("VIX / VIX3M",f"{R3:.3f}","קרוב ל־1 ומעלה = עקום פחד מתוח. נמוך משמעותית מ־1 = Contango רגיל.",txt,cls)

c7,c8,c9=st.columns(3)
with c7:
    metric_card("שינוי VIX (5 ימים)",f"{C5:+.2f}%","כמה הפחד השתנה בחמשת ימי המסחר האחרונים.","עולה" if C5>0 else "יורד","tag-red" if C5>0 else "tag-green")
with c8:
    metric_card(f"{market_name} / EMA20","מתחת" if M<ME20 else "מעל","מתחת EMA20 = חולשה קצרה. מעל EMA20 = מבנה חזק יותר.","חלש" if M<ME20 else "חזק","tag-red" if M<ME20 else "tag-green")
with c9:
    metric_card("מומנטום המדד",f"2D {M2:+.2f}% · 5D {M5:+.2f}%","בודק אם המדד באמת נע באותו כיוון שה־VIX מרמז.","שלילי" if M5<0 else "חיובי","tag-red" if M5<0 else "tag-green")

def div_ui(div):
    if div == "bullish":
        return "Bullish", "VIX עשוי לעלות → מחזק SHORT במדד", "tag-red"
    if div == "bearish":
        return "Bearish", "VIX עשוי לרדת → מחזק LONG במדד", "tag-green"
    return "אין", "אין דיברג'נס מאושר כרגע", "tag-blue"

d4_txt,d4_desc,d4_cls = div_ui(DIV4)
d12_txt,d12_desc,d12_cls = div_ui(DIV12)
sync = DIV4 == DIV12 and DIV4 in ("bullish","bearish")
cs1,cs2,cs3=st.columns(3)
with cs1:
    metric_card("RSI Divergence · VIX 4H", d4_txt, DIV4_DETAIL, d4_desc, d4_cls)
with cs2:
    metric_card("RSI Divergence · VIX 12H", d12_txt, DIV12_DETAIL, d12_desc, d12_cls)
with cs3:
    sync_txt = "מסונכרן" if sync else ("סותר" if DIV4 != "none" and DIV12 != "none" and DIV4 != DIV12 else "ללא סנכרון")
    sync_desc = "4H ו־12H באותו כיוון — מתקבל בונוס נוסף בציון." if sync else "אין כרגע בונוס סנכרון בין הטיימפריימים."
    sync_bias = ("+0.35 SHORT" if DIV4 == "bullish" else "−0.35 LONG") if sync else "0.00"
    metric_card("סנכרון Divergence", sync_txt, sync_desc, sync_bias, "tag-orange" if sync else "tag-blue")

st.markdown('<div class="panel"><h3>🎯 מה חסר כדי לעלות שלב?</h3>', unsafe_allow_html=True)

if "SHORT" in state:
    tasks=[
        ("VIX9D/VIX ≥ 1.00",R9>=1.00,f"כרגע {R9:.3f}"),
        ("VIX/VIX3M ≥ 0.98",R3>=0.98,f"כרגע {R3:.3f}"),
        (f"{market_name} מתחת EMA20",M<ME20,"כן" if M<ME20 else "לא"),
        ("מומנטום 2 ימים שלילי",M2<0,f"כרגע {M2:+.2f}%"),
        ("VIX מעל EMA5",V>VE5,f"{V:.2f} מול {VE5:.2f}"),
        ("Divergence VIX תומך SHORT (4H/12H)",DIV4=="bullish" or DIV12=="bullish",f"4H {d4_txt} · 12H {d12_txt}")
    ]; goal="STRONG SHORT"
elif "LONG" in state:
    tasks=[
        ("VIX9D/VIX ≤ 1.00",R9<=1.00,f"כרגע {R9:.3f}"),
        ("VIX/VIX3M ≤ 0.96",R3<=0.96,f"כרגע {R3:.3f}"),
        (f"{market_name} מעל EMA20",M>ME20,"כן" if M>ME20 else "לא"),
        ("מומנטום 2 ימים חיובי",M2>0,f"כרגע {M2:+.2f}%"),
        ("VIX מתחת EMA5",V<VE5,f"{V:.2f} מול {VE5:.2f}"),
        ("Divergence VIX תומך LONG (4H/12H)",DIV4=="bearish" or DIV12=="bearish",f"4H {d4_txt} · 12H {d12_txt}")
    ]; goal="STRONG LONG"
else:
    tasks=[
        ("כיוון VIX ברור",abs(C1)>=3,f"{C1:+.2f}%"),
        ("מומנטום 2 ימים משמעותי",abs(M2)>=0.5,f"{M2:+.2f}%"),
        ("מיקום מדד מול EMA20",True,"מעל" if M>ME20 else "מתחת")
    ]; goal="WATCH"

done=sum(1 for _,ok,_ in tasks if ok)
for title,ok,current in tasks:
    st.markdown(f'<div class="checkrow">{"✅" if ok else "➖"} <b>{title}</b> <span class="small-muted">— {current}</span></div>', unsafe_allow_html=True)

st.markdown(f'<div class="scorebox" style="margin-top:12px"><div><b>התקדמות ל־{goal}</b><div class="small-muted">{done} מתוך {len(tasks)} תנאים</div></div><div class="scorepill">{done}/{len(tasks)}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="panel"><h3>💡 מה זה אומר בפועל?</h3>', unsafe_allow_html=True)
if "SHORT" in state:
    st.write("יש לחץ עולה בשוק. עכשיו לא נכנסים בגלל ה־VIX בלבד — מחפשים ב־4H/1H שבירת תמיכה, Lower High או כישלון פריצה.")
    st.write("אישור סופי: גרף 1H צריך לתמוך בירידה.")
elif "LONG" in state:
    st.write("הפחד נרגע והשוק מקבל סביבה תומכת יותר. מחפשים ב־4H/1H שמירת תמיכה, Higher Low או פריצה איכותית.")
    st.write("אישור סופי: גרף 1H צריך לתמוך בעלייה.")
else:
    st.write("אין כרגע יתרון ברור. המטרה היא לא להכריח עסקה — מחכים שה־VIX והמדד יסתנכרנו.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('''<div class="panel"><h3>🧮 משקל ה־Divergence במשוואה</h3>
<div class="small-muted">
<b>4H:</b> Bullish +0.45 / Bearish −0.45 ·
<b>12H:</b> Bullish +0.75 / Bearish −0.75 ·
<b>אותו כיוון בשניהם:</b> בונוס נוסף +0.35 או −0.35.<br>
ציון חיובי מחזק Risk-Off / SHORT במדד; ציון שלילי מחזק Risk-On / LONG במדד. Divergence הוא אישור — לא טריגר כניסה עצמאי.
</div></div>''', unsafe_allow_html=True)

with st.expander("📊 פירוט הציון"):
    for pts,txt in sorted(reasons,key=lambda x: abs(x[0]),reverse=True):
        st.write(f"{'🔴' if pts>0 else ('🟢' if pts<0 else '⚪')} **{pts:+.2f}** — {txt}")

st.caption(f"עודכן: {datetime.now().strftime('%H:%M')} · נתונים דרך Yahoo Finance / yfinance · כלי סינון מחקרי, לא ייעוץ השקעות.")
