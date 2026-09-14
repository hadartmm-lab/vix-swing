
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="VIX Swing Live", page_icon="📈", layout="centered")

# ---------- UI ----------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 760px;}
h1 {font-size: 2.4rem !important; line-height: 1.05;}
.signal {
    padding: 18px; border-radius: 18px; text-align: center;
    margin: 14px 0 10px 0; border: 1px solid rgba(120,120,120,.25);
}
.signal-long {background: rgba(31, 180, 95, .12);}
.signal-short {background: rgba(230, 70, 70, .12);}
.signal-wait {background: rgba(235, 180, 25, .12);}
.big {font-size: 2.25rem; font-weight: 850; margin: 0;}
.small {opacity: .72; font-size: .92rem;}
.rule {padding: 10px 0; border-top: 1px solid rgba(120,120,120,.18);}
</style>
""", unsafe_allow_html=True)

st.title("VIX Swing Live")
st.caption("מסנן סביבת שוק לעסקאות קצרות של 2–3 ימי מסחר. ה‑VIX הוא מסנן כיוון וסיכון — לא טריגר כניסה עצמאי.")

# ---------- Helpers ----------
def rsi(series: pd.Series, period: int = 14) -> float:
    s = series.dropna()
    if len(s) < period + 2:
        return np.nan
    d = s.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    avg_up = up.ewm(alpha=1/period, adjust=False).mean()
    avg_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    return float(value.iloc[-1])

def pct_change_n(s: pd.Series, n: int) -> float:
    s = s.dropna()
    if len(s) <= n:
        return np.nan
    return float((s.iloc[-1] / s.iloc[-1-n] - 1) * 100)

def latest(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1])

def ema(s: pd.Series, n: int) -> float:
    return float(s.dropna().ewm(span=n, adjust=False).mean().iloc[-1])

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data():
    tickers = {
        "VIX": "^VIX",
        "VIX9D": "^VIX9D",
        "VIX3M": "^VIX3M",
        "VVIX": "^VVIX",
        "NASDAQ": "^NDX",
        "SP500": "^GSPC",
    }
    out = {}
    for k, t in tickers.items():
        df = yf.download(t, period="6mo", interval="1d", progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            out[k] = None
            continue
        # yfinance may return MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"][t] if t in df["Close"].columns else df["Close"].iloc[:, 0]
        else:
            close = df["Close"]
        out[k] = close.dropna()
    return out

if st.button("🔄 רענן נתונים", use_container_width=True):
    st.cache_data.clear()

with st.spinner("מושך נתוני שוק..."):
    data = fetch_data()

required = ["VIX", "VIX9D", "VIX3M", "NASDAQ", "SP500"]
missing = [k for k in required if data.get(k) is None or len(data[k]) < 25]
if missing:
    st.error("לא הצלחתי לקבל כרגע את כל הנתונים הדרושים: " + ", ".join(missing))
    st.stop()

market_choice = st.radio("מדד לניתוח", ["Nasdaq 100", "S&P 500"], horizontal=True)
mkt_key = "NASDAQ" if market_choice == "Nasdaq 100" else "SP500"

vix_s = data["VIX"]
v9_s = data["VIX9D"]
v3m_s = data["VIX3M"]
vvix_s = data.get("VVIX")
mkt_s = data[mkt_key]

vix = latest(vix_s)
v9 = latest(v9_s)
v3m = latest(v3m_s)
vvix = latest(vvix_s) if vvix_s is not None and len(vvix_s) else np.nan

vix_ema5 = ema(vix_s, 5)
vix_ema10 = ema(vix_s, 10)
vix_1d = pct_change_n(vix_s, 1)
vix_5d = pct_change_n(vix_s, 5)
vix_rsi = rsi(vix_s, 14)

mkt = latest(mkt_s)
mkt_ema5 = ema(mkt_s, 5)
mkt_ema20 = ema(mkt_s, 20)
mkt_2d = pct_change_n(mkt_s, 2)
mkt_5d = pct_change_n(mkt_s, 5)

# Ratios: short fear vs broad 30d, and 30d vs 3m term slope
short_ratio = v9 / vix if vix else np.nan
term_ratio = vix / v3m if v3m else np.nan

# ---------- Scoring ----------
# Positive = risk-off / favors looking for SHORT
# Negative = risk-on / favors looking for LONG
score = 0.0
reasons = []

def add(points, text):
    global score
    score += points
    reasons.append((points, text))

# 1) VIX short trend
if vix > vix_ema5:
    add(+0.9, "VIX מעל EMA5")
else:
    add(-0.9, "VIX מתחת EMA5")

if vix_ema5 > vix_ema10:
    add(+0.8, "EMA5 של VIX מעל EMA10")
else:
    add(-0.8, "EMA5 של VIX מתחת EMA10")

# 2) Very-short fear curve: VIX9D relative to VIX
if short_ratio >= 1.04:
    add(+1.25, "VIX9D גבוה משמעותית מ‑VIX — לחץ קצר־טווח")
elif short_ratio <= 0.96:
    add(-0.85, "VIX9D נמוך משמעותית מ‑VIX — לחץ קצר נרגע")

# 3) Term structure proxy: VIX vs VIX3M
if term_ratio >= 1.00:
    add(+1.20, "VIX ≥ VIX3M — עקום הפוך/לחוץ")
elif term_ratio <= 0.90:
    add(-0.75, "VIX נמוך משמעותית מ‑VIX3M — עקום רגוע")
else:
    add(-0.15, "עקום VIX רגיל אך לא עמוק")

# 4) VIX impulse
if vix_1d >= 8:
    add(+1.0, "קפיצת VIX יומית חדה")
elif vix_1d >= 3:
    add(+0.5, "VIX עולה היום")
elif vix_1d <= -8:
    add(-1.0, "נפילת VIX יומית חדה")
elif vix_1d <= -3:
    add(-0.5, "VIX יורד היום")

if vix_5d >= 10:
    add(+0.8, "VIX עלה חזק ב‑5 ימים")
elif vix_5d <= -10:
    add(-0.8, "VIX ירד חזק ב‑5 ימים")

# 5) Market confirmation — important for 2–3 day trades
if mkt > mkt_ema20:
    add(-0.85, f"{market_choice} מעל EMA20")
else:
    add(+0.85, f"{market_choice} מתחת EMA20")

if mkt > mkt_ema5:
    add(-0.55, f"{market_choice} מעל EMA5")
else:
    add(+0.55, f"{market_choice} מתחת EMA5")

if mkt_2d >= 1.0:
    add(-0.6, f"{market_choice} במומנטום חיובי ל‑2 ימים")
elif mkt_2d <= -1.0:
    add(+0.6, f"{market_choice} במומנטום שלילי ל‑2 ימים")

# 6) VVIX = volatility of VIX; use modestly
if not np.isnan(vvix):
    if vvix >= 115:
        add(+0.45, "VVIX גבוה — אי־ודאות בשוק האופציות")
    elif vvix <= 90:
        add(-0.25, "VVIX רגוע")

# 7) RSI only as momentum confirmation, never as automatic reversal
if not np.isnan(vix_rsi):
    if vix_rsi >= 60:
        add(+0.25, "RSI VIX תומך במומנטום עולה")
    elif vix_rsi <= 40:
        add(-0.25, "RSI VIX תומך במומנטום יורד")

# ---------- Decision ----------
# Slightly stricter threshold to avoid overtrading
if score >= 3.2:
    signal = "🔴 חיפוש SHORT"
    css = "signal-short"
    regime = "RISK-OFF"
    action = "לחפש טריגר שורט במדד/מניה ב־1H–4H. לא לרדוף אחרי ירידה שכבר התרחשה."
elif score <= -3.2:
    signal = "🟢 חיפוש LONG"
    css = "signal-long"
    regime = "RISK-ON"
    action = "לחפש טריגר לונג במדד/מניה ב־1H–4H, עדיפות לנכס שמראה חוזק יחסי."
else:
    signal = "🟡 המתנה"
    css = "signal-wait"
    regime = "NEUTRAL"
    action = "אין מספיק סנכרון לעסקה קצרה. עדיף להמתין מאשר להכריח עסקה."

abs_s = abs(score)
quality = (
    "חזק מאוד" if abs_s >= 5.0 else
    "חזק" if abs_s >= 4.2 else
    "בינוני־חזק" if abs_s >= 3.2 else
    "בינוני" if abs_s >= 2.0 else
    "חלש"
)

# ---------- Display ----------
c1, c2, c3 = st.columns(3)
c1.metric("VIX", f"{vix:.2f}", f"{vix_1d:+.2f}%")
c2.metric("VIX9D", f"{v9:.2f}")
c3.metric("VIX3M", f"{v3m:.2f}")

if not np.isnan(vvix):
    st.caption(f"VVIX: {vvix:.1f} · VIX EMA5: {vix_ema5:.2f} · EMA10: {vix_ema10:.2f} · RSI: {vix_rsi:.0f}")

st.markdown(
    f'<div class="signal {css}"><div class="big">{signal}</div>'
    f'<div>{regime} · ציון {score:.2f} · איכות: <b>{quality}</b></div></div>',
    unsafe_allow_html=True
)

st.info(action)

st.subheader("מה המערכת רואה עכשיו")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**VIX9D / VIX:** {short_ratio:.3f}")
    st.write(f"**VIX / VIX3M:** {term_ratio:.3f}")
    st.write(f"**VIX שינוי 5 ימים:** {vix_5d:+.2f}%")
with col2:
    st.write(f"**{market_choice} מול EMA20:** {'מעל' if mkt > mkt_ema20 else 'מתחת'}")
    st.write(f"**מומנטום 2 ימים:** {mkt_2d:+.2f}%")
    st.write(f"**מומנטום 5 ימים:** {mkt_5d:+.2f}%")

with st.expander("פירוט הציון"):
    for pts, text in sorted(reasons, key=lambda x: abs(x[0]), reverse=True):
        icon = "🔴" if pts > 0 else "🟢"
        st.markdown(f'<div class="rule">{icon} <b>{pts:+.2f}</b> — {text}</div>', unsafe_allow_html=True)

st.subheader("פרוטוקול כניסה ל־2–3 ימים")
if signal.startswith("🟢"):
    st.write("1. האפליקציה נותנת **LONG bias**.")
    st.write("2. מחפשים ב־1H–4H: higher low / פריצה / החזקת תמיכה.")
    st.write("3. כניסה רק אם גם 1H ו־15m מתחילים לנוע באותו כיוון.")
    st.write("4. אם VIX חוזר מעל EMA5/EMA10 והמדד נשבר מתחת EMA20 — מבטלים/מקטינים.")
    st.write("5. לקחת רווח חלקי מוקדם; העסקה מיועדת ל־1–3 ימי מסחר, לא להחזקה ארוכה.")
elif signal.startswith("🔴"):
    st.write("1. האפליקציה נותנת **SHORT bias**.")
    st.write("2. מחפשים ב־1H–4H: lower high / שבירת תמיכה / כישלון פריצה.")
    st.write("3. כניסה רק אם גם 1H ו־15m תומכים בירידה.")
    st.write("4. אם VIX יורד חזרה מתחת EMA5 והמדד חוזר מעל EMA20 — מבטלים/מקטינים.")
    st.write("5. לא לרדוף אחרי נר ירידה גדול; לקחת רווח חלקי בתוך 1–2 ימים.")
else:
    st.write("אין עסקה רק בגלל ה‑VIX. ממתינים עד שהפחד והמדד מסתנכרנים.")

st.caption("הנתונים מתקבלים מ‑Yahoo Finance דרך yfinance ומתעדכנים לפי זמינות המקור. זהו כלי סינון מחקרי, לא ייעוץ השקעות ולא הבטחת תשואה.")
st.caption("רענון אוטומטי של המטמון: עד 5 דקות. אפשר ללחוץ על 'רענן נתונים' בכל עת.")
