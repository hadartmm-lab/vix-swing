import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from io import StringIO
from urllib.parse import quote
from datetime import datetime

st.set_page_config(page_title="VIX Swing", page_icon="🎯", layout="centered")

st.markdown('''
<style>
:root{--bg:#07131f;--card:#0b1f31;--line:#184869;--txt:#f5f7fb;--muted:#b9c8d6}
html,body,[class*="css"]{background:var(--bg);color:var(--txt)}
.stApp{background:linear-gradient(180deg,#07131f 0%,#081725 100%)}
.block-container{max-width:820px;padding-top:.8rem;padding-bottom:2rem}
.hero{background:#0b1f31;border:1px solid var(--line);border-radius:20px;padding:16px 18px;margin-bottom:12px}
.hero-title{font-size:1.7rem;font-weight:950;color:#ffffff}.muted{color:#c9d7e3;font-size:.88rem}
.signal{background:linear-gradient(180deg,#0c2235 0%,#091b2b 100%);border:1px solid #245a7f;border-radius:24px;padding:20px 18px 18px;margin:12px 0;text-align:center;box-shadow:0 10px 35px rgba(0,0,0,.20)}
.signal-title{font-size:2.15rem;font-weight:950;margin-bottom:3px}.score{font-size:1.45rem;color:#ffffff;font-weight:950;margin-top:8px}.confidence{font-size:.86rem;color:#b8c9d7;margin-top:6px}
.gauge-wrap{margin:22px 3px 8px;position:relative;padding-top:30px}
.gauge{height:22px;border-radius:999px;background:linear-gradient(90deg,#0d7a50 0%,#197b6d 18%,#356f78 34%,#55626c 46%,#6a624b 50%,#7f6037 54%,#96602a 66%,#a64d28 82%,#9a2632 100%);border:1px solid rgba(255,255,255,.22);position:relative;box-shadow:inset 0 1px 3px rgba(255,255,255,.12),0 4px 14px rgba(0,0,0,.22)}
.gauge:before,.gauge:after{content:"";position:absolute;top:-4px;bottom:-4px;width:2px;background:rgba(255,255,255,.75);border-radius:2px}.gauge:before{left:25%}.gauge:after{left:75%}
.midline{position:absolute;left:50%;top:-4px;bottom:-4px;width:2px;background:rgba(255,255,255,.4);border-radius:2px}
.pointer{position:absolute;top:0;transform:translateX(-50%);width:0;height:0;border-left:9px solid transparent;border-right:9px solid transparent;border-top:15px solid #ffffff;filter:drop-shadow(0 2px 3px rgba(0,0,0,.45))}
.gauge-labels{display:grid;grid-template-columns:1fr 1fr 1fr;font-size:.72rem;color:#dbe7f0;margin-top:9px;font-weight:800}.gauge-labels span:nth-child(1){text-align:left}.gauge-labels span:nth-child(2){text-align:center}.gauge-labels span:nth-child(3){text-align:right}
.gauge-zones{display:grid;grid-template-columns:1fr 1fr 1fr;font-size:.64rem;color:#8fa7b9;margin-top:3px}.gauge-zones span:nth-child(1){text-align:left}.gauge-zones span:nth-child(2){text-align:center}.gauge-zones span:nth-child(3){text-align:right}
.status-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:9px;margin-top:10px}
@media(max-width:640px){.status-grid{grid-template-columns:1fr}}
.status{background:#0a1a28;border:1px solid #214d6b;border-radius:15px;padding:11px 13px;display:flex;justify-content:space-between;align-items:center;gap:12px}
.status-name{font-weight:850;font-size:.92rem;color:#eef5fa}.status-val{font-weight:900;text-align:left;white-space:nowrap}
.green{color:#72e8a7}.red{color:#ff8c92}.orange{color:#ffc06d}.blue{color:#82c9ff}.yellow{color:#ffe47b}.white{color:#fff}
.panel{background:#0b1f31;border:1px solid var(--line);border-radius:18px;padding:14px;margin-top:11px;color:#ffffff}
.panel, .panel *{color:#ffffff!important}
.stButton>button{width:100%;border-radius:14px;border:1px solid #1f6fa0;background:#0c2d46;color:white;font-weight:850;padding:.65rem 1rem}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,[data-testid="stRadio"] label,[data-testid="stWidgetLabel"] p,[data-testid="stCaptionContainer"] p,[data-testid="stExpander"] summary,[data-testid="stExpander"] summary p{color:#f3f7fb!important}
</style>
''', unsafe_allow_html=True)

DATA_TIMEOUT = 8
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (VIX-Swing/10.1; Streamlit)",
    "Accept": "application/json,text/csv,*/*",
}

# Official Cboe daily files. Used only as a fallback for daily volatility-index data.
CBOE_SYMBOLS = {
    "^VIX": "VIX",
    "^VIX9D": "VIX9D",
    "^VIX3M": "VIX3M",
    "^VVIX": "VVIX",
}

STOOQ_SYMBOLS = {
    "^NDX": "^ndx",
    "^GSPC": "^spx",
}

def _mark_source(obj, source):
    if obj is not None:
        try:
            obj.attrs["data_source"] = source
        except Exception:
            pass
    return obj

def source_name(obj):
    try:
        return obj.attrs.get("data_source", "לא ידוע") if obj is not None else "נכשל"
    except Exception:
        return "לא ידוע"

def _extract_yf_close(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        c = df["Close"] if "Close" in df.columns.get_level_values(0) else df.iloc[:, 0]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
    else:
        if "Close" not in df.columns:
            return None
        c = df["Close"]
    c = pd.to_numeric(c, errors="coerce").dropna().astype(float)
    return c if len(c) else None

def _extract_yf_ohlc(df):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        out = pd.DataFrame(index=df.index)
        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns.get_level_values(0):
                return None
            x = df[col]
            if isinstance(x, pd.DataFrame):
                x = x.iloc[:, 0]
            out[col] = pd.to_numeric(x, errors="coerce")
        return out.dropna()
    needed = ["Open", "High", "Low", "Close"]
    if not all(c in df.columns for c in needed):
        return None
    return df[needed].apply(pd.to_numeric, errors="coerce").dropna().astype(float)

def _yfinance_download(ticker, period, interval):
    """yfinance with short retry/backoff. Returns the raw DataFrame or None."""
    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=DATA_TIMEOUT,
            )
            if df is not None and not df.empty:
                return df
        except TypeError:
            # Compatibility with yfinance versions that do not expose timeout here.
            try:
                df = yf.download(
                    ticker, period=period, interval=interval,
                    auto_adjust=False, progress=False, threads=False
                )
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
        except Exception:
            pass
        time.sleep(0.6 * (attempt + 1))
    return None

def _yahoo_chart_df(ticker, period, interval):
    """Independent Yahoo Chart API path; useful when the yfinance wrapper itself breaks."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}"
    params = {
        "range": period,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits",
    }
    for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
        try:
            u = url.replace("query1.finance.yahoo.com", host)
            r = requests.get(u, params=params, headers=HTTP_HEADERS, timeout=DATA_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            result = (payload.get("chart", {}).get("result") or [None])[0]
            if not result:
                continue
            ts = result.get("timestamp") or []
            quote_data = ((result.get("indicators") or {}).get("quote") or [{}])[0]
            if not ts or not quote_data:
                continue
            tz = (result.get("meta") or {}).get("exchangeTimezoneName") or "America/New_York"
            idx = pd.to_datetime(ts, unit="s", utc=True)
            try:
                idx = idx.tz_convert(tz)
            except Exception:
                pass
            out = pd.DataFrame(index=idx)
            for src, dst in [("open","Open"),("high","High"),("low","Low"),("close","Close")]:
                vals = quote_data.get(src)
                if vals is not None and len(vals) == len(idx):
                    out[dst] = pd.to_numeric(vals, errors="coerce")
            if "Close" in out.columns and out["Close"].notna().sum() > 0:
                return out
        except Exception:
            continue
    return None

def _cboe_daily_close(ticker):
    """Official Cboe daily history fallback for VIX-family indices.

    Cboe migrated the live CSV host from cdn.cboe.com to cdn-api.cboe.com.
    Try the current host first and keep the older host as a compatibility fallback.
    """
    symbol = CBOE_SYMBOLS.get(ticker)
    if not symbol:
        return None
    for host in ["https://cdn-api.cboe.com", "https://cdn.cboe.com"]:
        url = f"{host}/api/global/us_indices/daily_prices/{symbol}_History.csv"
        try:
            r = requests.get(url, headers=HTTP_HEADERS, timeout=DATA_TIMEOUT)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            df.columns = [str(c).strip().upper() for c in df.columns]
            if "DATE" not in df.columns:
                continue
            close_col = "CLOSE" if "CLOSE" in df.columns else (symbol if symbol in df.columns else None)
            if close_col is None:
                candidates = [c for c in df.columns if c != "DATE"]
                close_col = candidates[-1] if candidates else None
            if close_col is None:
                continue
            idx = pd.to_datetime(df["DATE"], errors="coerce")
            vals = pd.to_numeric(df[close_col], errors="coerce")
            out = pd.Series(vals.values, index=idx).dropna().sort_index().astype(float)
            if len(out):
                return out
        except Exception:
            continue
    return None

def _fred_daily_close(ticker):
    """FRED fallback for selected Cboe daily series; no API key required."""
    series_map = {
        "^VIX": "VIXCLS",
        "^VIX3M": "VXVCLS",  # CBOE S&P 500 3-Month Volatility Index
    }
    series_id = series_map.get(ticker)
    if not series_id:
        return None
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        r = requests.get(url, headers=HTTP_HEADERS, timeout=DATA_TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if "DATE" not in df.columns or series_id not in df.columns:
            return None
        idx = pd.to_datetime(df["DATE"], errors="coerce")
        vals = pd.to_numeric(df[series_id].replace(".", np.nan), errors="coerce")
        out = pd.Series(vals.values, index=idx).dropna().sort_index().astype(float)
        return out if len(out) else None
    except Exception:
        return None

def _stooq_daily_close(ticker):
    """Independent daily-index fallback for NDX/SPX."""
    symbol = STOOQ_SYMBOLS.get(ticker)
    if not symbol:
        return None
    try:
        url = f"https://stooq.com/q/d/l/?s={quote(symbol)}&i=d"
        r = requests.get(url, headers=HTTP_HEADERS, timeout=DATA_TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        if "Date" not in df.columns or "Close" not in df.columns:
            return None
        idx = pd.to_datetime(df["Date"], errors="coerce")
        vals = pd.to_numeric(df["Close"], errors="coerce")
        s = pd.Series(vals.values, index=idx).dropna().sort_index().astype(float)
        return s if len(s) else None
    except Exception:
        return None

def _tail_for_period(s, period):
    if s is None:
        return None
    # We only need enough history for the app's indicators. Keeping a bounded tail
    # also avoids carrying decades of Cboe/Stooq data through Streamlit cache.
    n = {"1mo": 35, "3mo": 90, "6mo": 180, "1y": 300}.get(period, 300)
    return s.iloc[-n:]

def _min_points_required(period, interval):
    # Reject obviously broken responses (for example Yahoo returning a single row).
    if interval == "1d":
        return 20 if period == "1mo" else 30
    return 30

def _valid_history(s, period, interval):
    return s is not None and len(s) >= _min_points_required(period, interval)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_close(ticker, period="6mo", interval="1d"):
    # 1) yfinance wrapper. A non-empty result is not enough: Yahoo sometimes
    # returns only one bar for VIX9D/VIX3M, which is unusable for indicators.
    df = _yfinance_download(ticker, period, interval)
    c = _extract_yf_close(df)
    if _valid_history(c, period, interval):
        return _mark_source(c, "Yahoo / yfinance")

    # 2) Direct Yahoo Chart API through two hosts.
    df = _yahoo_chart_df(ticker, period, interval)
    c = _extract_yf_close(df)
    if _valid_history(c, period, interval):
        return _mark_source(c, "Yahoo Chart API")

    # 3) Independent daily fallbacks.
    if interval == "1d":
        c = _cboe_daily_close(ticker)
        c = _tail_for_period(c, period) if c is not None else None
        if _valid_history(c, period, interval):
            return _mark_source(c, "Cboe official")

        c = _fred_daily_close(ticker)
        c = _tail_for_period(c, period) if c is not None else None
        if _valid_history(c, period, interval):
            return _mark_source(c, "FRED / Cboe")

        c = _stooq_daily_close(ticker)
        c = _tail_for_period(c, period) if c is not None else None
        if _valid_history(c, period, interval):
            return _mark_source(c, "Stooq")
    return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlc(ticker, period="3mo", interval="1d"):
    df = _yfinance_download(ticker, period, interval)
    out = _extract_yf_ohlc(df)
    if out is not None and len(out):
        return _mark_source(out, "Yahoo / yfinance")

    df = _yahoo_chart_df(ticker, period, interval)
    out = _extract_yf_ohlc(df)
    if out is not None and len(out):
        return _mark_source(out, "Yahoo Chart API")
    return None

def ema_series(s,n): return s.ewm(span=n,adjust=False).mean()
def pctn(s,n): return float((s.iloc[-1]/s.iloc[-1-n]-1)*100) if s is not None and len(s)>n else np.nan

def rsi_series(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False).mean(); ad=dn.ewm(alpha=1/n,adjust=False).mean()
    rs=au/ad.replace(0,np.nan)
    return 100-100/(1+rs)

def _to_new_york_index(obj):
    """Return a copy indexed in America/New_York for stable US-market session bars."""
    if obj is None or not isinstance(obj.index, pd.DatetimeIndex):
        return None
    x = obj.copy()
    idx = x.index
    try:
        if idx.tz is None:
            # Yahoo intraday data for US tickers is normally exchange-local;
            # make that assumption explicit rather than letting resample anchor to UTC/local host time.
            idx = idx.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        else:
            idx = idx.tz_convert("America/New_York")
        x.index = idx
        return x.sort_index()
    except Exception:
        return None

def resample_close(s, hours):
    """Build 4H/12H bars anchored to the US regular session (09:30 ET).

    This avoids midnight/UTC bucket drift. 4H bars become 09:30-13:30 and
    13:30-close; a 12H bar starts at 09:30 and contains the full regular
    session. Empty overnight buckets are discarded.
    """
    if s is None or len(s) < 20 or not isinstance(s.index, pd.DatetimeIndex):
        return None
    x = _to_new_york_index(s)
    if x is None:
        return None
    try:
        # Keep regular US market session only. Yahoo hourly stamps can be 09:30 or 09:00
        # depending on feed normalization, so include 09:00 then anchor buckets at 09:30.
        x = x.between_time("09:00", "16:00", inclusive="both")
        out = x.resample(
            f"{hours}h",
            origin="start_day",
            offset="9h30min",
            label="left",
            closed="left",
        ).last().dropna()
        return out
    except Exception:
        return None

def resample_ohlc(df, hours):
    """Build session-anchored OHLC bars in America/New_York."""
    if df is None or len(df) < 20 or not isinstance(df.index, pd.DatetimeIndex):
        return None
    x = _to_new_york_index(df)
    if x is None:
        return None
    try:
        x = x.between_time("09:00", "16:00", inclusive="both")
        out = x.resample(
            f"{hours}h",
            origin="start_day",
            offset="9h30min",
            label="left",
            closed="left",
        ).agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
        return out
    except Exception:
        return None

def detect_divergence(s, lookback=45, pivot=2, min_sep=3, min_price_pct=0.7, min_rsi_delta=4.0, recent_bars=8):
    """Strict RSI divergence detector for VIX.

    A signal is accepted only when the two swing points are clearly separated,
    the price makes a meaningful new high/low, RSI moves materially the opposite
    way, and the most recent pivot is still recent. This filters weak/ambiguous
    divergences that can appear from small local wiggles.
    """
    if s is None or len(s) < 30:
        return "none"

    x = s.dropna().iloc[-lookback:]
    if len(x) < 30:
        return "none"
    r = rsi_series(x, 14)

    lows, highs = [], []
    for i in range(pivot, len(x) - pivot):
        w = x.iloc[i-pivot:i+pivot+1]
        # Require a unique local extreme where possible; this avoids flat/noisy pivots.
        if x.iloc[i] == w.min() and (w == x.iloc[i]).sum() == 1:
            lows.append(i)
        if x.iloc[i] == w.max() and (w == x.iloc[i]).sum() == 1:
            highs.append(i)

    def valid_pair(a, b):
        if b - a < min_sep:
            return False
        if len(x) - 1 - b > recent_bars:
            return False
        return pd.notna(r.iloc[a]) and pd.notna(r.iloc[b])

    # Bullish divergence on VIX: lower VIX low + higher RSI low -> supports VIX rebound / QQQ short.
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if valid_pair(a, b):
            price_change = (x.iloc[b] / x.iloc[a] - 1) * 100
            rsi_change = float(r.iloc[b] - r.iloc[a])
            if price_change <= -min_price_pct and rsi_change >= min_rsi_delta:
                return "bullish"

    # Bearish divergence on VIX: higher VIX high + lower RSI high -> supports VIX fade / QQQ long.
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if valid_pair(a, b):
            price_change = (x.iloc[b] / x.iloc[a] - 1) * 100
            rsi_change = float(r.iloc[b] - r.iloc[a])
            if price_change >= min_price_pct and rsi_change <= -min_rsi_delta:
                return "bearish"

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
    m_intra=fetch_close(market_ticker,period="60d",interval="60m")
    mo_intra=fetch_ohlc(market_ticker,period="60d",interval="60m")

required_daily = {
    "VIX": v,
    "VIX9D": v9,
    "VIX3M": v3,
    market_name: m,
}
failed_daily = [name for name, x in required_daily.items() if x is None or len(x) < 30]
if failed_daily:
    st.error("לא הצלחתי להשיג נתונים תקינים עבור: " + ", ".join(failed_daily) + ". ניסיתי Yahoo, מקור Yahoo ישיר, Cboe הרשמי, FRED ובמידת האפשר גם Stooq.")
    with st.expander("אבחון מקורות נתונים"):
        for name, x in required_daily.items():
            st.write(f"{name}: {'✅' if x is not None and len(x)>=30 else '❌'} · {source_name(x)} · {len(x) if x is not None else 0} נקודות")
    st.stop()

V=float(v.iloc[-1]); V9=float(v9.iloc[-1]); V3=float(v3.iloc[-1]); VV=float(vv.iloc[-1]) if vv is not None and len(vv) else np.nan
V12=resample_close(v_intra,12)
M12=resample_close(m_intra,12)
MO12=resample_ohlc(mo_intra,12)
if V12 is None or len(V12)<30 or M12 is None or len(M12)<20 or MO12 is None or len(MO12)<20:
    failed_intraday=[]
    if V12 is None or len(V12)<30: failed_intraday.append("VIX 60m → 12H")
    if M12 is None or len(M12)<20: failed_intraday.append(f"{market_name} 60m → 12H")
    if MO12 is None or len(MO12)<20: failed_intraday.append(f"{market_name} OHLC 60m → 12H")
    st.error("לא הצלחתי לבנות נתוני 12H עבור: " + ", ".join(failed_intraday) + ". בוצעו ניסיונות חוזרים דרך yfinance וגם דרך Yahoo Chart API ישיר.")
    with st.expander("אבחון מקורות נתונים"):
        st.write(f"VIX 60m: {source_name(v_intra)} · {len(v_intra) if v_intra is not None else 0} נקודות")
        st.write(f"{market_name} 60m: {source_name(m_intra)} · {len(m_intra) if m_intra is not None else 0} נקודות")
        st.write(f"{market_name} OHLC 60m: {source_name(mo_intra)} · {len(mo_intra) if mo_intra is not None else 0} נקודות")
        st.caption("נתוני 12H הם חלק קריטי מהשיטה, לכן האפליקציה לא מחליפה אותם בנתוני Daily שעלולים לשנות את האות.")
    st.stop()
V12_LAST=float(V12.iloc[-1])
VE9,VE26,CROSS,CROSS_AGE,APPROACH,CROSS_NEAR,GAP=cross_status(V12)
C1=pctn(v,1); C5=pctn(v,5); R9=V9/V; R3=V/V3
M=float(m.iloc[-1]); M2=pctn(m,2); M5=pctn(m,5)
DIV4=detect_divergence(resample_close(v_intra,4), lookback=60, pivot=3, min_sep=4, min_price_pct=0.75, min_rsi_delta=4.0, recent_bars=8)
DIV12=detect_divergence(V12, lookback=50, pivot=2, min_sep=3, min_price_pct=1.0, min_rsi_delta=4.0, recent_bars=6)
SR, SUPPORT, RESISTANCE=support_resistance_signal(MO12,44,2)

score=0.0; reasons=[]
def add(p,t):
    global score
    score+=p; reasons.append((p,t))

# VIX EMA structure
add(1.2 if V12_LAST>VE9 else -1.2, "VIX 12H מעל EMA9" if V12_LAST>VE9 else "VIX 12H מתחת EMA9")
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

# Nasdaq / S&P: no moving-average score; only short momentum and 12H support/resistance
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

# Support / resistance from about one month of 12H market structure
if SR=="reject_resistance": add(0.5,"דחייה מהתנגדות")
elif SR=="breakout": add(-0.6,"פריצה מעל התנגדות")
elif SR=="bounce_support": add(-0.5,"תגובה מתמיכה")
elif SR=="breakdown": add(0.6,"שבירה מתחת לתמיכה")

# Modest VIX level adjustment
if V>=30: add(0.5,"VIX ≥30")
elif V>=25: add(0.25,"VIX ≥25")
elif V<15: add(-0.2,"VIX <15")

# One clear signal score
# The internal weighted score and the visible scale now use the SAME units.
# +5 raw points = SHORT threshold 5/10; -5 raw points = LONG threshold 5/10.
# This keeps the score intuitive and avoids artificial compression against a theoretical maximum.
signed_signal = max(-10.0, min(10.0, float(score)))
signal_score = abs(signed_signal)

ENTRY_THRESHOLD = 5.0
STRONG_THRESHOLD = 7.0
VERY_STRONG_THRESHOLD = 8.5

if signed_signal >= VERY_STRONG_THRESHOLD:
    state,icon,cls = "VERY STRONG SHORT","🔴","red"
elif signed_signal >= STRONG_THRESHOLD:
    state,icon,cls = "STRONG SHORT","🔴","red"
elif signed_signal >= ENTRY_THRESHOLD:
    state,icon,cls = "SHORT","🟠","orange"
elif signed_signal <= -VERY_STRONG_THRESHOLD:
    state,icon,cls = "VERY STRONG LONG","🟢","green"
elif signed_signal <= -STRONG_THRESHOLD:
    state,icon,cls = "STRONG LONG","🟢","green"
elif signed_signal <= -ENTRY_THRESHOLD:
    state,icon,cls = "LONG","🔵","blue"
else:
    state,icon,cls = "WAIT","🟡","yellow"

# Gauge: -10 = LONG, 0 = neutral, +10 = SHORT.
pos = max(2, min(98, (signed_signal+10)/20*100))
if state == "WAIT":
    lean = "SHORT" if signed_signal > 0 else ("LONG" if signed_signal < 0 else "NEUTRAL")
    score_line = f"Signal Score <b>{signal_score:.1f}/10</b> · נטייה {lean}"
else:
    score_line = f"Signal Score <b>{signal_score:.1f}/10</b>"

st.markdown(f"""
<div class="signal">
  <div class="signal-title {cls}">{icon} {state}</div>
  <div class="score">{score_line}</div>
  <div class="confidence">כניסה לחיפוש עסקה רק מ־<b>5.0/10</b></div>
  <div class="gauge-wrap">
    <div class="pointer" style="left:{pos:.1f}%"></div>
    <div class="gauge"><div class="midline"></div></div>
    <div class="gauge-labels"><span>LONG 10</span><span>WAIT 0</span><span>SHORT 10</span></div>
    <div class="gauge-zones"><span>LONG ≤ -5</span><span>WAIT</span><span>SHORT ≥ +5</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Compact statuses
ema_bias="SHORT" if VE9>VE26 else "LONG"
ema_cls="red" if ema_bias=="SHORT" else "green"
if CROSS and CROSS_AGE is not None and CROSS_AGE<=5:
    cross_text=("Golden Cross" if CROSS=="golden" else "Death Cross")+" · קרבה 10/10"
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

st.markdown('<div class="status-grid">'+
    status_row("VIX · EMA9/26 · 12H",ema_bias,ema_cls)+
    status_row("Cross · EMA9/26 · 12H",cross_text,cross_cls)+
    status_row("Divergence 4H",d4,d4c)+
    status_row("Divergence 12H",d12,d12c)+
    status_row("Support / Resistance · 12H",srt,src)+
    status_row("Term Structure",term_text,term_cls)+
    status_row("VIX9D / VIX",("לחץ" if R9>=1.03 else ("רגוע" if R9<=0.97 else "מאוזן")),"red" if R9>=1.03 else ("green" if R9<=0.97 else "white"))+
    '</div>',unsafe_allow_html=True)

# Minimal execution reminder
if "SHORT" in state: action="חפש טריגר SHORT ב-1H"
elif "LONG" in state: action="חפש טריגר LONG ב-1H"
else: action="אין עסקה — המתן לסנכרון"
st.markdown(f'<div class="panel" style="text-align:center;font-weight:900">{action}</div>',unsafe_allow_html=True)

with st.expander("פירוט החישוב"):
    st.write(f"VIX 12H {V12_LAST:.2f} · EMA9 {VE9:.2f} · EMA26 {VE26:.2f} · Cross proximity {CROSS_NEAR}/10")
    st.write(f"VIX 1D {C1:+.2f}% · 5D {C5:+.2f}% · VIX9D/VIX {R9:.3f} · VIX/VIX3M {R3:.3f}")
    st.write(f"{market_name}: ללא EMA בציון · מומנטום 2D {M2:+.2f}% · 5D {M5:+.2f}% · Support/Resistance מחושב על 12H")
    st.write("RSI של VIX אינו מקבל ניקוד ישיר; הוא משמש רק לזיהוי Divergence מאומת ב-4H/12H עם סינון רעש מחמיר.")
    st.write(f"ציון משוקלל נטו: {score:+.2f} → Signal Score {signal_score:.2f}/10 · כיוון: {'SHORT' if signed_signal>0 else ('LONG' if signed_signal<0 else 'NEUTRAL')}")
    for pts,txt in sorted(reasons,key=lambda z:abs(z[0]),reverse=True):
        st.write(f"**{pts:+.2f}** — {txt}")

with st.expander("מקורות נתונים / גיבוי"):
    st.write(f"VIX Daily: **{source_name(v)}**")
    st.write(f"VIX9D Daily: **{source_name(v9)}**")
    st.write(f"VIX3M Daily: **{source_name(v3)}**")
    st.write(f"{market_name} Daily: **{source_name(m)}**")
    st.write(f"VIX 60m: **{source_name(v_intra)}**")
    st.write(f"{market_name} 60m: **{source_name(m_intra)}**")
    st.caption("סדר הגיבוי: Yahoo/yfinance → Yahoo Chart API ישיר. לנתוני Daily בלבד: Cboe הרשמי למדדי VIX, ו-Stooq למדדי NDX/SPX.")

st.caption(f"עודכן {datetime.now().strftime('%H:%M')} · מנגנון Multi-Source + Retry פעיל · כלי מחקרי, לא ייעוץ השקעות")
