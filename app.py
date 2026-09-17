import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from io import StringIO
from urllib.parse import quote
from datetime import datetime

# Embedded data validation; no separate data_quality.py is required.
"""Closed-session validation shared by the UI and regression tests."""
from functools import lru_cache
import numpy as np
import pandas as pd
import exchange_calendars as xc

TZ = 'America/New_York'


def utc_now(now=None):
    x = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    return x.tz_localize('UTC') if x.tz is None else x.tz_convert('UTC')


@lru_cache(maxsize=8)
def calendar_for(year):
    return xc.get_calendar('XNYS', start=f'{year-2}-01-01', end=f'{year+1}-12-31')


def schedule(now=None):
    return calendar_for(utc_now(now).year).schedule


def latest_daily_session(now=None):
    now = utc_now(now)
    # Allow the volatility index closing calculation and provider publication.
    sched = schedule(now)
    closed = sched[sched['close'] + pd.Timedelta(hours=1) <= now]
    return closed.index[-1].tz_localize(None).normalize()


def clean_history(obj):
    if obj is None or obj.empty or not isinstance(obj.index, pd.DatetimeIndex):
        return None
    x = obj.copy()
    x = x.loc[~x.index.isna()]
    x = x.loc[~x.index.duplicated(keep='last')].sort_index()
    x = x.apply(pd.to_numeric, errors='coerce') if isinstance(x, pd.DataFrame) else pd.to_numeric(x, errors='coerce')
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if isinstance(x, pd.DataFrame):
        required = ['Open', 'High', 'Low', 'Close']
        if not all(c in x for c in required):
            return None
        good = (x[required] > 0).all(axis=1)
        good &= x.High >= x[['Open','Close','Low']].max(axis=1)
        good &= x.Low <= x[['Open','Close','High']].min(axis=1)
        x = x.loc[good]
    else:
        x = x[x > 0]
    return x if len(x) else None


def daily_history(obj, now=None):
    x = clean_history(obj)
    if x is None:
        return None
    # Daily dates are exchange date labels, not UTC instants.
    x.index = x.index.tz_localize(None).normalize()
    x = x.loc[~x.index.duplicated(keep='last')]
    sched = schedule(now)
    x = x.loc[x.index.isin(sched.index.tz_localize(None))]
    cutoff = latest_daily_session(now)
    return x.loc[x.index <= cutoff]


def feed_phase(obj):
    """Use the dominant hourly timestamp phase; never shift source timestamps."""
    idx=obj.index
    idx=idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
    minutes=pd.Series(idx.minute)
    return int(minutes.mode().iloc[0])


def hourly_grid(op, cl, phase):
    first=op.floor('h')+pd.Timedelta(minutes=phase)
    if first<op:
        first+=pd.Timedelta(hours=1)
    return pd.date_range(first,cl,freq='1h',inclusive='left')


def bucket_grid(op,cl,hours,phase):
    starts=hourly_grid(op,cl,phase)
    for label in pd.date_range(op,cl,freq=f'{hours}h',inclusive='left'):
        boundary=min(label+pd.Timedelta(hours=hours),cl)
        expected=starts[(starts>=label)&(starts<boundary)]
        if not len(expected):
            continue
        # Source hours can straddle a bucket edge. Do not pretend they closed earlier.
        end=min(expected[-1]+pd.Timedelta(hours=1),cl)
        yield label,expected,end


def fresh_history(obj, interval, now=None):
    if obj is None or obj.empty:
        return False
    if interval == '1d':
        return obj.index[-1].tz_localize(None).normalize() == latest_daily_session(now)
    now=utc_now(now)
    phase=feed_phase(obj)
    sched=schedule(now)
    sched=sched[(sched['open']<=now)&(sched['close']>=now-pd.Timedelta(days=10))]
    expected=None
    for _,day in sched.iterrows():
        for t in hourly_grid(day['open'],day['close'],phase):
            if min(t+pd.Timedelta(hours=1),day['close'])+pd.Timedelta(minutes=15)<=now:
                expected=t
    idx=obj.index
    idx=idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
    # Ignore premarket and after-hours ticks when deciding regular-session freshness.
    return expected is not None and expected in idx.tz_convert('UTC')


def session_bars(obj, hours, now=None):
    """Aggregate actual hourly samples using their native timestamp phase.

    These are session summaries, not exact exchange-native 4h/12h candles.
    :00/:15/:30 feeds remain on their real timestamps. Require every expected
    in-session sample on that grid and wait until its source candle closes.
    """
    x=clean_history(obj)
    if x is None or hours not in (4,12):
        return None
    now=utc_now(now)
    phase=feed_phase(x)
    x.index=x.index.tz_localize(TZ) if x.index.tz is None else x.index.tz_convert(TZ)
    x.index=x.index.tz_convert('UTC')
    sched=schedule(now)
    sched=sched[(sched['close']>=x.index.min())&(sched['open']<=now)]
    rows,labels=[],[]
    for _,day in sched.iterrows():
        for label,expected,end in bucket_grid(day['open'],day['close'],hours,phase):
            if end+pd.Timedelta(minutes=15)>now or not expected.isin(x.index).all():
                continue
            part=x.loc[expected]
            labels.append(label)
            if isinstance(x,pd.DataFrame):
                rows.append({'Open':part.Open.iloc[0],'High':part.High.max(),
                             'Low':part.Low.min(),'Close':part.Close.iloc[-1]})
            else:
                rows.append(part.iloc[-1])
    if not rows:
        return None
    idx=pd.DatetimeIndex(labels).tz_convert(TZ)
    out=pd.DataFrame(rows,index=idx) if isinstance(x,pd.DataFrame) else pd.Series(rows,index=idx)
    out.attrs=obj.attrs.copy()
    out.attrs['hourly_phase_minutes']=phase
    return out


def rsi_series(s, n=14):
    """Wilder RSI: SMA seed then recursive smoothing, with explicit flat cases."""
    s = pd.to_numeric(s,errors='coerce')
    result = pd.Series(np.nan,index=s.index,dtype=float)
    if len(s) <= n:
        return result
    d=s.diff(); up=d.clip(lower=0); down=-d.clip(upper=0)
    gain,loss=float(up.iloc[1:n+1].mean()),float(down.iloc[1:n+1].mean())
    for i in range(n,len(s)):
        if i>n:
            gain=(gain*(n-1)+up.iloc[i])/n
            loss=(loss*(n-1)+down.iloc[i])/n
        result.iloc[i] = (50.0 if gain==0 else 100.0) if loss==0 else 100-100/(1+gain/loss)
    return result


def latest_bucket_start(hours, now=None, obj=None):
    now=utc_now(now)
    phase=feed_phase(obj) if obj is not None else 30
    sched=schedule(now)
    sched=sched[(sched['open']<=now)&(sched['close']>=now-pd.Timedelta(days=10))]
    result=None
    for _,day in sched.iterrows():
        for label,_,end in bucket_grid(day['open'],day['close'],hours,phase):
            if end+pd.Timedelta(minutes=15)<=now:
                result=label.tz_convert(TZ)
    return result

st.set_page_config(page_title="VIX Swing", page_icon="🎯", layout="centered")

st.markdown('''
<style>
:root{--bg:#07131f;--card:#0b1f31;--line:#184869;--txt:#f5f7fb;--muted:#b9c8d6}
html,body,[class*="css"]{background:var(--bg);color:var(--txt)}
[data-testid="stHeader"]{background:#07131f;color:#f5f7fb}.stApp{background:linear-gradient(180deg,#07131f 0%,#081725 100%)}
.block-container{max-width:820px;padding-top:3.5rem;padding-bottom:2rem}
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
.status-name{font-weight:850;font-size:.92rem;color:#eef5fa}.status-val{font-weight:900;text-align:left;overflow-wrap:anywhere}.hero,.status,.panel,.drivers-card{direction:rtl} [data-testid="stMarkdownContainer"] p,[data-testid="stCaptionContainer"] p,[data-testid="stAlertContainer"]{direction:rtl;text-align:right;unicode-bidi:plaintext} bdi{unicode-bidi:isolate}.gauge-wrap{direction:ltr}
.green{color:#72e8a7}.red{color:#ff8c92}.orange{color:#ffc06d}.blue{color:#82c9ff}.yellow{color:#ffe47b}.white{color:#fff}
.panel{background:#0b1f31;border:1px solid var(--line);border-radius:18px;padding:14px;margin-top:11px;color:#ffffff}
.panel, .panel *{color:#ffffff!important}
.drivers-card{background:linear-gradient(180deg,#0b1f31 0%,#0a1b2b 100%);border:1px solid var(--line);border-radius:18px;padding:14px 14px 10px;margin:10px 0 12px;box-shadow:0 8px 24px rgba(0,0,0,.14)}
.drivers-title{font-size:1.02rem;font-weight:950;color:#ffffff;margin-bottom:10px}
.driver-row{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:9px 0;border-top:1px solid rgba(255,255,255,.07)}
.driver-row:first-of-type{border-top:none;padding-top:0}
.driver-text{font-size:.92rem;font-weight:800;color:#eef5fa;line-height:1.35;flex:1}
.driver-badge{font-size:.84rem;font-weight:950;padding:4px 10px;border-radius:999px;border:1px solid rgba(255,255,255,.16);white-space:nowrap;background:rgba(255,255,255,.04)}
.driver-badge.red{background:rgba(255,140,146,.10);border-color:rgba(255,140,146,.35)}
.driver-badge.green{background:rgba(114,232,167,.10);border-color:rgba(114,232,167,.35)}
.driver-badge.white{background:rgba(255,255,255,.06);border-color:rgba(255,255,255,.15)}
.stButton>button{width:100%;border-radius:14px;border:1px solid #1f6fa0;background:#0c2d46;color:white;font-weight:850;padding:.65rem 1rem}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,[data-testid="stRadio"] label,[data-testid="stWidgetLabel"] p,[data-testid="stCaptionContainer"] p,[data-testid="stExpander"] summary,[data-testid="stExpander"] summary p{color:#f3f7fb!important}
</style>
''', unsafe_allow_html=True)

DATA_TIMEOUT = 8
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (VIX-Swing/10.7; Streamlit)",
    "Accept": "application/json,text/csv,*/*",
}

# Official Cboe daily files. Used only as a fallback for daily volatility-index data.
CBOE_SYMBOLS = {
    "^VIX": "VIX",
    "^VIX9D": "VIX9D",
    "^VIX3M": "VIX3M",
    "^VVIX": "VVIX",
    "^VIX1D": "VIX1D",
    "^SKEW": "SKEW",
    "^COR1M": "COR1M",
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
    for attempt in range(2):
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
        df = df.rename(columns={"observation_date": "DATE"})
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

def _prepare_history(obj, interval):
    return daily_history(obj) if interval == "1d" else clean_history(obj)

def _valid_history(s, period, interval):
    return (s is not None and len(s) >= _min_points_required(period, interval)
            and fresh_history(s, interval))

@st.cache_data(ttl=300, show_spinner=False)
def fetch_close(ticker, period="6mo", interval="1d"):
    # 1) yfinance wrapper. A non-empty result is not enough: Yahoo sometimes
    # returns only one bar for VIX9D/VIX3M, which is unusable for indicators.
    df = _yfinance_download(ticker, period, interval)
    c = _prepare_history(_extract_yf_close(df), interval)
    if _valid_history(c, period, interval):
        return _mark_source(c, "Yahoo / yfinance")

    # 2) Direct Yahoo Chart API through two hosts.
    df = _yahoo_chart_df(ticker, period, interval)
    c = _prepare_history(_extract_yf_close(df), interval)
    if _valid_history(c, period, interval):
        return _mark_source(c, "Yahoo Chart API")

    # 3) Independent daily fallbacks.
    if interval == "1d":
        c = _cboe_daily_close(ticker)
        c = _prepare_history(_tail_for_period(c, period), interval)
        if _valid_history(c, period, interval):
            return _mark_source(c, "Cboe official")

        c = _fred_daily_close(ticker)
        c = _prepare_history(_tail_for_period(c, period), interval)
        if _valid_history(c, period, interval):
            return _mark_source(c, "FRED / Cboe")

        c = _stooq_daily_close(ticker)
        c = _prepare_history(_tail_for_period(c, period), interval)
        if _valid_history(c, period, interval):
            return _mark_source(c, "Stooq")
    return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlc(ticker, period="3mo", interval="1d"):
    df = _yfinance_download(ticker, period, interval)
    out = _prepare_history(_extract_yf_ohlc(df), interval)
    if _valid_history(out, period, interval):
        return _mark_source(out, "Yahoo / yfinance")

    df = _yahoo_chart_df(ticker, period, interval)
    out = _prepare_history(_extract_yf_ohlc(df), interval)
    if _valid_history(out, period, interval):
        return _mark_source(out, "Yahoo Chart API")
    return None

def ema_series(s,n): return s.ewm(span=n,adjust=False).mean()
def pctn(s,n): return float((s.iloc[-1]/s.iloc[-1-n]-1)*100) if s is not None and len(s)>n else np.nan

def resample_close(s, hours):
    return session_bars(s, hours)

def resample_ohlc(df, hours):
    return session_bars(df, hours)

def rolling_zscore_last(s, window=60):
    if s is None:
        return np.nan
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < max(20, window // 2):
        return np.nan
    w = x.iloc[-window:]
    sd = float(w.std(ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return 0.0
    return float((w.iloc[-1] - w.mean()) / sd)

def aligned_ratio(a, b):
    if a is None or b is None:
        return None
    aa = pd.to_numeric(a, errors="coerce").dropna()
    bb = pd.to_numeric(b, errors="coerce").dropna()
    # Normalize timezone differences so daily Cboe/Yahoo series can align reliably.
    try:
        aa.index = pd.to_datetime(aa.index).tz_localize(None).normalize()
    except Exception:
        try: aa.index = pd.to_datetime(aa.index).tz_convert(None).normalize()
        except Exception: pass
    try:
        bb.index = pd.to_datetime(bb.index).tz_localize(None).normalize()
    except Exception:
        try: bb.index = pd.to_datetime(bb.index).tz_convert(None).normalize()
        except Exception: pass
    z = pd.concat([aa.rename("a"), bb.rename("b")], axis=1, join="inner").dropna()
    if len(z) < 20:
        return None
    r = z["a"] / z["b"].replace(0, np.nan)
    return r.replace([np.inf, -np.inf], np.nan).dropna()

def swing_exhaustion_signal(s, lookback=45, pivot=2, recent_bars=7, min_swing_pct=0.8, rejection_pct=1.0):
    """Detect recent VIX swing exhaustion, not trend continuation.

    - New higher high that has already rejected lower can precede VIX fading -> מדד LONG.
    - New lower low that has already rebounded can precede VIX rising -> מדד SHORT.
    We deliberately require the rejection/rebound so a bare higher-high/lower-low is not
    incorrectly treated as a reversal by itself.
    """
    if s is None or len(s) < 25:
        return "none"
    x = pd.to_numeric(s, errors="coerce").dropna().iloc[-lookback:]
    if len(x) < 20:
        return "none"
    highs=[]; lows=[]
    for i in range(pivot, len(x)-pivot):
        w=x.iloc[i-pivot:i+pivot+1]
        if x.iloc[i] == w.max() and (w == x.iloc[i]).sum() == 1: highs.append(i)
        if x.iloc[i] == w.min() and (w == x.iloc[i]).sum() == 1: lows.append(i)
    last=float(x.iloc[-1])
    if len(highs)>=2:
        a,b=highs[-2],highs[-1]
        if len(x)-1-b <= recent_bars:
            new_high=(x.iloc[b]/x.iloc[a]-1)*100
            rejection=(last/x.iloc[b]-1)*100
            if new_high >= min_swing_pct and rejection <= -rejection_pct:
                return "higher_high_rejection"  # VIX downside / מדד LONG
    if len(lows)>=2:
        a,b=lows[-2],lows[-1]
        if len(x)-1-b <= recent_bars:
            new_low=(x.iloc[b]/x.iloc[a]-1)*100
            rebound=(last/x.iloc[b]-1)*100
            if new_low <= -min_swing_pct and rebound >= rejection_pct:
                return "lower_low_rebound"  # VIX upside / מדד SHORT
    return "none"

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
    r = rsi_series(s.dropna(), 14).reindex(x.index)

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

def smart_fib_state(df, lookback=55, pivot=2, min_impulse_pct=7.0):
    """Two-way Smart Fibonacci on closed 12H VIX OHLC bars.

    Finds the latest meaningful confirmed swing impulse, then extends its endpoint
    if a newer extreme is made in the same direction. The 0.50-0.618 retracement
    area is treated as a *decision zone*, never as a standalone entry signal.

    Returns a dict with direction, anchors, zone and phase. Direction refers to
    the VIX impulse: 'down' = high->low; 'up' = low->high.
    """
    empty={"valid":False,"direction":None,"start":np.nan,"end":np.nan,
           "level50":np.nan,"level618":np.nan,"zone_low":np.nan,"zone_high":np.nan,
           "current":np.nan,"phase":"none","distance_pct":np.nan,"impulse_pct":np.nan}
    x=clean_history(df)
    if x is None or len(x)<18:
        return empty
    x=x.iloc[-lookback:].copy()
    highs=[]; lows=[]
    for i in range(pivot,len(x)-pivot):
        wh=x['High'].iloc[i-pivot:i+pivot+1]
        wl=x['Low'].iloc[i-pivot:i+pivot+1]
        if x['High'].iloc[i]==wh.max() and (wh==x['High'].iloc[i]).sum()==1:
            highs.append(i)
        if x['Low'].iloc[i]==wl.min() and (wl==x['Low'].iloc[i]).sum()==1:
            lows.append(i)
    pts=sorted([(i,'H',float(x['High'].iloc[i])) for i in highs]+[(i,'L',float(x['Low'].iloc[i])) for i in lows])
    # Consecutive same-type pivots are compressed to the more extreme one.
    compact=[]
    for pt in pts:
        if compact and compact[-1][1]==pt[1]:
            if (pt[1]=='H' and pt[2]>=compact[-1][2]) or (pt[1]=='L' and pt[2]<=compact[-1][2]):
                compact[-1]=pt
        else:
            compact.append(pt)
    candidates=[]
    for a,b in zip(compact,compact[1:]):
        if a[1]=='L' and b[1]=='H':
            move=(b[2]/a[2]-1)*100
            if move>=min_impulse_pct: candidates.append((b[0],a,b,'up',move))
        elif a[1]=='H' and b[1]=='L':
            move=(a[2]/b[2]-1)*100
            if move>=min_impulse_pct: candidates.append((b[0],a,b,'down',move))
    if not candidates:
        return empty
    _,a,b,direction,move=max(candidates,key=lambda z:z[0])
    start_i,end_i=a[0],b[0]
    start=float(a[2]); end=float(b[2])
    # Dynamic endpoint: if the same impulse keeps making a new extreme, extend it.
    tail=x.iloc[end_i:]
    if direction=='up':
        rel=int(np.argmax(tail['High'].to_numpy()))
        candidate=float(tail['High'].iloc[rel])
        if candidate>end:
            end=candidate; end_i=end_i+rel
    else:
        rel=int(np.argmin(tail['Low'].to_numpy()))
        candidate=float(tail['Low'].iloc[rel])
        if candidate<end:
            end=candidate; end_i=end_i+rel
    rng=abs(end-start)
    if rng<=0:
        return empty
    if direction=='down':
        level50=end+0.50*rng
        level618=end+0.618*rng
    else:
        level50=end-0.50*rng
        level618=end-0.618*rng
    zone_low=min(level50,level618); zone_high=max(level50,level618)
    cur=float(x['Close'].iloc[-1]); prev=float(x['Close'].iloc[-2])
    recent=x.iloc[-3:]
    mom=cur-float(x['Close'].iloc[-3])
    tol=max(0.04,0.015*rng)
    break_buf=max(0.03,0.008*rng)
    phase='tracking'
    # The retracement route and the reaction are intentionally symmetric.
    if direction=='down':
        touched=float(recent['High'].max()) >= zone_low-tol
        if cur < zone_low-tol:
            phase='approaching_up' if mom>0 else 'below_zone'
        elif cur <= zone_high+tol:
            phase='reject_down' if touched and cur<prev and mom<0 else 'decision_zone'
        else:
            phase='break_above_0618' if cur>zone_high+break_buf else 'above_zone'
    else:
        touched=float(recent['Low'].min()) <= zone_high+tol
        if cur > zone_high+tol:
            phase='approaching_down' if mom<0 else 'above_zone'
        elif cur >= zone_low-tol:
            phase='rebound_up' if touched and cur>prev and mom>0 else 'decision_zone'
        else:
            phase='break_below_0618' if cur<zone_low-break_buf else 'below_zone'
    if cur<zone_low: dist=(zone_low-cur)/cur*100
    elif cur>zone_high: dist=(cur-zone_high)/cur*100
    else: dist=0.0
    return {"valid":True,"direction":direction,"start":start,"end":end,
            "start_time":x.index[start_i],"end_time":x.index[end_i],
            "level50":float(level50),"level618":float(level618),
            "zone_low":float(zone_low),"zone_high":float(zone_high),"current":cur,
            "phase":phase,"distance_pct":float(dist),"impulse_pct":float(move)}

def status_row(name,value,cls="white"):
    return f'<div class="status"><div class="status-name"><bdi>{name}</bdi></div><div class="status-val {cls}"><bdi>{value}</bdi></div></div>'

st.markdown('<div class="hero"><div class="hero-title">🎯 VIX Swing</div><div class="muted">כיוון המדד במבט אחד · v10.7 Smart Fib</div></div>',unsafe_allow_html=True)
if st.button("🔄 רענן",use_container_width=True): st.cache_data.clear(); st.rerun()

market_name=st.radio("מדד",["Nasdaq 100","S&P 500"],horizontal=True,label_visibility="collapsed")
market_ticker="^NDX" if market_name=="Nasdaq 100" else "^GSPC"
st.markdown(f'<div class="panel" dir="rtl">הכיוון מתייחס למדד שנבחר: <bdi dir="ltr">{market_name}</bdi><br><bdi dir="ltr">LONG</bdi> — חיפוש עלייה במדד<br><bdi dir="ltr">SHORT</bdi> — חיפוש ירידה במדד</div>', unsafe_allow_html=True)
with st.expander("איך לקרוא את האיתות והזמנים"):
    st.markdown('<div dir="rtl">החישוב משתמש בנרות סגורים בלבד. הכינוי <bdi dir="ltr">12H</bdi> הוא סיכום סשן מסחר, ולא נר של 12 שעות רצופות. סיכומי <bdi dir="ltr">4H</bdi> ונתוני הסשן תלויים בשעות שמספק המקור; כשנרות המקור אינם מיושרים לפתיחה, הסיכומים מקורבים ואינם כוללים חלק פתיחה שאין עליו נר מלא. נשמר מרווח פרסום של 15 דקות.</div>',unsafe_allow_html=True)

with st.spinner("בודק נתונים ומחשב… מקור שאינו מגיב עלול להאריך את הטעינה"):
    v=fetch_close("^VIX",period="6mo")
    vo_intra=fetch_ohlc("^VIX",period="60d",interval="60m")
    v_intra=vo_intra["Close"] if vo_intra is not None else None
    v9=fetch_close("^VIX9D",period="6mo")
    v3=fetch_close("^VIX3M",period="6mo")
    vv=fetch_close("^VVIX",period="6mo")
    # Institutional / options-market layer. These are optional: the core model keeps
    # running when one source is temporarily unavailable, and the UI reports coverage.
    v1=fetch_close("^VIX1D",period="6mo")
    skew=fetch_close("^SKEW",period="6mo")
    cor1m=fetch_close("^COR1M",period="6mo")
    m=fetch_close(market_ticker,period="6mo")
    mo_intra=fetch_ohlc(market_ticker,period="60d",interval="60m")
    m_intra=mo_intra["Close"] if mo_intra is not None else None

# Revalidate cached frames as the session changes.
v, v9, v3, vv, v1, skew, cor1m, m = [
    x if _valid_history(x,"6mo","1d") else None
    for x in (v,v9,v3,vv,v1,skew,cor1m,m)
]
required_daily = {
    "VIX": v,
    "VIX9D": v9,
    "VIX3M": v3,
    market_name: m,
}
failed_daily = [name for name, x in required_daily.items() if x is None or len(x) < 30]
if failed_daily:
    st.error("לא הצלחתי להשיג נתונים מלאים ועדכניים עבור: " + ", ".join(failed_daily) + ". ניסיתי Yahoo, מקור Yahoo ישיר, Cboe הרשמי, FRED ובמידת האפשר גם Stooq.")
    with st.expander("אבחון מקורות נתונים"):
        for name, x in required_daily.items():
            st.write(f"{name}: {'✅' if x is not None and len(x)>=30 else '❌'} · {source_name(x)} · {len(x) if x is not None else 0} נקודות")
    st.stop()

V=float(v.iloc[-1]); V9=float(v9.iloc[-1]); V3=float(v3.iloc[-1]); VV=float(vv.iloc[-1]) if vv is not None and len(vv) else np.nan
V1=float(v1.iloc[-1]) if v1 is not None and len(v1) else np.nan
SKEW=float(skew.iloc[-1]) if skew is not None and len(skew) else np.nan
COR1M=float(cor1m.iloc[-1]) if cor1m is not None and len(cor1m) else np.nan
V12=resample_close(v_intra,12)
M12=resample_close(m_intra,12)
VO12=resample_ohlc(vo_intra,12)
MO12=resample_ohlc(mo_intra,12)
if V12 is None or len(V12)<30 or VO12 is None or len(VO12)<20 or M12 is None or len(M12)<20 or MO12 is None or len(MO12)<20:
    failed_intraday=[]
    if V12 is None or len(V12)<30: failed_intraday.append("VIX Close 60m → 12H")
    if VO12 is None or len(VO12)<20: failed_intraday.append("VIX OHLC 60m → 12H")
    if M12 is None or len(M12)<20: failed_intraday.append(f"{market_name} 60m → 12H")
    if MO12 is None or len(MO12)<20: failed_intraday.append(f"{market_name} OHLC 60m → 12H")
    st.error("אין כרגע מספיק נתונים תוך־יומיים לבניית הניתוח. פירוט המקורות מופיע למטה.")
    with st.expander("אבחון מקורות נתונים", expanded=True):
        st.write("רכיבים חסרים: " + ", ".join(failed_intraday))
        st.write(f"VIX Close 60m: {source_name(v_intra)} · {len(v_intra) if v_intra is not None else 0} נקודות")
        st.write(f"VIX OHLC 60m: {source_name(vo_intra)} · {len(vo_intra) if vo_intra is not None else 0} נקודות")
        st.write(f"{market_name} 60m: {source_name(m_intra)} · {len(m_intra) if m_intra is not None else 0} נקודות")
        st.write(f"{market_name} OHLC 60m: {source_name(mo_intra)} · {len(mo_intra) if mo_intra is not None else 0} נקודות")
        for label,frame in [("VIX",vo_intra),(market_name,mo_intra)]:
            if frame is not None and len(frame):
                st.code(f"{label}: last={frame.index[-1]} | recent timestamps=" + ", ".join(str(t) for t in frame.index[-8:]),language=None)
        st.caption("נדרש סיכום סשן מנתונים תוך־יומיים מלאים. המערכת תומכת בחותמות זמן שעתיות שונות ואינה מזיזה נרות או משלימה מחירים חסרים.")
    st.stop()
V4=resample_close(v_intra,4)
M4=resample_close(m_intra,4)
expected_day=latest_daily_session()
if (V4 is None or len(V4)<30 or M4 is None
    or V12.index[-1].tz_localize(None).normalize()<expected_day
    or M12.index[-1].tz_localize(None).normalize()<expected_day
    or V4.index[-1] != latest_bucket_start(4,obj=v_intra)
    or M4.index[-1] != latest_bucket_start(4,obj=m_intra)
    or V12.index[-1] != latest_bucket_start(12,obj=v_intra)
    or M12.index[-1] != latest_bucket_start(12,obj=m_intra)
    or not fresh_history(v_intra,"60m") or not fresh_history(m_intra,"60m")):
    st.error("ממתינים לנתונים מלאים ומסונכרנים של VIX והמדד. אין איתות כאשר נתוני המקור ישנים או חסרים.")
    st.stop()
V12_LAST=float(V12.iloc[-1])
VE9,VE26,CROSS,CROSS_AGE,APPROACH,CROSS_NEAR,GAP=cross_status(V12)
C1=pctn(v,1); C5=pctn(v,5); R9=V9/V; R3=V/V3
R1=(V1/V) if np.isfinite(V1) and V>0 else np.nan
R1_9=(V1/V9) if np.isfinite(V1) and V9>0 else np.nan
M=float(m.iloc[-1]); M2=pctn(m,2); M5=pctn(m,5)
DIV4=detect_divergence(V4, lookback=60, pivot=3, min_sep=4, min_price_pct=0.75, min_rsi_delta=4.0, recent_bars=8)
DIV12=detect_divergence(V12, lookback=50, pivot=2, min_sep=3, min_price_pct=1.0, min_rsi_delta=4.0, recent_bars=6)
SWING12=swing_exhaustion_signal(V12, lookback=50, pivot=2, recent_bars=7, min_swing_pct=1.5, rejection_pct=1.5)
SR, SUPPORT, RESISTANCE=support_resistance_signal(MO12,44,2)
VIX_SR, VIX_SUPPORT, VIX_RESISTANCE=support_resistance_signal(VO12,44,2)
FIB12=smart_fib_state(VO12,lookback=55,pivot=2,min_impulse_pct=7.0)

# Institutional pressure metrics. Use relative/z-score logic rather than fixed raw levels
# so the model adapts across calm and stressed volatility regimes.
VV5=pctn(vv,5) if vv is not None else np.nan
VV_RATIO=aligned_ratio(vv,v)
VV_Z=rolling_zscore_last(VV_RATIO,60) if VV_RATIO is not None else np.nan
VV_REL=(VV5-C5) if np.isfinite(VV5) and np.isfinite(C5) else np.nan
SKEW5=pctn(skew,5) if skew is not None else np.nan
SKEW_Z=rolling_zscore_last(skew,60) if skew is not None else np.nan
COR5=pctn(cor1m,5) if cor1m is not None else np.nan
COR_Z=rolling_zscore_last(cor1m,60) if cor1m is not None else np.nan

score=0.0; reasons=[]
category_scores={
    "VIX Reversal":0.0,
    "Institutional":0.0,
    "Vol Curve":0.0,
    "Market Confirmation":0.0,
    "EMA":0.0,
}

def addcat(cat,p,t):
    global score
    p=float(p)
    category_scores[cat]+=p
    score+=p
    reasons.append((p,t,cat))

# -----------------------------------------------------------------------------
# 1) VIX REVERSAL / STRUCTURE — ~45% of total model weight
# Divergence is intentionally the dominant signal. Higher-high/lower-low swing
# structure is scored only after rejection/rebound confirmation, never by itself.
# -----------------------------------------------------------------------------
if DIV12=="bullish": addcat("VIX Reversal", 1.90,"VIX 12H Bullish RSI Divergence → VIX UP / מדד SHORT")
elif DIV12=="bearish": addcat("VIX Reversal",-1.90,"VIX 12H Bearish RSI Divergence → VIX DOWN / מדד LONG")

if DIV4=="bullish": addcat("VIX Reversal", 1.00,"VIX 4H Bullish RSI Divergence → SHORT")
elif DIV4=="bearish": addcat("VIX Reversal",-1.00,"VIX 4H Bearish RSI Divergence → LONG")

if DIV4==DIV12=="bullish": addcat("VIX Reversal", 0.45,"סנכרון Divergence 4H+12H → SHORT")
elif DIV4==DIV12=="bearish": addcat("VIX Reversal",-0.45,"סנכרון Divergence 4H+12H → LONG")

if SWING12=="higher_high_rejection":
    addcat("VIX Reversal",-0.65,"VIX יצר שיא עולה ודחה מטה → פוטנציאל ירידת VIX / LONG")
elif SWING12=="lower_low_rebound":
    addcat("VIX Reversal", 0.30,"VIX יצר שפל יורד וחזר מעלה → פוטנציאל עליית VIX / SHORT")

# VIX S/R is supportive, but deliberately smaller than divergence.
if VIX_SR=="bounce_support": addcat("VIX Reversal", 0.40,"VIX 12H תגובה מתמיכה → SHORT")
elif VIX_SR=="reject_resistance": addcat("VIX Reversal",-0.40,"VIX 12H דחייה מהתנגדות → LONG")
elif VIX_SR=="breakout": addcat("VIX Reversal", 0.50,"VIX 12H פריצה מעל התנגדות → SHORT")
elif VIX_SR=="breakdown": addcat("VIX Reversal",-0.50,"VIX 12H שבירה מתחת לתמיכה → LONG")

# Smart Fib is supportive only. It never scores from location alone: the route to
# 0.50-0.618 or the reaction from it needs at least two independent confirmations.
rise_conf=sum([
    DIV12=="bullish", DIV4=="bullish", VIX_SR in ("breakout","bounce_support"),
    V12_LAST>VE9, VE9>VE26, len(V4)>=2 and float(V4.iloc[-1])>float(V4.iloc[-2])
])
fall_conf=sum([
    DIV12=="bearish", DIV4=="bearish", VIX_SR in ("breakdown","reject_resistance"),
    V12_LAST<VE9, VE9<VE26, len(V4)>=2 and float(V4.iloc[-1])<float(V4.iloc[-2])
])
if FIB12.get("valid"):
    fp=FIB12["phase"]
    if fp=="approaching_up" and rise_conf>=2:
        addcat("VIX Reversal", 0.45,"Smart Fib: VIX עולה לכיוון 0.50–0.618 + אישורים → SHORT במדד")
    elif fp=="approaching_down" and fall_conf>=2:
        addcat("VIX Reversal",-0.45,"Smart Fib: VIX יורד לכיוון 0.50–0.618 + אישורים → LONG במדד")
    elif fp=="reject_down" and fall_conf>=2:
        addcat("VIX Reversal",-0.55,"Smart Fib: דחיית VIX מטה מאזור 0.50–0.618 → LONG במדד")
    elif fp=="rebound_up" and rise_conf>=2:
        addcat("VIX Reversal", 0.55,"Smart Fib: rebound של VIX מאזור 0.50–0.618 → SHORT במדד")
    elif fp=="break_above_0618" and rise_conf>=2:
        addcat("VIX Reversal", 0.55,"Smart Fib: VIX פרץ 0.618 כלפי מעלה + אישורים → המשך SHORT במדד")
    elif fp=="break_below_0618" and fall_conf>=2:
        addcat("VIX Reversal",-0.55,"Smart Fib: VIX שבר 0.618 מטה + אישורים → המשך LONG במדד")

# -----------------------------------------------------------------------------
# 2) INSTITUTIONAL VOLATILITY PRESSURE — 25%
# Approximation of professional options-desk pressure using public Cboe indices:
# VVIX relative pressure, VIX1D short-end curve, SKEW tail demand, COR1M correlation.
# Every component is optional; missing data is shown as reduced coverage, not guessed.
# -----------------------------------------------------------------------------
institutional_available=0.0
institutional_total=2.5

if vv is not None and len(vv)>=30 and np.isfinite(VV_Z) and np.isfinite(VV_REL):
    institutional_available += 0.9
    if VV_Z>=1.0 and VV_REL>=4.0:
        addcat("Institutional", 0.90,"VVIX מוביל את VIX משמעותית → Hidden Vol Pressure / SHORT")
    elif VV_Z>=0.5 and VV_REL>=2.0:
        addcat("Institutional", 0.55,"VVIX מתחזק יחסית ל-VIX → לחץ סמוי / SHORT")
    elif VV_Z<=-1.0 and VV_REL<=-4.0:
        addcat("Institutional",-0.90,"VVIX נחלש משמעותית מול VIX → Vol Relief / LONG")
    elif VV_Z<=-0.5 and VV_REL<=-2.0:
        addcat("Institutional",-0.55,"VVIX נחלש יחסית ל-VIX → רגיעה / LONG")

if v1 is not None and len(v1)>=20 and np.isfinite(R1) and np.isfinite(R1_9):
    institutional_available += 0.8
    if R1_9>=1.05 and R9>=1.02:
        addcat("Institutional", 0.80,"VIX1D > VIX9D > VIX → לחץ בקצה הקצר / SHORT")
    elif R1>=1.10:
        addcat("Institutional", 0.55,"VIX1D בפרמיה חדה ל-VIX → Event Risk / SHORT")
    elif R1_9<=0.90 and R9<=0.98:
        addcat("Institutional",-0.80,"VIX1D < VIX9D < VIX → רגיעה בקצה הקצר / LONG")
    elif R1<=0.85 and R9<=1.00:
        addcat("Institutional",-0.40,"VIX1D נמוך משמעותית מ-VIX → לחץ מיידי נמוך / LONG")

if skew is not None and len(skew)>=30 and np.isfinite(SKEW_Z) and np.isfinite(SKEW5):
    institutional_available += 0.4
    if SKEW_Z>=0.75 and SKEW5>=2.0:
        addcat("Institutional", 0.40,"SKEW עולה ומעל הנורמה → ביקוש Tail Risk / SHORT")
    elif SKEW_Z<=-0.75 and SKEW5<=-2.0:
        addcat("Institutional",-0.40,"SKEW נחלש ומתחת לנורמה → ירידת Tail Demand / LONG")

if cor1m is not None and len(cor1m)>=30 and np.isfinite(COR_Z) and np.isfinite(COR5):
    institutional_available += 0.4
    if COR_Z>=0.50 and COR5>=3.0:
        addcat("Institutional", 0.40,"COR1M עולה → סיכון מערכתי / Herding / SHORT")
    elif COR_Z<=-0.50 and COR5<=-3.0:
        addcat("Institutional",-0.40,"COR1M יורד → פיזור סיכון משתפר / LONG")

# -----------------------------------------------------------------------------
# 3) VOL CURVE / VIX IMPULSE — 15%
# Existing VIX9D/VIX and VIX/VIX3M remain important but are no longer allowed to
# overwhelm reversal structure. Daily/5D VIX moves are compressed into one impulse.
# -----------------------------------------------------------------------------
if R9>=1.03: addcat("Vol Curve", 0.50,"VIX9D בפרמיה ל-VIX → לחץ קצר טווח")
elif R9<=0.97: addcat("Vol Curve",-0.50,"VIX9D מתחת ל-VIX → רגיעה קצרה")

if R3>=1.02: addcat("Vol Curve", 0.50,"VIX/VIX3M Backwardation → Risk-Off")
elif R3<=0.94: addcat("Vol Curve",-0.50,"VIX/VIX3M Contango → Risk-On")

if C1>=8 or C5>=10: addcat("Vol Curve", 0.40,"VIX impulse חזק מעלה")
elif C1>=3 or C5>=4: addcat("Vol Curve", 0.25,"VIX impulse מעלה")
elif C1<=-8 or C5<=-10: addcat("Vol Curve",-0.40,"VIX impulse חזק מטה")
elif C1<=-3 or C5<=-4: addcat("Vol Curve",-0.25,"VIX impulse מטה")

if V>=30: addcat("Vol Curve", 0.10,"VIX ≥30")
elif V<15: addcat("Vol Curve",-0.10,"VIX <15")

# -----------------------------------------------------------------------------
# 4) MARKET CONFIRMATION — 10%
# Nasdaq/S&P confirms the volatility read; it does not lead the VIX model.
# -----------------------------------------------------------------------------
if M2<=-0.5: addcat("Market Confirmation", 0.15,f"{market_name} מומנטום 2D שלילי")
elif M2>=0.5: addcat("Market Confirmation",-0.15,f"{market_name} מומנטום 2D חיובי")

if M5<=-1.0: addcat("Market Confirmation", 0.05,f"{market_name} מומנטום 5D שלילי")
elif M5>=1.0: addcat("Market Confirmation",-0.05,f"{market_name} מומנטום 5D חיובי")

if SR=="reject_resistance": addcat("Market Confirmation", 0.80,f"{market_name}: דחייה מהתנגדות 12H")
elif SR=="breakout": addcat("Market Confirmation",-0.80,f"{market_name}: פריצה מעל התנגדות 12H")
elif SR=="bounce_support": addcat("Market Confirmation",-0.80,f"{market_name}: תגובה מתמיכה 12H")
elif SR=="breakdown": addcat("Market Confirmation", 0.80,f"{market_name}: שבירה מתחת לתמיכה 12H")

# -----------------------------------------------------------------------------
# 5) EMA 9/26 — only 5%
# Kept as a small trend filter per design request; never a dominant signal.
# -----------------------------------------------------------------------------
addcat("EMA", 0.15 if V12_LAST>VE9 else -0.15,
       "VIX 12H מעל EMA9" if V12_LAST>VE9 else "VIX 12H מתחת EMA9")
addcat("EMA", 0.20 if VE9>VE26 else -0.20,
       "EMA9 מעל EMA26" if VE9>VE26 else "EMA9 מתחת EMA26")
if CROSS=="golden" and CROSS_AGE is not None and CROSS_AGE<=5:
    addcat("EMA", 0.15,"Golden Cross טרי ב-VIX")
elif CROSS=="death" and CROSS_AGE is not None and CROSS_AGE<=5:
    addcat("EMA",-0.15,"Death Cross טרי ב-VIX")

# Public-source data coverage. Core model = 75%; institutional layer = 25%.
data_coverage = 75.0 + 25.0 * (institutional_available / institutional_total)

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

# Strong labels require the defining reversal evidence.
if signal_score >= STRONG_THRESHOLD:
    matching_div = "bullish" if signed_signal > 0 else "bearish"
    if matching_div not in (DIV4,DIV12):
        state,icon,cls = "WAIT","🟡","yellow"

# Gauge: -10 = LONG, 0 = neutral, +10 = SHORT.
pos = max(2, min(98, (signed_signal+10)/20*100))
if state == "WAIT":
    lean = "SHORT" if signed_signal > 0 else ("LONG" if signed_signal < 0 else "NEUTRAL")
    score_line = f"עוצמת איתות <b>{signal_score:.1f}/10</b> · נטייה {lean}"
else:
    score_line = f"עוצמת איתות <b>{signal_score:.1f}/10</b>"

st.markdown(f"""
<div class="signal">
  <div class="signal-title {cls}">{icon} {state}</div>
  <div class="score">{score_line}</div>
  <div class="confidence">סף חיפוש עסקה: <b>5.0/10</b> · כיסוי נתונים <b>{data_coverage:.0f}%</b></div>
  <div class="gauge-wrap">
    <div class="pointer" style="left:{pos:.1f}%"></div>
    <div class="gauge"><div class="midline"></div></div>
    <div class="gauge-labels"><span>LONG 10</span><span>WAIT 0</span><span>SHORT 10</span></div>
    <div class="gauge-zones"><span>LONG ≤ -5</span><span>WAIT</span><span>SHORT ≥ +5</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.caption(f"נתונים יומיים: {v.index[-1]:%d/%m/%Y} · נר סשן אחרון (זמן פתיחה, ניו יורק): {V12.index[-1]:%d/%m %H:%M} · נר 4H: {V4.index[-1]:%d/%m %H:%M}")
st.caption("הנתונים היומיים מבוססים על הסשן האחרון שנסגר, וממתינים שעה לאחר הסגירה לפרסום. אין כאן מחירי זמן אמת. הציון הוא סכום משקלים, לא אחוז הצלחה. כיסוי הנתונים אינו מדד לאיכות העסקה. אין ביצוע עסקאות אוטומטי.")
if data_coverage < 100:
    missing = [name for name,x in [("VVIX",vv),("VIX1D",v1),("SKEW",skew),("COR1M",cor1m)] if x is None]
    st.warning("כיסוי חלקי — רכיבים חסרים או לא עדכניים: " + ", ".join(missing))

# Top drivers
top_drivers = sorted(reasons, key=lambda z: abs(z[0]), reverse=True)[:3]
driver_rows = []
for pts, txt, cat in top_drivers:
    badge_cls = "red" if pts > 0 else ("green" if pts < 0 else "white")
    badge_label = f"{pts:+.2f}"
    driver_rows.append(
        f'<div class="driver-row"><div class="driver-text">{txt}</div><div class="driver-badge {badge_cls}">{badge_label}</div></div>'
    )
if driver_rows:
    st.markdown(
        '<div class="drivers-card"><div class="drivers-title">שלושת הגורמים המרכזיים</div>' + ''.join(driver_rows) + '</div>',
        unsafe_allow_html=True
    )

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

def vix_sr_text(s):
    return {
        "reject_resistance":("התנגדות → LONG","green"),
        "breakout":("Breakout → SHORT","red"),
        "bounce_support":("תמיכה → SHORT","red"),
        "breakdown":("Breakdown → LONG","green"),
        "neutral":("ניטרלי","white")
    }[s]

def swing_text(s):
    return {
        "higher_high_rejection":("שיא עולה + דחייה → LONG","green"),
        "lower_low_rebound":("שפל יורד + חזרה → SHORT","red"),
        "none":("אין אישור","white")
    }.get(s,("אין אישור","white"))

def fib_text(f):
    if not f.get("valid"):
        return "אין Impulse ברור","white"
    p=f["phase"]
    mapping={
        "approaching_up":("VIX בדרך לאזור → SHORT Watch","orange"),
        "approaching_down":("VIX בדרך לאזור → LONG Watch","blue"),
        "decision_zone":("0.50–0.618 · DECISION ZONE","yellow"),
        "reject_down":("דחייה מטה מהאזור → LONG","green"),
        "rebound_up":("Rebound מהאזור → SHORT","red"),
        "break_above_0618":("פריצה מעל 0.618 → SHORT","red"),
        "break_below_0618":("שבירה מתחת 0.618 → LONG","green"),
        "below_zone":("מתחת לאזור · ממתין למומנטום","white"),
        "above_zone":("מעל לאזור · ממתין למומנטום","white"),
        "tracking":("מעקב","white"),
    }
    return mapping.get(p,(p,"white"))

def institutional_text(v):
    if v>=0.45: return f"Pressure {v:+.2f}/2.5 → SHORT","red"
    if v<=-0.45: return f"Relief {v:+.2f}/2.5 → LONG","green"
    return f"Neutral {v:+.2f}/2.5","white"

d4,d4c=div_text(DIV4); d12,d12c=div_text(DIV12); srt,src=sr_text(SR); vsrt,vsrc=vix_sr_text(VIX_SR)
swingt,swingc=swing_text(SWING12); instt,instc=institutional_text(category_scores["Institutional"])
fibt,fibc=fib_text(FIB12)
term_text="Risk-Off" if R3>=1.02 else ("Risk-On" if R3<=0.94 else "ניטרלי")
term_cls="red" if R3>=1.02 else ("green" if R3<=0.94 else "white")

st.markdown('<div class="status-grid">'+
    status_row("Divergence 12H · HIGH WEIGHT",d12,d12c)+
    status_row("Divergence 4H",d4,d4c)+
    status_row("VIX Swing Structure · 12H",swingt,swingc)+
    status_row("Smart Fib 0.50–0.618 · 12H",fibt,fibc)+
    status_row("Institutional Vol Pressure",instt,instc)+
    status_row("VIX S/R · 12H",vsrt,vsrc)+
    status_row(f"{market_name} S/R · 12H",srt,src)+
    status_row("Term Structure",term_text,term_cls)+
    status_row("VIX9D / VIX",("לחץ" if R9>=1.03 else ("רגוע" if R9<=0.97 else "מאוזן")),"red" if R9>=1.03 else ("green" if R9<=0.97 else "white"))+
    status_row("EMA9/26 · 12H · LOW WEIGHT",ema_bias,ema_cls)+
    status_row("Cross · EMA9/26",cross_text,cross_cls)+
    '</div>',unsafe_allow_html=True)

# Smart Fib detail panel
if FIB12.get("valid"):
    arrow="↓" if FIB12["direction"]=="down" else "↑"
    st.markdown(
        f'<div class="panel"><b>Smart Fib 12H {arrow}</b><br>'
        f'Impulse: <bdi dir="ltr">{FIB12["start"]:.2f} → {FIB12["end"]:.2f}</bdi> · '
        f'אזור החלטה <bdi dir="ltr">0.50–0.618 = {FIB12["zone_low"]:.2f}–{FIB12["zone_high"]:.2f}</bdi><br>'
        f'VIX נוכחי <bdi dir="ltr">{FIB12["current"]:.2f}</bdi> · {fibt} · '
        f'אישורי עלייה <bdi dir="ltr">{rise_conf}/6</bdi> · אישורי ירידה <bdi dir="ltr">{fall_conf}/6</bdi><br>'
        '<span class="muted">הפיבונאצ׳י הוא אזור יעד/תגובה בלבד. ניקוד ניתן רק עם לפחות 2 אישורים נוספים.</span></div>',
        unsafe_allow_html=True
    )

# Minimal execution reminder
if "SHORT" in state: action="חפש אישור SHORT במדד שנבחר; זה אינו אישור כניסה אוטומטי"
elif "LONG" in state: action="חפש אישור LONG במדד שנבחר; זה אינו אישור כניסה אוטומטי"
else: action="אין עסקה — המתן לסנכרון"
st.markdown(f'<div class="panel" style="text-align:center;font-weight:900">{action}</div>',unsafe_allow_html=True)

with st.expander("פירוט החישוב"):
    st.write("**תקרות משקל בקירוב (לא אחוזי הצלחה):** VIX Divergence/Structure + Smart Fib ~45–50% · Institutional Vol 25% · Vol Curve 15% · Market Confirmation 10% · EMA9/26 5%")
    st.caption("Smart Fib נוסף כרכיב תומך בלבד (עד ±0.55) ודורש לפחות שני אישורים נוספים. אין כאן טענה לרווחיות או לכיול אופטימלי ללא בק־טסט/Forward Test ייעודי.")
    st.write(f"VIX 12H {V12_LAST:.2f} · EMA9 {VE9:.2f} · EMA26 {VE26:.2f} · Cross proximity {CROSS_NEAR}/10")
    st.write(f"Divergence: 4H={DIV4} · 12H={DIV12} · Swing 12H={SWING12}")
    st.write(f"VIX 1D {C1:+.2f}% · 5D {C5:+.2f}% · VIX9D/VIX {R9:.3f} · VIX/VIX3M {R3:.3f}")
    if np.isfinite(V1):
        st.write(f"VIX1D {V1:.2f} · VIX1D/VIX {R1:.3f} · VIX1D/VIX9D {R1_9:.3f}")
    if np.isfinite(VV):
        st.write(f"VVIX {VV:.2f} · VVIX/VIX z-score {VV_Z:+.2f} · Relative 5D momentum {VV_REL:+.2f}pp")
    if np.isfinite(SKEW):
        st.write(f"SKEW {SKEW:.2f} · 5D {SKEW5:+.2f}% · z-score {SKEW_Z:+.2f}")
    if np.isfinite(COR1M):
        st.write(f"COR1M {COR1M:.2f} · 5D {COR5:+.2f}% · z-score {COR_Z:+.2f}")
    st.write(f"{market_name}: מומנטום 2D {M2:+.2f}% · 5D {M5:+.2f}% · S/R 12H: {SR} · תמיכה {SUPPORT:.2f} · התנגדות {RESISTANCE:.2f}")
    st.write(f"VIX S/R 12H: {VIX_SR} · תמיכה {VIX_SUPPORT:.2f} · התנגדות {VIX_RESISTANCE:.2f}")
    if FIB12.get("valid"):
        st.write(f"Smart Fib 12H: direction={FIB12['direction']} · phase={FIB12['phase']} · anchor {FIB12['start']:.2f}→{FIB12['end']:.2f} · zone {FIB12['zone_low']:.2f}–{FIB12['zone_high']:.2f} · rise_conf={rise_conf}/6 · fall_conf={fall_conf}/6")
    st.write(f"כיסוי נתונים: {data_coverage:.0f}% · Institutional availability {institutional_available:.1f}/{institutional_total:.1f}")
    st.write("RSI עצמו אינו מקבל נקודות. רק Divergence מאומת מקבל משקל, כדי להימנע מ-RSI overbought/oversold פשוט שעלול להטעות ב-VIX.")
    st.write(f"ציון משוקלל נטו: {score:+.2f} → עוצמת איתות {signal_score:.2f}/10 · כיוון: {'SHORT' if signed_signal>0 else ('LONG' if signed_signal<0 else 'NEUTRAL')}")
    st.write("**ציוני קטגוריות:** " + " · ".join([f"{k}: {v:+.2f}" for k,v in category_scores.items()]))
    for pts,txt,cat in sorted(reasons,key=lambda z:abs(z[0]),reverse=True):
        st.write(f"**{pts:+.2f}** — {txt}  ·  _{cat}_")

with st.expander("מקורות נתונים / גיבוי"):
    st.write(f"VIX Daily: **{source_name(v)}**")
    st.write(f"VIX9D Daily: **{source_name(v9)}**")
    st.write(f"VIX3M Daily: **{source_name(v3)}**")
    st.write(f"VVIX Daily: **{source_name(vv)}**")
    st.write(f"VIX1D Daily: **{source_name(v1)}**")
    st.write(f"SKEW Daily: **{source_name(skew)}**")
    st.write(f"COR1M Daily: **{source_name(cor1m)}**")
    st.write(f"{market_name} Daily: **{source_name(m)}**")
    st.write(f"VIX Close 60m: **{source_name(v_intra)}**")
    st.write(f"VIX OHLC 60m: **{source_name(vo_intra)}**")
    st.write(f"{market_name} 60m: **{source_name(m_intra)}**")
    st.caption("גיבוי אמיתי: Yahoo/yfinance → Yahoo Chart API ישיר → Cboe הרשמי למדדי תנודתיות/אופציות. FRED משמש ל-VIX/VIX3M במידת הצורך, ו-Stooq למדדי NDX/SPX. רכיב Institutional הוא אופציונלי: מקור חסר מוריד Data Coverage ואינו מוחלף בנתון מומצא.")

st.caption(f"עודכן {pd.Timestamp.now(tz='Asia/Jerusalem').strftime('%H:%M')} · v10.7 Smart Fib · שעון ישראל · Multi-Source + Retry פעיל · כלי מחקרי, לא ייעוץ השקעות")
