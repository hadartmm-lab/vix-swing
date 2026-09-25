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


def closed_session_hourly(obj, now=None):
    """Return only fully closed regular-session hourly source bars.

    This prevents the 1H trigger from reading a still-forming Yahoo candle.
    """
    x=clean_history(obj)
    if x is None:
        return None
    now=utc_now(now); phase=feed_phase(x)
    idx=x.index
    idx=idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
    x=x.copy(); x.index=idx.tz_convert('UTC')
    sched=schedule(now)
    sched=sched[(sched['close']>=x.index.min())&(sched['open']<=now)]
    keep=[]
    for _,day in sched.iterrows():
        for t in hourly_grid(day['open'],day['close'],phase):
            if min(t+pd.Timedelta(hours=1),day['close'])+pd.Timedelta(minutes=15)<=now and t in x.index:
                keep.append(t)
    if not keep:
        return None
    out=x.loc[pd.DatetimeIndex(keep)].copy()
    out.index=out.index.tz_convert(TZ)
    out.attrs=obj.attrs.copy()
    return out


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

st.set_page_config(page_title="VIX Tactical", page_icon="🎯", layout="centered")

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
    "User-Agent": "Mozilla/5.0 (VIX-Tactical/11.0; Streamlit)",
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
    "^VIX6M": "VIX6M",
    "^VIX1Y": "VIX1Y",
    "DSPX": "DSPX",
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
            if price_change <= -min_price_pct and rsi_change >= min_rsi_delta and x.iloc[-1]>=x.iloc[b]:
                return "bullish"

    # Bearish divergence on VIX: higher VIX high + lower RSI high -> supports VIX fade / QQQ long.
    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if valid_pair(a, b):
            price_change = (x.iloc[b] / x.iloc[a] - 1) * 100
            rsi_change = float(r.iloc[b] - r.iloc[a])
            if price_change >= min_price_pct and rsi_change <= -min_rsi_delta and x.iloc[-1]<=x.iloc[b]:
                return "bearish"

    return "none"

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
    """Two-way Smart Fibonacci on closed VIX OHLC bars (used on 4H and 12H).

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
    if (direction=='up' and cur<=start) or (direction=='down' and cur>=start):
        return empty
    move=abs(end-start)/start*100
    recent=x.iloc[max(end_i+1,len(x)-3):]
    mom=cur-float(x['Close'].iloc[-3])
    tol=max(0.04,0.015*rng)
    break_buf=max(0.03,0.008*rng)
    phase='tracking'
    # The retracement route and the reaction are intentionally symmetric.
    if direction=='down':
        touched=float(recent['High'].max()) >= zone_low-tol
        if cur < zone_low-tol:
            phase='reject_down' if touched and cur<prev and mom<0 else ('approaching_up' if mom>0 else 'below_zone')
        elif cur <= zone_high+tol:
            phase='reject_down' if touched and cur<prev and mom<0 else 'decision_zone'
        else:
            # A move beyond 0.618 is NOT an entry by itself. It means the correction
            # has gone deeper; wait to see whether the old downtrend resumes.
            phase='deep_retracement_up'
    else:
        touched=float(recent['Low'].min()) <= zone_high+tol
        if cur > zone_high+tol:
            phase='rebound_up' if touched and cur>prev and mom>0 else ('approaching_down' if mom<0 else 'above_zone')
        elif cur >= zone_low-tol:
            phase='rebound_up' if touched and cur>prev and mom>0 else 'decision_zone'
        else:
            # A move beyond 0.618 is NOT an entry by itself. It means the correction
            # has gone deeper; wait to see whether the old uptrend resumes.
            phase='deep_retracement_down'
    if cur<zone_low: dist=(zone_low-cur)/cur*100
    elif cur>zone_high: dist=(cur-zone_high)/cur*100
    else: dist=0.0
    return {"valid":True,"direction":direction,"start":start,"end":end,
            "start_time":x.index[start_i],"end_time":x.index[end_i],
            "level50":float(level50),"level618":float(level618),
            "zone_low":float(zone_low),"zone_high":float(zone_high),"current":cur,
            "phase":phase,"distance_pct":float(dist),"impulse_pct":float(move)}


def repeated_sr_zone(df, lookback=50, pivot=2, tol_pct=1.2, min_touches=3):
    """Find repeated VIX support/resistance zones from clustered pivot candles.

    A level is considered meaningful only when several separate pivot candles touched
    roughly the same price area. This is intentionally stricter than a single pivot.
    """
    out={"support":np.nan,"resistance":np.nan,"support_touches":0,"resistance_touches":0,
         "near_support":False,"near_resistance":False,"state":"none"}
    x=clean_history(df)
    if x is None or len(x)<max(18,pivot*2+8):
        return out
    x=x.iloc[-lookback:].copy()
    highs=[]; lows=[]
    for i in range(pivot,len(x)-pivot):
        wh=x['High'].iloc[i-pivot:i+pivot+1]
        wl=x['Low'].iloc[i-pivot:i+pivot+1]
        if x['High'].iloc[i]==wh.max() and (wh==x['High'].iloc[i]).sum()==1: highs.append(float(x['High'].iloc[i]))
        if x['Low'].iloc[i]==wl.min() and (wl==x['Low'].iloc[i]).sum()==1: lows.append(float(x['Low'].iloc[i]))

    def best_cluster(vals):
        if len(vals)<min_touches: return (np.nan,0)
        best=(np.nan,0)
        for v0 in vals:
            tol=max(abs(v0)*tol_pct/100,1e-9)
            members=[v for v in vals if abs(v-v0)<=tol]
            if len(members)>best[1]:
                best=(float(np.mean(members)),len(members))
        return best

    support,st=best_cluster(lows); resistance,rt=best_cluster(highs)
    cur=float(x['Close'].iloc[-1]); hi=float(x['High'].iloc[-1]); lo=float(x['Low'].iloc[-1])
    near_s=bool(np.isfinite(support) and (abs(lo-support)/support*100<=tol_pct or abs(cur-support)/support*100<=tol_pct))
    near_r=bool(np.isfinite(resistance) and (abs(hi-resistance)/resistance*100<=tol_pct or abs(cur-resistance)/resistance*100<=tol_pct))
    state='resistance' if near_r and rt>=min_touches else ('support' if near_s and st>=min_touches else 'none')
    return {"support":support,"resistance":resistance,"support_touches":st,"resistance_touches":rt,
            "near_support":near_s,"near_resistance":near_r,"state":state}


def detect_clear_pattern(df, lookback=42, pivot=2):
    """Detect confirmed M/W or wedge breakout on VIX OHLC.

    The detector deliberately returns only confirmed/clear structures: M/W needs a
    neckline break and wedges need converging trendlines plus a breakout.
    """
    out={"pattern":"none","bias":"none","detail":""}
    x=clean_history(df)
    if x is None or len(x)<24:
        return out
    x=x.iloc[-lookback:].copy()
    highs=[]; lows=[]
    for i in range(pivot,len(x)-pivot):
        wh=x['High'].iloc[i-pivot:i+pivot+1]; wl=x['Low'].iloc[i-pivot:i+pivot+1]
        if x['High'].iloc[i]==wh.max() and (wh==x['High'].iloc[i]).sum()==1: highs.append((i,float(x['High'].iloc[i])))
        if x['Low'].iloc[i]==wl.min() and (wl==x['Low'].iloc[i]).sum()==1: lows.append((i,float(x['Low'].iloc[i])))
    close=float(x['Close'].iloc[-1])

    # Confirmed M: two similar highs, meaningful valley, then close below neckline.
    if len(highs)>=2:
        h1,h2=highs[-2],highs[-1]
        if h2[0]-h1[0]>=4 and len(x)-1-h2[0]<=8:
            between=x['Low'].iloc[h1[0]:h2[0]+1]
            neckline=float(between.min())
            peak_avg=(h1[1]+h2[1])/2
            if abs(h2[1]-h1[1])/peak_avg<=0.025 and (peak_avg/neckline-1)>=0.03 and close<neckline*0.998:
                return {"pattern":"M","bias":"bearish","detail":f"M מאושר · neckline {neckline:.2f}"}

    # Confirmed W: two similar lows, meaningful middle peak, then close above neckline.
    if len(lows)>=2:
        l1,l2=lows[-2],lows[-1]
        if l2[0]-l1[0]>=4 and len(x)-1-l2[0]<=8:
            between=x['High'].iloc[l1[0]:l2[0]+1]
            neckline=float(between.max())
            low_avg=(l1[1]+l2[1])/2
            if abs(l2[1]-l1[1])/low_avg<=0.025 and (neckline/low_avg-1)>=0.03 and close>neckline*1.002:
                return {"pattern":"W","bias":"bullish","detail":f"W מאושר · neckline {neckline:.2f}"}

    # Wedge: regress recent pivot highs/lows. Both boundaries must point the same way,
    # converge, and price must break the expected boundary.
    if len(highs)>=3 and len(lows)>=3:
        hh=highs[-4:]; ll=lows[-4:]
        hx=np.array([a for a,_ in hh],float); hy=np.array([b for _,b in hh],float)
        lx=np.array([a for a,_ in ll],float); ly=np.array([b for _,b in ll],float)
        hs,hi=np.polyfit(hx,hy,1); ls,li=np.polyfit(lx,ly,1)
        n=len(x)-1
        upper_now=hs*n+hi; lower_now=ls*n+li
        n0=max(0,n-12); upper_old=hs*n0+hi; lower_old=ls*n0+li
        width_old=upper_old-lower_old; width_now=upper_now-lower_now
        converging=width_old>0 and width_now>0 and width_now<=width_old*0.82
        if converging and hs>0 and ls>0 and ls>hs and close<lower_now*0.998:
            return {"pattern":"Bearish Wedge","bias":"bearish","detail":"יתד עולה/דובית מאושרת בשבירה מטה"}
        if converging and hs<0 and ls<0 and hs<ls and close>upper_now*1.002:
            return {"pattern":"Bullish Wedge","bias":"bullish","detail":"יתד יורדת/שורית מאושרת בפריצה מעלה"}
    return out


def detect_wedge_setup(df, lookback=55, pivot=2):
    """Detect a developing or confirmed wedge without waiting for a late pivot.

    This is intentionally used as an *early structure* detector. A developing rising
    wedge on VIX is bearish for VIX (QQQ/Nasdaq LONG watch); a developing falling
    wedge is bullish for VIX (QQQ/Nasdaq SHORT watch). A breakout upgrades the setup.
    """
    out={"pattern":"none","bias":"none","confirmed":False,"detail":""}
    x=clean_history(df)
    if x is None or len(x)<24:
        return out
    x=x.iloc[-lookback:].copy()
    highs=[]; lows=[]
    for i in range(pivot,len(x)-pivot):
        wh=x['High'].iloc[i-pivot:i+pivot+1]
        wl=x['Low'].iloc[i-pivot:i+pivot+1]
        if x['High'].iloc[i]==wh.max() and (wh==x['High'].iloc[i]).sum()==1:
            highs.append((i,float(x['High'].iloc[i])))
        if x['Low'].iloc[i]==wl.min() and (wl==x['Low'].iloc[i]).sum()==1:
            lows.append((i,float(x['Low'].iloc[i])))
    if len(highs)<3 or len(lows)<3:
        return out
    hh=highs[-4:]; ll=lows[-4:]
    hx=np.array([a for a,_ in hh],float); hy=np.array([b for _,b in hh],float)
    lx=np.array([a for a,_ in ll],float); ly=np.array([b for _,b in ll],float)
    hs,hi=np.polyfit(hx,hy,1); ls,li=np.polyfit(lx,ly,1)
    n=len(x)-1
    upper_now=hs*n+hi; lower_now=ls*n+li
    n0=max(0,n-14); upper_old=hs*n0+hi; lower_old=ls*n0+li
    width_old=upper_old-lower_old; width_now=upper_now-lower_now
    # Setup detection is looser than confirmed-pattern detection, but still requires
    # real convergence and same-direction boundaries.
    converging=width_old>0 and width_now>0 and width_now<=width_old*0.90
    close=float(x['Close'].iloc[-1])
    if converging and hs>0 and ls>0 and ls>hs:
        confirmed=close<lower_now*0.998
        return {"pattern":"Rising Wedge","bias":"bearish","confirmed":bool(confirmed),
                "detail":"יתד עולה/דובית " + ("מאושרת בשבירה מטה" if confirmed else "בבנייה")}
    if converging and hs<0 and ls<0 and hs<ls:
        confirmed=close>upper_now*1.002
        return {"pattern":"Falling Wedge","bias":"bullish","confirmed":bool(confirmed),
                "detail":"יתד יורדת/שורית " + ("מאושרת בפריצה מעלה" if confirmed else "בבנייה")}
    return out


def recent_zone_overlap(df, low, high, recent_bars=8, pad_pct=0.6):
    """Whether a recent closed candle overlapped a price zone."""
    x=clean_history(df)
    if x is None or not np.isfinite(low) or not np.isfinite(high):
        return False
    zlo=min(float(low),float(high)); zhi=max(float(low),float(high))
    pad=max(0.03,((zlo+zhi)/2.0)*pad_pct/100.0)
    r=x.iloc[-recent_bars:]
    return bool(((r['High']>=zlo-pad)&(r['Low']<=zhi+pad)).any())


def recent_level_rejection(df, level, side='resistance', recent_bars=8, tol_pct=1.25):
    """Detect a recent closed-candle rejection from a repeated S/R level."""
    x=clean_history(df)
    if x is None or not np.isfinite(level) or len(x)<4:
        return False
    r=x.iloc[-recent_bars:]
    level=float(level); tol=tol_pct/100.0
    if side=='resistance':
        touched=(r['High']>=level*(1-tol))
        rejected=touched & (r['Close']<level*(1-0.002)) & (r['Close']<r['Open'])
        if bool(rejected.any()): return True
        # Also retain the event after price has already moved away from the zone.
        if bool(touched.any()) and float(r['Close'].iloc[-1])<level*(1-0.02): return True
    else:
        touched=(r['Low']<=level*(1+tol))
        rejected=touched & (r['Close']>level*(1+0.002)) & (r['Close']>r['Open'])
        if bool(rejected.any()): return True
        if bool(touched.any()) and float(r['Close'].iloc[-1])>level*(1+0.02): return True
    return False


def reversal_confluence_12h(df, sr, fib, wedge):
    """12H early-warning engine.

    Core WATCH requires three independent location/structure facts:
      1) repeated S/R with >=3 touches,
      2) a recent Smart-Fib 0.50-0.618 overlap in the correct retracement direction,
      3) a developing/confirmed wedge in the reversal direction.
    Rejection is tracked separately and can upgrade WATCH to DEVELOPING together
    with 4H confirmation later in the pipeline.
    """
    out={"direction":"none","stage":"NONE","sr_ok":False,"fib_ok":False,
         "wedge_ok":False,"rejection":False,"detail":""}
    x=clean_history(df)
    if x is None or not fib.get('valid'):
        return out

    # VIX bearish reversal -> QQQ/Nasdaq LONG.
    if fib.get('direction')=='down' and wedge.get('bias')=='bearish':
        level=sr.get('resistance',np.nan); touches=sr.get('resistance_touches',0)
        sr_ok=bool(touches>=3 and np.isfinite(level) and recent_zone_overlap(x,level,level,8,1.25))
        fib_ok=recent_zone_overlap(x,fib.get('zone_low',np.nan),fib.get('zone_high',np.nan),8,0.6)
        rejection=recent_level_rejection(x,level,'resistance',8,1.25) if sr_ok else False
        if sr_ok and fib_ok:
            return {"direction":"LONG","stage":"WATCH","sr_ok":True,"fib_ok":True,
                    "wedge_ok":True,"rejection":bool(rejection),
                    "detail":f"12H: התנגדות {touches} נגיעות + Fib 0.50–0.618 + Rising Wedge"}

    # VIX bullish reversal -> QQQ/Nasdaq SHORT.
    if fib.get('direction')=='up' and wedge.get('bias')=='bullish':
        level=sr.get('support',np.nan); touches=sr.get('support_touches',0)
        sr_ok=bool(touches>=3 and np.isfinite(level) and recent_zone_overlap(x,level,level,8,1.25))
        fib_ok=recent_zone_overlap(x,fib.get('zone_low',np.nan),fib.get('zone_high',np.nan),8,0.6)
        rejection=recent_level_rejection(x,level,'support',8,1.25) if sr_ok else False
        if sr_ok and fib_ok:
            return {"direction":"SHORT","stage":"WATCH","sr_ok":True,"fib_ok":True,
                    "wedge_ok":True,"rejection":bool(rejection),
                    "detail":f"12H: תמיכה {touches} נגיעות + Fib 0.50–0.618 + Falling Wedge"}
    return out


def four_hour_followthrough(df, direction):
    """Conservative closed-candle follow-through check used only for ENTRY READY."""
    x=clean_history(df)
    if x is None or len(x)<5:
        return False
    c=x['Close']; h=x['High']; l=x['Low']
    if direction=='LONG':  # Want VIX continuing down.
        return bool(c.iloc[-1]<c.iloc[-2] and c.iloc[-1]<=l.iloc[-3:-1].min()*1.002)
    if direction=='SHORT': # Want VIX continuing up.
        return bool(c.iloc[-1]>c.iloc[-2] and c.iloc[-1]>=h.iloc[-3:-1].max()*0.998)
    return False

def atr_series(df, n=14):
    """Closed-bar ATR used only by the execution layer."""
    x=clean_history(df)
    if x is None or len(x)<n+2:
        return pd.Series(dtype=float)
    prev=x['Close'].shift(1)
    tr=pd.concat([(x['High']-x['Low']).abs(),(x['High']-prev).abs(),(x['Low']-prev).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()


def fast_continuation_setup(qqq1h, qqq4h, vix1h, vix4h, direction):
    """Second-chance execution layer after VIX has already established direction.

    QQQ never creates the direction and never adds points to the core score. The
    layer waits for a controlled 1H QQQ pullback, a closed-candle resumption and an
    inverse 1H VIX confirmation. A no-chase gate rejects extended late entries.
    """
    out={"stage":"OFF","direction":direction or "none","detail":"אין כיוון VIX מאושר",
         "entry":np.nan,"stop":np.nan,"tp1":np.nan,"tp2":np.nan,"atr":np.nan,
         "risk_pct":np.nan,"tp1_pct":np.nan,"tp2_pct":np.nan,"rr1":np.nan,"rr2":np.nan,
         "pullback_atr":np.nan,"extended":False,"vix_confirm":False,"htf_confirm":False}
    if direction not in ('LONG','SHORT'):
        return out
    q1=clean_history(qqq1h); q4=clean_history(qqq4h); v1=clean_history(vix1h); v4=clean_history(vix4h)
    if q1 is None or q4 is None or v1 is None or v4 is None or len(q1)<25 or len(q4)<8 or len(v1)<20 or len(v4)<8:
        out.update(stage="UNAVAILABLE",detail="נתוני QQQ/VIX סגורים אינם מספיקים לשכבת Fast Continuation")
        return out

    # Require comparable source candle windows; mismatched feed phases cannot confirm each other.
    if q1.index[-1] != v1.index[-1] or q4.index[-1] != v4.index[-1]:
        out.update(stage="UNAVAILABLE",detail="נרות QQQ ו-VIX אינם מסונכרנים בזמן")
        return out
    atrs=atr_series(q1.iloc[:-1],14)
    if len(atrs)==0 or not np.isfinite(atrs.iloc[-1]) or atrs.iloc[-1]<=0:
        out.update(stage="UNAVAILABLE",detail="ATR של QQQ אינו זמין")
        return out
    atr=float(atrs.iloc[-1]); out['atr']=atr
    qc=q1['Close']; qh=q1['High']; ql=q1['Low']; q4c=q4['Close']; vc=v1['Close']; v4c=v4['Close']
    qr=rsi_series(qc,14); vr=rsi_series(vc,14)
    if pd.isna(qr.iloc[-1]) or pd.isna(vr.iloc[-1]):
        out.update(stage="UNAVAILABLE",detail="RSI 1H עדיין לא זמין")
        return out

    if direction=='LONG':
        htf=bool(q4c.iloc[-1]>q4c.iloc[-3] and v4c.iloc[-1]<v4c.iloc[-3])
        counter=bool((qc.diff().iloc[-5:-1] < 0).any())
        prior_close=qc.iloc[-6:-1]
        pre_high=float(prior_close.max()); pull=float(qc.iloc[-4:-1].min())
        pb_depth=(pre_high-pull)/atr
        resume=bool(qc.iloc[-1]>qc.iloc[-2] and qc.iloc[-1]>=qh.iloc[-2]*0.999 and qr.iloc[-1]>=50 and qr.iloc[-1]>qr.iloc[-2])
        vix_ok=bool(vc.iloc[-1]<vc.iloc[-2] and vr.iloc[-1]<=55 and vr.iloc[-1]<=vr.iloc[-2]+0.5)
        extension=abs(float(qc.iloc[-1]-qc.iloc[-2]))/atr
        entry=float(qc.iloc[-1]); stop=float(q1['Low'].iloc[-4:].min()-0.10*atr)
        risk=entry-stop
        if risk<0.28*atr:
            stop=entry-0.28*atr; risk=entry-stop
        tp1=entry+max(0.65*atr,1.05*risk); tp2=entry+max(0.95*atr,1.55*risk)
    else:
        htf=bool(q4c.iloc[-1]<q4c.iloc[-3] and v4c.iloc[-1]>v4c.iloc[-3])
        counter=bool((qc.diff().iloc[-5:-1] > 0).any())
        prior_close=qc.iloc[-6:-1]
        pre_low=float(prior_close.min()); pull=float(qc.iloc[-4:-1].max())
        pb_depth=(pull-pre_low)/atr
        resume=bool(qc.iloc[-1]<qc.iloc[-2] and qc.iloc[-1]<=ql.iloc[-2]*1.001 and qr.iloc[-1]<=50 and qr.iloc[-1]<qr.iloc[-2])
        vix_ok=bool(vc.iloc[-1]>vc.iloc[-2] and vr.iloc[-1]>=45 and vr.iloc[-1]>=vr.iloc[-2]-0.5)
        extension=abs(float(qc.iloc[-1]-qc.iloc[-2]))/atr
        entry=float(qc.iloc[-1]); stop=float(q1['High'].iloc[-4:].max()+0.10*atr)
        risk=stop-entry
        if risk<0.28*atr:
            stop=entry+0.28*atr; risk=stop-entry
        tp1=entry-max(0.65*atr,1.05*risk); tp2=entry-max(0.95*atr,1.55*risk)

    out['htf_confirm']=htf; out['vix_confirm']=vix_ok; out['pullback_atr']=float(pb_depth)
    pullback_ok=counter and 0.22<=pb_depth<=1.60
    too_wide=(risk>0.90*atr)
    trigger_range=float(qh.iloc[-1]-ql.iloc[-1])/atr
    too_extended=(extension>0.80 or trigger_range>1.50)
    out['extended']=bool(too_wide or too_extended)

    if not htf:
        out.update(stage="WAIT",detail="הכיוון הראשי קיים, אבל QQQ 4H ו-VIX 4H עדיין לא מסונכרנים להמשך")
        return out
    if not pullback_ok:
        out.update(stage="WAIT",detail="אין כרגע Pullback 1H נקי מספיק; לא רודפים אחרי המחיר")
        return out
    if not resume:
        out.update(stage="WATCH",detail=f"Pullback 1H קיים ({pb_depth:.2f} ATR) · מחכים לנר חידוש בכיוון {direction}")
        return out
    if not vix_ok:
        out.update(stage="WATCH",detail="QQQ מנסה לחדש מומנטום, אבל VIX 1H עדיין לא מאשר את הכיוון ההפוך")
        return out
    if too_wide or too_extended:
        out.update(stage="NO_CHASE",detail="הטריגר הגיע אבל המחיר כבר נמתח/הסטופ רחב מדי — מחכים ל-Pullback נוסף")
        return out

    risk_pct=abs(entry-stop)/entry*100
    tp1_pct=abs(tp1-entry)/entry*100; tp2_pct=abs(tp2-entry)/entry*100
    rr1=abs(tp1-entry)/abs(entry-stop) if abs(entry-stop)>0 else np.nan
    rr2=abs(tp2-entry)/abs(entry-stop) if abs(entry-stop)>0 else np.nan
    out.update(stage="ENTRY_READY",detail="FAST CONTINUATION: Pullback + חידוש 1H + VIX inverse confirmation",
               entry=entry,stop=stop,tp1=tp1,tp2=tp2,risk_pct=risk_pct,tp1_pct=tp1_pct,tp2_pct=tp2_pct,rr1=rr1,rr2=rr2)
    return out

def stoch_rsi_state(close):
    """Diagnostic only: no additional score for a derivative of the same RSI."""
    r=rsi_series(close,14)
    lo=r.rolling(14).min(); hi=r.rolling(14).max()
    raw=100*(r-lo)/(hi-lo).replace(0,np.nan)
    k=raw.rolling(3).mean(); d=k.rolling(3).mean()
    if len(k)<2 or not np.isfinite(k.iloc[-2:]).all() or not np.isfinite(d.iloc[-2:]).all():
        return 'N/A'
    cross='cross up' if k.iloc[-2]<=d.iloc[-2] and k.iloc[-1]>d.iloc[-1] else ('cross down' if k.iloc[-2]>=d.iloc[-2] and k.iloc[-1]<d.iloc[-1] else 'no new cross')
    return f'K {k.iloc[-1]:.1f} / D {d.iloc[-1]:.1f} · {cross}'


def entry_gate(direction, div4, pattern4, fib_ok, trigger_ok, conflict):
    expected = 'bearish' if direction == 'LONG' else 'bullish'
    issues = []
    if direction not in ('LONG', 'SHORT'): issues.append('אין כיוון')
    if div4 != expected: issues.append('חסרה סטיית RSI ב-4H')
    if pattern4 != expected: issues.append('חסרה תבנית 4H תואמת')
    if not fib_ok: issues.append('חסרה תגובת Fib עם S/R')
    if not trigger_ok: issues.append('חסר אישור נר 1H')
    if conflict: issues.append('קונפליקט בין הליבה לרקע')
    return issues


def execution_session_open(now=None):
    now=utc_now(now)
    sc=schedule(now)
    return bool(((sc['open']<=now)&(now<sc['close'])).any())


def status_row(name,value,cls="white"):
    return f'<div class="status"><div class="status-name"><bdi>{name}</bdi></div><div class="status-val {cls}"><bdi>{value}</bdi></div></div>'

st.markdown('<div class="hero"><div class="hero-title">🎯 VIX Tactical</div><div class="muted">עסקאות קצרות · 1H טריגר · 4H אישור · 12H Reversal + Fast Continuation · v11.4</div></div>',unsafe_allow_html=True)
if st.button("🔄 רענן",use_container_width=True): st.cache_data.clear(); st.rerun()

st.markdown('<div class="panel" dir="rtl"><bdi dir="ltr">LONG</bdi> — חיפוש עלייה ב-QQQ/Nasdaq · <bdi dir="ltr">SHORT</bdi> — חיפוש ירידה ב-QQQ/Nasdaq<br><span class="muted">הכיוון הראשי עדיין נגזר מה-VIX בלבד. נתוני QQQ אינם מוסיפים ניקוד לכיוון; הם משמשים רק לתזמון FAST CONTINUATION / LATE ENTRY לאחר שכיוון כבר אושר.</span></div>', unsafe_allow_html=True)
with st.expander("איך לקרוא את האיתות והזמנים"):
    st.markdown('<div dir="rtl">המודל מיועד לסקאלפ/עסקאות קצרות: <bdi dir="ltr">4H</bdi> הוא גרף ה-Setup המרכזי, <bdi dir="ltr">1H</bdi> משמש לתזמון, ו-<bdi dir="ltr">12H</bdi> מפעיל גם מנוע Reversal Confluence. שילוב של S/R רב-נגיעות + Fib + יתד + דחייה יכול להעלות WATCH עוד לפני Entry. ב-v11.4 נוסף <bdi dir="ltr">FAST CONTINUATION / LATE ENTRY</bdi>: אחרי שהכיוון כבר אושר, QQQ משמש רק כשכבת ביצוע כדי לחפש Pullback קטן וחידוש מומנטום במקום לרדוף אחרי המחיר. החישוב משתמש בנרות סגורים בלבד. סיכומי 4H/12H נוצרים מנתוני שעה זמינים ולכן הם סיכומי סשן ולא נרות בורסה מקוריים.</div>',unsafe_allow_html=True)

with st.spinner("בודק נתוני VIX ומחשב… מקור שאינו מגיב עלול להאריך את הטעינה"):
    v=fetch_close("^VIX",period="6mo")
    vo_intra=fetch_ohlc("^VIX",period="60d",interval="60m")
    v_intra=vo_intra["Close"] if vo_intra is not None else None
    v9=fetch_close("^VIX9D",period="6mo")
    v3=fetch_close("^VIX3M",period="6mo")
    vv=fetch_close("^VVIX",period="6mo")
    v1=fetch_close("^VIX1D",period="6mo")
    skew=fetch_close("^SKEW",period="6mo")
    cor1m=fetch_close("^COR1M",period="6mo")
    v6=fetch_close("^VIX6M",period="6mo")
    v1y=fetch_close("^VIX1Y",period="6mo")
    dspx=fetch_close("DSPX",period="6mo")
    # v11.4: QQQ is optional and used ONLY for fast execution timing after VIX direction is established.
    qqq_intra=fetch_ohlc("QQQ",period="60d",interval="60m")

# Revalidate cached frames as the session changes.
v, v9, v3, vv, v1, skew, cor1m = [
    x if _valid_history(x,"6mo","1d") else None
    for x in (v,v9,v3,vv,v1,skew,cor1m)
]
v6, v1y, dspx = [x if _valid_history(x,"6mo","1d") else None for x in (v6,v1y,dspx)]
required_daily = {"VIX":v,"VIX9D":v9,"VIX3M":v3}
failed_daily=[name for name,x in required_daily.items() if x is None or len(x)<30]
if failed_daily:
    st.error("לא הצלחתי להשיג נתונים מלאים ועדכניים עבור: "+", ".join(failed_daily)+". ניסיתי Yahoo, Yahoo API ישיר, Cboe הרשמי ו-FRED במידת האפשר.")
    st.stop()

if vo_intra is None or v_intra is None or len(vo_intra)<40 or not fresh_history(v_intra,"60m"):
    st.error("אין כרגע מספיק נתוני VIX שעתיים מלאים ועדכניים. אין איתות עד שהמקור חוזר להיות מסונכרן.")
    st.stop()

V=float(v.iloc[-1]); V9=float(v9.iloc[-1]); V3=float(v3.iloc[-1]); VV=float(vv.iloc[-1]) if vv is not None and len(vv) else np.nan
V1=float(v1.iloc[-1]) if v1 is not None and len(v1) else np.nan
SKEW=float(skew.iloc[-1]) if skew is not None and len(skew) else np.nan
COR1M=float(cor1m.iloc[-1]) if cor1m is not None and len(cor1m) else np.nan
V6=float(v6.iloc[-1]) if v6 is not None and len(v6) else np.nan
V1Y=float(v1y.iloc[-1]) if v1y is not None and len(v1y) else np.nan
DSPX=float(dspx.iloc[-1]) if dspx is not None and len(dspx) else np.nan

VO1H=closed_session_hourly(vo_intra)
V1H=VO1H["Close"] if VO1H is not None else None
V4=resample_close(v_intra,4); V12=resample_close(v_intra,12)
VO4=resample_ohlc(vo_intra,4); VO12=resample_ohlc(vo_intra,12)
# Optional QQQ execution bars. Failure here must never disable the core VIX model.
if qqq_intra is not None and _valid_history(qqq_intra,"60d","60m"):
    QQQ1H=closed_session_hourly(qqq_intra)
    QQQ4=resample_ohlc(qqq_intra,4)
else:
    QQQ1H=None; QQQ4=None
if (V1H is None or len(V1H)<35 or VO1H is None or len(VO1H)<35 or V4 is None or len(V4)<30 or V12 is None or len(V12)<25 or VO4 is None or len(VO4)<20 or VO12 is None or len(VO12)<18
    or V4.index[-1] != latest_bucket_start(4,obj=v_intra)
    or V12.index[-1] != latest_bucket_start(12,obj=v_intra)):
    st.error("ממתינים לנרות VIX מלאים של 1H/4H/12H. אין איתות כאשר אחד הטיימפריימים חסר או ישן.")
    st.stop()

C1=pctn(v,1); C5=pctn(v,5); R9=V9/V; R3=V/V3
R1=(V1/V) if np.isfinite(V1) and V>0 else np.nan
R1_9=(V1/V9) if np.isfinite(V1) and V9>0 else np.nan
R3_6=(V3/V6) if np.isfinite(V6) and V6>0 else np.nan
R6_1Y=(V6/V1Y) if np.isfinite(V6) and np.isfinite(V1Y) and V1Y>0 else np.nan

# Regime engine: slow curve context is not a trigger, but it changes how much confidence
# we place in mean-reversion versus persistence at the front end.
def volatility_regime(vix, r9, r3, r36=np.nan):
    stress = 0
    if vix >= 25: stress += 2
    elif vix >= 20: stress += 1
    elif vix < 15: stress -= 1
    if r9 >= 1.03: stress += 2
    elif r9 <= 0.97: stress -= 1
    if r3 >= 1.02: stress += 2
    elif r3 <= 0.94: stress -= 1
    if np.isfinite(r36):
        if r36 >= 1.01: stress += 1
        elif r36 <= 0.96: stress -= 1
    if stress >= 4: return "STRESS", stress
    if stress <= -2: return "CALM", stress
    return "TRANSITION", stress

REGIME, REGIME_SCORE = volatility_regime(V,R9,R3,R3_6)

# Short-trade divergence hierarchy: 1H trigger + 4H main confirmation + 12H bonus.
DIV1=detect_divergence(V1H, lookback=72, pivot=3, min_sep=4, min_price_pct=0.55, min_rsi_delta=3.5, recent_bars=10)
DIV4=detect_divergence(V4, lookback=60, pivot=3, min_sep=4, min_price_pct=0.75, min_rsi_delta=4.0, recent_bars=8)
DIV12=detect_divergence(V12, lookback=50, pivot=2, min_sep=3, min_price_pct=1.0, min_rsi_delta=4.0, recent_bars=6)

# Repeated candle zones, Fib and clear patterns on the timeframes that matter for short trades.
SR4=repeated_sr_zone(VO4,lookback=50,pivot=2,tol_pct=1.15,min_touches=3)
SR12=repeated_sr_zone(VO12,lookback=45,pivot=2,tol_pct=1.25,min_touches=3)
FIB4=smart_fib_state(VO4,lookback=60,pivot=2,min_impulse_pct=5.0)
FIB12=smart_fib_state(VO12,lookback=55,pivot=2,min_impulse_pct=7.0)
PAT1=detect_clear_pattern(VO1H,lookback=48,pivot=2)
PAT4=detect_clear_pattern(VO4,lookback=42,pivot=2)
PAT12=detect_clear_pattern(VO12,lookback=46,pivot=2)
WEDGE12=detect_wedge_setup(VO12,lookback=55,pivot=2)
# Small 4H divergence is an enhancer only; strict DIV4 remains the main confirmation.
DIV4_EARLY=detect_divergence(V4, lookback=60, pivot=2, min_sep=3, min_price_pct=0.25, min_rsi_delta=1.5, recent_bars=10)
REV12=reversal_confluence_12h(VO12,SR12,FIB12,WEDGE12)


def directional_context_12h(df):
    """12H context only; never a mandatory trigger."""
    if df is None or len(df) < 10:
        return "neutral"
    c = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(c) < 10:
        return "neutral"
    r = rsi_series(c, 14).dropna()
    if len(r) < 4:
        return "neutral"
    price_move = float(c.iloc[-1] / c.iloc[-4] - 1.0)
    rsi_move = float(r.iloc[-1] - r.iloc[-4])
    if price_move >= 0.012 and rsi_move >= 2.0:
        return "bullish"
    if price_move <= -0.012 and rsi_move <= -2.0:
        return "bearish"
    return "neutral"

CTX12 = directional_context_12h(VO12)

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

# v11.1 Precision Continuation: emphasize *change/acceleration* in the front end.
VV1=pctn(vv,1) if vv is not None else np.nan
VV3=pctn(vv,3) if vv is not None else np.nan
VIX1=pctn(v,1) if v is not None else np.nan
VIX3=pctn(v,3) if v is not None else np.nan
VV_ACCEL=(VV1-VIX1) if np.isfinite(VV1) and np.isfinite(VIX1) else np.nan
VV_ACCEL3=(VV3-VIX3) if np.isfinite(VV3) and np.isfinite(VIX3) else np.nan

FRONT_RATIO=aligned_ratio(v1,v9) if v1 is not None and v9 is not None else None
FRONT_NOW=float(FRONT_RATIO.iloc[-1]) if FRONT_RATIO is not None and len(FRONT_RATIO) else np.nan
FRONT_PREV=float(FRONT_RATIO.iloc[-2]) if FRONT_RATIO is not None and len(FRONT_RATIO)>=2 else np.nan
FRONT_DELTA=((FRONT_NOW/FRONT_PREV)-1.0)*100 if np.isfinite(FRONT_NOW) and np.isfinite(FRONT_PREV) and FRONT_PREV!=0 else np.nan

# Realized VIX motion diagnostics from closed 1H bars: acceleration / shock persistence.
_v1h_ret = pd.to_numeric(V1H, errors="coerce").pct_change().dropna()*100 if V1H is not None else pd.Series(dtype=float)
VIX_RV_6H=float(_v1h_ret.iloc[-6:].std(ddof=0)) if len(_v1h_ret)>=6 else np.nan
VIX_RV_24H=float(_v1h_ret.iloc[-24:].std(ddof=0)) if len(_v1h_ret)>=24 else np.nan
VIX_RV_RATIO=(VIX_RV_6H/VIX_RV_24H) if np.isfinite(VIX_RV_6H) and np.isfinite(VIX_RV_24H) and VIX_RV_24H>0 else np.nan

score=0.0; reasons=[]
category_scores={"VIX Tactical":0.0,"12H Reversal":0.0,"Institutional":0.0,"Vol Curve":0.0,"Regime":0.0}

def addcat(cat,p,t):
    global score
    p=float(p); category_scores[cat]+=p; score+=p; reasons.append((p,t,cat))

# -----------------------------------------------------------------------------
# 1) VIX TACTICAL STRUCTURE — short-trade model
# 1H = fast trigger, 4H = main confirmation, 12H = bonus/context.
# -----------------------------------------------------------------------------
if DIV1=="bullish": addcat("VIX Tactical", 2.00,"VIX 1H Bullish RSI Divergence → VIX UP / QQQ SHORT")
elif DIV1=="bearish": addcat("VIX Tactical",-2.00,"VIX 1H Bearish RSI Divergence → VIX DOWN / QQQ LONG")

if DIV4=="bullish": addcat("VIX Tactical", 2.20,"VIX 4H Bullish RSI Divergence → אישור חזק SHORT")
elif DIV4=="bearish": addcat("VIX Tactical",-2.20,"VIX 4H Bearish RSI Divergence → אישור חזק LONG")

if DIV1==DIV4=="bullish": addcat("VIX Tactical", 0.80,"סנכרון Divergence 1H+4H → SHORT")
elif DIV1==DIV4=="bearish": addcat("VIX Tactical",-0.80,"סנכרון Divergence 1H+4H → LONG")

if DIV12=="bullish": addcat("VIX Tactical", 0.50,"בונוס: VIX 12H Bullish Divergence → SHORT")
elif DIV12=="bearish": addcat("VIX Tactical",-0.50,"בונוס: VIX 12H Bearish Divergence → LONG")

# 12H can support direction even without divergence; small bonus only.
if CTX12=="bullish": addcat("VIX Tactical", 0.35,"12H תומך בעליית VIX → בונוס SHORT (Divergence לא חובה)")
elif CTX12=="bearish": addcat("VIX Tactical",-0.35,"12H תומך בירידת VIX → בונוס LONG (Divergence לא חובה)")

# Clear patterns. Bullish pattern on VIX supports SHORT in QQQ; bearish pattern supports LONG.
def score_pattern(pat, tf, w):
    if pat.get("bias")=="bullish": addcat("VIX Tactical", w,f"VIX {tf} {pat['pattern']} שורי מאושר → SHORT")
    elif pat.get("bias")=="bearish": addcat("VIX Tactical",-w,f"VIX {tf} {pat['pattern']} דובי מאושר → LONG")
score_pattern(PAT1,"1H",0.55)
score_pattern(PAT4,"4H",0.90)
if PAT1.get("bias")!='none' and PAT1.get("bias")==PAT4.get("bias"):
    addcat("VIX Tactical",0.35 if PAT1['bias']=='bullish' else -0.35,"סנכרון תבנית 1H+4H")

# Repeated support/resistance zones from several candles.
if SR4["state"]=="support": addcat("VIX Tactical",0.45,f"VIX 4H באזור תמיכה חוזרת ({SR4['support_touches']} נגיעות) → SHORT")
elif SR4["state"]=="resistance": addcat("VIX Tactical",-0.45,f"VIX 4H באזור התנגדות חוזרת ({SR4['resistance_touches']} נגיעות) → LONG")
if SR12["state"]=="support": addcat("VIX Tactical",0.25,f"בונוס 12H: תמיכה חוזרת ({SR12['support_touches']} נגיעות) → SHORT")
elif SR12["state"]=="resistance": addcat("VIX Tactical",-0.25,f"בונוס 12H: התנגדות חוזרת ({SR12['resistance_touches']} נגיעות) → LONG")

# Smart Fib on 4H + 12H. Location alone does not score; reaction/break or repeated
# support/resistance + matching divergence/pattern is required.
def fib_confirmation_count(direction):
    if direction=='up':
        return sum([DIV1=='bullish',DIV4=='bullish',DIV12=='bullish',PAT1.get('bias')=='bullish',PAT4.get('bias')=='bullish'])
    return sum([DIV1=='bearish',DIV4=='bearish',DIV12=='bearish',PAT1.get('bias')=='bearish',PAT4.get('bias')=='bearish'])

def fib_repeated_level_overlap(f,z):
    """Fib zone must overlap a repeated candle S/R cluster (3+ touches)."""
    if not f.get('valid'): return False
    pad=max(0.05,0.01*f['current'])
    if f.get('direction')=='down':
        level=z.get('resistance',np.nan); touches=z.get('resistance_touches',0)
    else:
        level=z.get('support',np.nan); touches=z.get('support_touches',0)
    return bool(touches>=3 and np.isfinite(level) and f['zone_low']-pad<=level<=f['zone_high']+pad)

def score_fib(f,z,tf,base,react):
    # Fibonacci is a LOCATION + CONTINUATION tool:
    # impulse -> correction into 0.50-0.618 -> reaction back toward the OLD trend.
    # Crossing 0.618 never scores as a breakout signal.
    if not f.get('valid') or not fib_repeated_level_overlap(f,z): return
    fp=f['phase']; rise=fib_confirmation_count('up'); fall=fib_confirmation_count('down')
    if fp=='approaching_up' and fall>=1:
        addcat("VIX Tactical",-base,f"Smart Fib {tf}: תיקון עולה אל 0.50–0.618 בתוך Impulse יורד → Watch LONG אם המגמה הישנה חוזרת")
    elif fp=='approaching_down' and rise>=1:
        addcat("VIX Tactical", base,f"Smart Fib {tf}: תיקון יורד אל 0.50–0.618 בתוך Impulse עולה → Watch SHORT אם המגמה הישנה חוזרת")
    elif fp=='reject_down' and fall>=1:
        addcat("VIX Tactical",-react,f"Smart Fib {tf}: נגיעה/תיקון + דחייה מטה → חזרה למגמת VIX היורדת / LONG")
    elif fp=='rebound_up' and rise>=1:
        addcat("VIX Tactical", react,f"Smart Fib {tf}: נגיעה/תיקון + חזרה מעלה → חזרה למגמת VIX העולה / SHORT")
    # deep_retracement_* is context only: no points until continuation is confirmed.

score_fib(FIB4,SR4,'4H',0.45,0.75)
score_fib(FIB12,SR12,'12H',0.15,0.25)


# v11.2 12H Reversal Confluence Engine — early warning before the classic 4H entry.
# The 12H structure can now create WATCH/DEVELOPING status, but cannot become
# ENTRY READY without a closed-candle 4H continuation confirmation.
if REV12.get('stage')=='WATCH':
    rev_sign=-1.0 if REV12.get('direction')=='LONG' else 1.0
    addcat("12H Reversal",1.10*rev_sign,
           f"👀 12H REVERSAL WATCH: {REV12.get('detail','')} → QQQ/Nasdaq {REV12.get('direction')}")
    if REV12.get('rejection'):
        addcat("12H Reversal",0.45*rev_sign,"12H rejection מאזור S/R+Fib → ה-Reversal מתחיל לקבל אישור")
    # A small 4H divergence is useful here only because higher-timeframe confluence already exists.
    early_match = (REV12.get('direction')=='LONG' and DIV4_EARLY=='bearish') or (REV12.get('direction')=='SHORT' and DIV4_EARLY=='bullish')
    strict_match = (REV12.get('direction')=='LONG' and DIV4=='bearish') or (REV12.get('direction')=='SHORT' and DIV4=='bullish')
    wedge_break = bool(WEDGE12.get('confirmed'))
    if early_match and not strict_match:
        addcat("12H Reversal",0.35*rev_sign,"4H Early RSI Divergence קטן מאשר את ה-12H Reversal")
    if REV12.get('rejection') or strict_match or early_match or wedge_break:
        REV12['stage']='DEVELOPING'
        addcat("12H Reversal",0.55*rev_sign,"⚡ DEVELOPING: 12H confluence + rejection/break/divergence ב-4H")
    # ENTRY READY is deliberately stricter: closed 4H follow-through plus a concrete 4H structure reaction.
    fourh_structure = ((REV12.get('direction')=='LONG' and (PAT4.get('bias')=='bearish' or FIB4.get('phase')=='reject_down')) or
                       (REV12.get('direction')=='SHORT' and (PAT4.get('bias')=='bullish' or FIB4.get('phase')=='rebound_up')))
    if REV12.get('stage')=='DEVELOPING' and fourh_structure and four_hour_followthrough(VO4,REV12.get('direction')):
        REV12['stage']='ENTRY_READY'
        addcat("12H Reversal",0.55*rev_sign,"✅ ENTRY READY: סגירת 4H + המשכיות מאשרות את ה-Reversal")

# Reference Setup inherited from v10.9: complete 4H scalp setup can stand on its own.
def fib_is_confirmed_for(direction):
    if not FIB4.get("valid") or not fib_repeated_level_overlap(FIB4, SR4):
        return False
    phase = FIB4.get("phase")
    if direction == "down":
        return phase == "reject_down"
    return phase == "rebound_up"

ref_long = DIV4=="bearish" and PAT4.get("bias")=="bearish" and fib_is_confirmed_for("down")
ref_short = DIV4=="bullish" and PAT4.get("bias")=="bullish" and fib_is_confirmed_for("up")

if ref_long:
    addcat("VIX Tactical",-1.20,"🔥 4H REFERENCE SETUP: Bearish Divergence + יתד/M + Fib 0.50–0.618 ב-S/R + חזרה למגמת VIX היורדת → LONG חזק")
elif ref_short:
    addcat("VIX Tactical", 1.20,"🔥 4H REFERENCE SETUP: Bullish Divergence + יתד/W + Fib 0.50–0.618 ב-S/R + חזרה למגמת VIX העולה → SHORT חזק")


# -----------------------------------------------------------------------------
# 2) INSTITUTIONAL FAST PRESSURE — 25%
# Scalp-oriented confirmation layer, not an entry generator.
# Weight mix inside this layer:
# VIX1D immediate pressure 35% | VVIX acceleration 30% |
# front-end curve acceleration 25% | SKEW + COR1M context 10%.
# Missing public data reduces coverage; nothing is guessed.
# -----------------------------------------------------------------------------
institutional_available=0.0
institutional_total=2.5

# A) Immediate VIX1D pressure — max ±0.875 (35%)
if v1 is not None and len(v1)>=20 and np.isfinite(R1) and np.isfinite(R1_9):
    institutional_available += 0.875
    if R1_9>=1.05 and R9>=1.02:
        addcat("Institutional", 0.875,"⚡ VIX1D > VIX9D > VIX → לחץ בסגירה היומית / SHORT")
    elif R1>=1.08:
        addcat("Institutional", 0.60,"⚡ VIX1D בפרמיה ל-VIX → Immediate Event Pressure / SHORT")
    elif R1_9<=0.90 and R9<=0.98:
        addcat("Institutional",-0.875,"⚡ VIX1D < VIX9D < VIX → רגיעה בסגירה היומית / LONG")
    elif R1<=0.88:
        addcat("Institutional",-0.55,"⚡ VIX1D חלש מול VIX → Immediate Vol Relief / LONG")

# B) VVIX acceleration vs VIX — max ±0.75 (30%)
if vv is not None and len(vv)>=30 and np.isfinite(VV_Z) and np.isfinite(VV_ACCEL):
    institutional_available += 0.75
    accel = VV_ACCEL
    accel3 = VV_ACCEL3 if np.isfinite(VV_ACCEL3) else accel
    if accel>=4.0 and accel3>=3.0:
        addcat("Institutional", 0.75,"⚡ VVIX מאיץ מעל VIX → Fast Hidden Vol Pressure / SHORT")
    elif accel>=2.0:
        addcat("Institutional", 0.45,"VVIX מתחזק מהר יותר מ-VIX → SHORT confirmation")
    elif accel<=-4.0 and accel3<=-3.0:
        addcat("Institutional",-0.75,"⚡ VVIX נחלש מהר מול VIX → Fast Vol Relief / LONG")
    elif accel<=-2.0:
        addcat("Institutional",-0.45,"VVIX מאבד כוח מול VIX → LONG confirmation")

# C) Front-end VIX1D/VIX9D acceleration — max ±0.625 (25%)
if FRONT_RATIO is not None and len(FRONT_RATIO)>=2 and np.isfinite(FRONT_NOW) and np.isfinite(FRONT_DELTA):
    institutional_available += 0.625
    if FRONT_NOW>=1.03 and FRONT_DELTA>=2.0:
        addcat("Institutional", 0.625,"⚡ Front-End curve מתהדק מהר → SHORT confirmation")
    elif FRONT_NOW>=1.00 and FRONT_DELTA>=1.0:
        addcat("Institutional", 0.35,"Front-End pressure עולה → SHORT")
    elif FRONT_NOW<=0.95 and FRONT_DELTA<=-2.0:
        addcat("Institutional",-0.625,"⚡ Front-End pressure נשבר מטה → LONG confirmation")
    elif FRONT_NOW<=0.98 and FRONT_DELTA<=-1.0:
        addcat("Institutional",-0.35,"Front-End pressure נחלש → LONG")

# D) Slow context only — SKEW + COR1M together max ±0.25 (10%)
if skew is not None and len(skew)>=30 and np.isfinite(SKEW_Z) and np.isfinite(SKEW5):
    institutional_available += 0.125
    if SKEW_Z>=0.75 and SKEW5>=2.0:
        addcat("Institutional", 0.125,"SKEW Tail Risk עולה → Context SHORT")
    elif SKEW_Z<=-0.75 and SKEW5<=-2.0:
        addcat("Institutional",-0.125,"SKEW Tail Risk נחלש → Context LONG")

if cor1m is not None and len(cor1m)>=30 and np.isfinite(COR_Z) and np.isfinite(COR5):
    institutional_available += 0.125
    if COR_Z>=0.50 and COR5>=3.0:
        addcat("Institutional", 0.125,"COR1M עולה → Risk-Off context")
    elif COR_Z<=-0.50 and COR5<=-3.0:
        addcat("Institutional",-0.125,"COR1M יורד → Risk-On context")

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
# 4) VOLATILITY INTELLIGENCE / REGIME — confirmation, never a standalone entry.
# Front-end slope gets priority; longer curve and intraday VIX realized motion prevent
# a reversal setup from fighting an accelerating stress regime blindly.
# -----------------------------------------------------------------------------
if REGIME=="STRESS": addcat("Regime",0.35,"Stress regime: עקומת התנודתיות תומכת בהתמדה של VIX / SHORT context")
elif REGIME=="CALM": addcat("Regime",-0.25,"Calm regime: מבנה התנודתיות תומך יותר ב-Vol Relief / LONG context")

if np.isfinite(R3_6):
    if R3_6>=1.01: addcat("Regime",0.20,"VIX3M/VIX6M inverted → stress extends beyond front end")
    elif R3_6<=0.96: addcat("Regime",-0.15,"VIX3M/VIX6M steep contango → calmer medium-term curve")

if np.isfinite(VIX_RV_RATIO):
    if VIX_RV_RATIO>=1.35 and pctn(V1H,6)>0: addcat("Regime",0.30,"VIX intraday realized-vol acceleration upward → SHORT confirmation")
    elif VIX_RV_RATIO>=1.35 and pctn(V1H,6)<0: addcat("Regime",-0.30,"VIX intraday realized-vol acceleration downward → LONG confirmation")

# Optional dispersion context. It is deliberately tiny: dispersion is useful for
# separating index-wide fear from stock-specific volatility, not for timing by itself.
DSPX5=pctn(dspx,5) if dspx is not None else np.nan
if False:  # DSPX remains informational until validated out of sample.
    spread_move=C5-DSPX5
    if spread_move>=5.0: addcat("Regime",0.15,"Index vol outruns dispersion → more systemic stress / SHORT context")
    elif spread_move<=-5.0: addcat("Regime",-0.15,"Dispersion outruns index vol → less systemic VIX pressure / LONG context")

# Public-source data coverage. Core tactical model = 75%; institutional layer = 25%.
data_coverage = 75.0 + 25.0 * (institutional_available / institutional_total)

signed_signal=max(-10.0,min(10.0,float(score)))
signal_score=abs(signed_signal)
ENTRY_THRESHOLD=4.0
STRONG_THRESHOLD=6.5
VERY_STRONG_THRESHOLD=8.0

# Conflict gate: a high raw score is not enough if the tactical core and fast institutional
# pressure point in opposite directions. This reduces false confidence in scalp entries.
tactical_score=category_scores["VIX Tactical"] + category_scores["12H Reversal"]
fast_score=category_scores["Institutional"]
core_conflict = (tactical_score*fast_score < 0 and abs(tactical_score)>=2.0 and abs(fast_score)>=0.75)
if core_conflict:
    signed_signal *= 0.72
    signal_score=abs(signed_signal)
    reasons.append((0.0,"⚠️ Conflict Gate: Tactical vs Fast Pressure disagree → confidence reduced","Regime"))


# Human-readable confluence/conflict summary for fast scanning.
if core_conflict:
    conflict_text = "⚠️ MIXED — Tactical Core ו-Fast Pressure בכיוונים מנוגדים"
    conflict_cls = "yellow"
elif abs(tactical_score) < 2.0 and abs(fast_score) >= 0.45:
    conflict_text = "רקע מוסדי בלבד — אין עדיין 4H Setup"
    conflict_cls = "yellow"
else:
    conflict_text = "מסונכרן / ללא קונפליקט מהותי"
    conflict_cls = "green" if abs(tactical_score)>=2.0 else "white"
if signed_signal >= VERY_STRONG_THRESHOLD: state,icon,cls="VERY STRONG SHORT","🔴","red"
elif signed_signal >= STRONG_THRESHOLD: state,icon,cls="STRONG SHORT","🔴","red"
elif signed_signal >= ENTRY_THRESHOLD: state,icon,cls="SHORT","🟠","orange"
elif signed_signal <= -VERY_STRONG_THRESHOLD: state,icon,cls="VERY STRONG LONG","🟢","green"
elif signed_signal <= -STRONG_THRESHOLD: state,icon,cls="STRONG LONG","🟢","green"
elif signed_signal <= -ENTRY_THRESHOLD: state,icon,cls="LONG","🔵","blue"
else: state,icon,cls="WAIT","⚪","white"

# Strong normally requires matching divergence on 1H/4H. v11.2 exception: a DEVELOPING/ENTRY_READY
# 12H Reversal Confluence with wedge + S/R + Fib can substitute because it already contains
# independent higher-timeframe structure; WATCH alone is never enough for this exception.
if signal_score>=STRONG_THRESHOLD:
    matching_div='bullish' if signed_signal>0 else 'bearish'
    rev_override = REV12.get('stage') in ('DEVELOPING','ENTRY_READY') and ((signed_signal<0 and REV12.get('direction')=='LONG') or (signed_signal>0 and REV12.get('direction')=='SHORT'))
    if DIV1!=matching_div and DIV4!=matching_div and not rev_override:
        state,icon,cls="WAIT · חסר Divergence 1H/4H","⚪","white"

# Stage-aware early warning. This is deliberately separate from the raw score so a high-quality
# 12H setup is visible before it reaches the normal 4.0/10 entry threshold. Major conflict blocks
# an ENTRY READY label but not the WATCH information.
rev_dir=REV12.get('direction')
rev_stage=REV12.get('stage')
rev_sign=-1 if rev_dir=='LONG' else (1 if rev_dir=='SHORT' else 0)
rev_agrees=(rev_sign!=0 and signed_signal*rev_sign>0)
if rev_stage=='ENTRY_READY' and rev_agrees and not core_conflict and signal_score>=3.5:
    state,icon,cls=f"ENTRY READY · {rev_dir}","✅","green" if rev_dir=='LONG' else "red"
elif rev_stage=='DEVELOPING' and rev_agrees and not core_conflict:
    state,icon,cls=f"DEVELOPING · {rev_dir}","⚡","green" if rev_dir=='LONG' else "orange"
elif rev_stage=='WATCH' and not core_conflict and (rev_agrees or signal_score<ENTRY_THRESHOLD):
    state,icon,cls=f"WATCH · {rev_dir}","👀","blue" if rev_dir=='LONG' else "orange"
elif rev_stage in ('WATCH','DEVELOPING','ENTRY_READY') and core_conflict:
    state,icon,cls=f"WATCH · {rev_dir} · MIXED","⚠️","yellow"

# v11.4: no secondary path may bypass the same mandatory core conditions.
candidate_direction='LONG' if signed_signal<0 else ('SHORT' if signed_signal>0 else 'none')
trigger_ok=four_hour_followthrough(VO1H,candidate_direction)
gate_issues=entry_gate(candidate_direction,DIV4,PAT4.get('bias'),
                       fib_is_confirmed_for('down' if candidate_direction=='LONG' else 'up'),
                       trigger_ok,core_conflict)
core_approved=not gate_issues and signal_score>=ENTRY_THRESHOLD
market_open=execution_session_open()
if not core_approved and not state.startswith(('WATCH','DEVELOPING')):
    state,icon,cls='WAIT · חסר אישור ליבה','⚪','white'
if not market_open:
    state,icon,cls='WATCH · מחוץ לשעות המסחר','👀','yellow'
fast_direction=candidate_direction if core_approved and market_open else 'none'
FAST_CONT=fast_continuation_setup(QQQ1H,QQQ4,VO1H,VO4,fast_direction)
st.caption('אישור ליבה: '+('תקין' if core_approved else ' · '.join(gate_issues) or 'הציון נמוך מהסף'))
st.info('זמנים: 4H = מקטעי סשן (4 שעות ואז יתרת היום); 12H = סיכום יום מסחר רגיל בלבד, לא נר 12 שעות. נתוני VVIX/VIX1D והעקומה הם סגירות יומיות, לא לחץ חי. אין כאן חוזי VX1–VX3.')

pos = max(2, min(98, (signed_signal+10)/20*100))
if state.startswith("WAIT"):
    lean = "SHORT" if signed_signal > 0 else ("LONG" if signed_signal < 0 else "NEUTRAL")
    score_line = f"עוצמת איתות <b>{signal_score:.1f}/10</b> · נטייה {lean}"
else:
    score_line = f"עוצמת איתות <b>{signal_score:.1f}/10</b>"

st.markdown(f"""
<div class="signal">
  <div class="signal-title {cls}">{icon} {state}</div>
  <div class="score">{score_line}</div>
  <div class="confidence">סף חיפוש עסקה קצרה: <b>4.0/10</b> · כיסוי משוקלל של רכיבי הליבה/הרקע <b>{data_coverage:.0f}%</b></div>
  <div class="gauge-wrap">
    <div class="pointer" style="left:{pos:.1f}%"></div>
    <div class="gauge"><div class="midline"></div></div>
    <div class="gauge-labels"><span>LONG 10</span><span>WAIT 0</span><span>SHORT 10</span></div>
    <div class="gauge-zones"><span>LONG ≤ -4</span><span>WAIT</span><span>SHORT ≥ +4</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Dedicated second-chance execution card. It never overrides the primary VIX signal.
fc_stage=FAST_CONT.get('stage','OFF'); fc_dir=FAST_CONT.get('direction','none')
if fc_stage=='ENTRY_READY':
    st.markdown(f"""<div class="panel" dir="rtl" style="border:1px solid #2d8f68"><b>⚡ FAST CONTINUATION / LATE ENTRY · {fc_dir}</b><br>
    QQQ Entry (סגירת 1H): <bdi dir="ltr">{FAST_CONT['entry']:.2f}</bdi> · Stop: <bdi dir="ltr">{FAST_CONT['stop']:.2f}</bdi> ({FAST_CONT['risk_pct']:.2f}%)<br>
    TP1: <bdi dir="ltr">{FAST_CONT['tp1']:.2f}</bdi> ({FAST_CONT['tp1_pct']:.2f}%, R:R {FAST_CONT['rr1']:.2f}) · TP2: <bdi dir="ltr">{FAST_CONT['tp2']:.2f}</bdi> ({FAST_CONT['tp2_pct']:.2f}%, R:R {FAST_CONT['rr2']:.2f})<br>
    <span class="muted">עסקת המשך מהירה בלבד — כמה שעות ועד יום מסחר אחד. הכיוון מגיע מה-VIX; QQQ משמש לתזמון. אם אין Fill קרוב לטריגר או שהמחיר נמתח, לא רודפים.</span></div>""",unsafe_allow_html=True)
elif fc_stage in ('WATCH','NO_CHASE','WAIT','UNAVAILABLE'):
    badge={'WATCH':'👀','NO_CHASE':'⛔','WAIT':'⏳','UNAVAILABLE':'⚪'}.get(fc_stage,'⚪')
    st.markdown(f'<div class="panel" dir="rtl"><b>{badge} FAST CONTINUATION · {fc_stage}</b><br>{FAST_CONT.get("detail","")}<br><span class="muted">מצב משני בלבד; אינו משנה את האיתות הראשי.</span></div>',unsafe_allow_html=True)


# Freshness diagnostic: highlights unusually old derived bars.
def freshness_label(idx, hours):
    try:
        ts=pd.Timestamp(idx[-1])
        now=pd.Timestamp.now(tz=ts.tz) if getattr(ts, "tzinfo", None) is not None else pd.Timestamp.now()
        age=(now-ts).total_seconds()/3600.0
        # generous tolerance for market closures/weekends; informational only.
        stale = age > hours
        return age, stale
    except Exception:
        return np.nan, False

age1,stale1=freshness_label(V1H.index,8)
age4,stale4=freshness_label(V4.index,16)
age12,stale12=freshness_label(V12.index,28)
st.caption(f"Last data update · VIX Daily: {v.index[-1]:%d/%m/%Y} · 1H: {V1H.index[-1]:%d/%m %H:%M} · 4H: {V4.index[-1]:%d/%m %H:%M} · 12H: {V12.index[-1]:%d/%m %H:%M}")

if stale1 or stale4 or stale12:
    stale_parts=[]
    if stale1: stale_parts.append(f"1H ~{age1:.0f}h")
    if stale4: stale_parts.append(f"4H ~{age4:.0f}h")
    if stale12: stale_parts.append(f"12H ~{age12:.0f}h")
    st.warning("⚠️ Freshness: ייתכן שחלק מהברים ישנים: " + " · ".join(stale_parts) + ". בזמן סגירת שוק זה יכול להיות תקין; לפני עסקה יש לרענן ולאמת.")
st.caption("Refresh מושך את הנתון האחרון שהמקורות מספקים; הניתוח הטכני משתמש בנרות סגורים בלבד ואינו Tick-by-Tick. הציון הוא סכום משקלים, לא אחוז הצלחה. הוא מכוון לעסקאות קצרות יותר ולא לבניית Swing ארוך. אין ביצוע עסקאות אוטומטי.")
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
def div_text(d):
    if d=="bullish": return "Bullish → SHORT","red"
    if d=="bearish": return "Bearish → LONG","green"
    return "אין","white"

def fib_text(f):
    if not f.get("valid"): return "אין Impulse ברור","white"
    p=f["phase"]
    mapping={
        "approaching_up":("תיקון עולה לכיוון 0.50–0.618 · המתן לחזרה למגמה","yellow"),
        "approaching_down":("תיקון יורד לכיוון 0.50–0.618 · המתן לחזרה למגמה","yellow"),
        "decision_zone":("נגיעה באזור 0.50–0.618 · מחכים לאישור המשך","yellow"),
        "reject_down":("התיקון הסתיים? VIX חוזר למגמה היורדת → LONG","green"),
        "rebound_up":("התיקון הסתיים? VIX חוזר למגמה העולה → SHORT","red"),
        "deep_retracement_up":("תיקון עמוק מעבר 0.618 · אין אות, מחכים","yellow"),
        "deep_retracement_down":("תיקון עמוק מעבר 0.618 · אין אות, מחכים","yellow"),
        "below_zone":("לפני/מתחת לאזור התיקון","white"),
        "above_zone":("לפני/מעל לאזור התיקון","white"),
        "tracking":("מעקב אחרי התיקון","white")}
    return mapping.get(p,(p,"white"))

def pat_text(p):
    if p.get('bias')=='bullish': return f"{p['pattern']} → SHORT","red"
    if p.get('bias')=='bearish': return f"{p['pattern']} → LONG","green"
    return "אין תבנית מאושרת","white"


def wedge_text(w):
    if w.get('bias')=='bearish': return ("Rising Wedge · " + ("BROKE DOWN" if w.get('confirmed') else "SETUP") + " → LONG","green")
    if w.get('bias')=='bullish': return ("Falling Wedge · " + ("BROKE UP" if w.get('confirmed') else "SETUP") + " → SHORT","red")
    return "אין יתד 12H פעילה","white"

def reversal_text(r):
    d=r.get('direction','none'); stg=r.get('stage','NONE')
    if d=='LONG': return f"{stg} · QQQ/Nasdaq LONG","green" if stg!='WATCH' else "blue"
    if d=='SHORT': return f"{stg} · QQQ/Nasdaq SHORT","red" if stg!='WATCH' else "orange"
    return "אין Confluence מלא","white"

def fast_text(f):
    stg=f.get('stage','OFF'); d=f.get('direction','none')
    if stg=='ENTRY_READY': return f"ENTRY READY · {d}","green" if d=='LONG' else "red"
    if stg=='WATCH': return f"WATCH · {d}","blue"
    if stg=='NO_CHASE': return "NO CHASE · חכה ל-Pullback נוסף","yellow"
    if stg=='UNAVAILABLE': return "לא זמין","white"
    return "WAIT · אין Late Entry נקי","white"

def srzone_text(z):
    if z['state']=='support': return f"תמיכה · {z['support_touches']} נגיעות → SHORT","red"
    if z['state']=='resistance': return f"התנגדות · {z['resistance_touches']} נגיעות → LONG","green"
    return "אין אזור רב-נגיעות פעיל","white"

def institutional_text(v):
    if v>=0.45: return f"Daily Pressure {v:+.2f}/2.5 → SHORT","red"
    if v<=-0.45: return f"Daily Relief {v:+.2f}/2.5 → LONG","green"
    return f"Daily Neutral {v:+.2f}/2.5","white"

d1,d1c=div_text(DIV1); d4,d4c=div_text(DIV4); d12,d12c=div_text(DIV12)
f4t,f4c=fib_text(FIB4); f12t,f12c=fib_text(FIB12)
p1t,p1c=pat_text(PAT1); p4t,p4c=pat_text(PAT4); p12t,p12c=pat_text(PAT12)
w12t,w12c=wedge_text(WEDGE12); revt,revc=reversal_text(REV12)
fastt,fastc=fast_text(FAST_CONT)
sr4t,sr4c=srzone_text(SR4); sr12t,sr12c=srzone_text(SR12)
instt,instc=institutional_text(category_scores["Institutional"])
term_text="Risk-Off" if R3>=1.02 else ("Risk-On" if R3<=0.94 else "ניטרלי")
term_cls="red" if R3>=1.02 else ("green" if R3<=0.94 else "white")
regime_cls="red" if REGIME=="STRESS" else ("green" if REGIME=="CALM" else "yellow")
rv_text=(f"Acceleration {VIX_RV_RATIO:.2f}x" if np.isfinite(VIX_RV_RATIO) else "N/A")

st.markdown('<div class="status-grid">'+
    status_row("Divergence 1H · FAST TRIGGER",d1,d1c)+
    status_row("Divergence 4H · MAIN SETUP",d4,d4c)+
    status_row("Divergence 4H · EARLY",(DIV4_EARLY if DIV4=='none' else "strict already active"),"green" if DIV4_EARLY=='bearish' else ("red" if DIV4_EARLY=='bullish' else "white"))+
    status_row("Divergence 12H · BONUS",d12,d12c)+
    status_row("Pattern 1H · W/M/Wedge",p1t,p1c)+
    status_row("Pattern 4H · W/M/Wedge",p4t,p4c)+
    status_row("Pattern 12H · CONFIRMED",p12t,p12c)+
    status_row("Wedge 12H · EARLY SETUP",w12t,w12c)+
    status_row("12H Reversal Confluence",revt,revc)+
    status_row("FAST CONTINUATION / LATE ENTRY",fastt,fastc)+
    status_row("Repeated S/R · 4H",sr4t,sr4c)+
    status_row("Smart Fib · 4H",f4t,f4c)+
    status_row("Repeated S/R · 12H",sr12t,sr12c)+
    status_row("Smart Fib · 12H",f12t,f12c)+
    status_row("Daily Volatility Context",instt,instc)+
    status_row("Stoch RSI · 4H (מידע בלבד)",stoch_rsi_state(V4))+
    status_row("Stoch RSI · session context (מידע בלבד)",stoch_rsi_state(V12))+
    status_row("Setup Alignment",conflict_text,conflict_cls)+
    status_row("Volatility Regime",REGIME,regime_cls)+
    status_row("VIX Intraday RV",rv_text,"orange" if np.isfinite(VIX_RV_RATIO) and VIX_RV_RATIO>=1.35 else "white")+
    status_row("Term Structure",term_text,term_cls)+
    status_row("VIX9D / VIX",("לחץ" if R9>=1.03 else ("רגוע" if R9<=0.97 else "מאוזן")),"red" if R9>=1.03 else ("green" if R9<=0.97 else "white"))+
    '</div>',unsafe_allow_html=True)

# Smart Fib detail panels
for tf,f,z in [("4H",FIB4,SR4),("12H",FIB12,SR12)]:
    if f.get("valid"):
        arrow="↓" if f["direction"]=="down" else "↑"; ft,_=fib_text(f)
        zone_note=(f"תמיכה {z['support_touches']} נגיעות" if z['state']=='support' else (f"התנגדות {z['resistance_touches']} נגיעות" if z['state']=='resistance' else "ללא S/R רב-נגיעות פעיל"))
        st.markdown(f'<div class="panel"><b>Smart Fib {tf} {arrow}</b><br>Impulse: <bdi dir="ltr">{f["start"]:.2f} → {f["end"]:.2f}</bdi> · אזור <bdi dir="ltr">0.50–0.618 = {f["zone_low"]:.2f}–{f["zone_high"]:.2f}</bdi><br>VIX <bdi dir="ltr">{f["current"]:.2f}</bdi> · {ft} · {zone_note}<br><span class="muted">Fib הוא כלי תיקון/המשך: מחפשים Impulse, תיקון ל-0.50–0.618 ואז חזרה למגמה הקודמת. מעבר 0.618 אינו פריצה ואינו אות כניסה. נדרש אישור נוסף.</span></div>',unsafe_allow_html=True)

# Minimal execution reminder
if FAST_CONT.get('stage')=='ENTRY_READY':
    action=f"FAST CONTINUATION {FAST_CONT.get('direction')} — כניסה מאוחרת מבוקרת אחרי Pullback; יעד לעסקה של שעות ועד יום, בלי לרדוף"
elif state.startswith("ENTRY READY"):
    action=f"{state} — יש אישור 12H+4H; עדיין הגדר Entry/SL לפי המחיר בזמן אמת"
elif state.startswith("DEVELOPING"):
    action=f"{state} — הסטאפ מתחזק; המתן ל-4H continuation/retest לפני Entry"
elif state.startswith("WATCH"):
    action=f"{state} — התראה מוקדמת; עדיין לא Entry"
elif state.startswith("WAIT"): action="אין עסקה — חסר אישור ליבה"
elif "SHORT" in state: action="חפש טריגר SHORT קצר ב-QQQ/Nasdaq; זה אינו אישור כניסה אוטומטי"
elif "LONG" in state: action="חפש טריגר LONG קצר ב-QQQ/Nasdaq; זה אינו אישור כניסה אוטומטי"
else: action="אין עסקה — המתן לסנכרון"
st.markdown(f'<div class="panel" style="text-align:center;font-weight:900">{action}</div>',unsafe_allow_html=True)

with st.expander("פירוט החישוב"):
    st.write("**מבנה v11.4:** 4H נשאר Setup מרכזי ו-12H Reversal Confluence מייצר WATCH/DEVELOPING/ENTRY READY. שכבת FAST CONTINUATION נפתחת רק לאחר שכיוון כבר אושר; היא בודקת QQQ 4H/1H + VIX הפוך, Pullback מבוקר, חידוש מומנטום ו-No-Chase. QQQ אינו מוסיף ניקוד לכיוון הראשי ואין EMA.")
    st.write(f"Divergence: 1H={DIV1} · 4H={DIV4} · 12H={DIV12}")
    st.write(f"Pattern: 1H={PAT1['pattern']} ({PAT1['bias']}) · 4H={PAT4['pattern']} ({PAT4['bias']}) · 12H={PAT12['pattern']} ({PAT12['bias']})")
    st.write(f"12H Wedge Early: {WEDGE12['pattern']} ({WEDGE12['bias']}) · confirmed={WEDGE12['confirmed']} · Reversal={REV12['stage']} {REV12['direction']}")
    st.write(f"4H Early Divergence: {DIV4_EARLY}")
    st.write(f"Fast Continuation: {FAST_CONT.get('stage')} {FAST_CONT.get('direction')} · {FAST_CONT.get('detail')}")
    if FAST_CONT.get('stage')=='ENTRY_READY': st.write(f"QQQ fast plan: Entry {FAST_CONT['entry']:.2f} · SL {FAST_CONT['stop']:.2f} · TP1 {FAST_CONT['tp1']:.2f} · TP2 {FAST_CONT['tp2']:.2f}")
    st.write(f"Repeated S/R 4H: {SR4['state']} · support touches={SR4['support_touches']} · resistance touches={SR4['resistance_touches']}")
    st.write(f"Repeated S/R 12H: {SR12['state']} · support touches={SR12['support_touches']} · resistance touches={SR12['resistance_touches']}")
    if FIB4.get('valid'): st.write(f"Smart Fib 4H: {FIB4['direction']} · {FIB4['phase']} · zone {FIB4['zone_low']:.2f}–{FIB4['zone_high']:.2f}")
    if FIB12.get('valid'): st.write(f"Smart Fib 12H: {FIB12['direction']} · {FIB12['phase']} · zone {FIB12['zone_low']:.2f}–{FIB12['zone_high']:.2f}")
    st.write(f"VIX 1D {C1:+.2f}% · 5D {C5:+.2f}% · VIX9D/VIX {R9:.3f} · VIX/VIX3M {R3:.3f} · Regime {REGIME}")
    if np.isfinite(V6): st.write(f"VIX6M {V6:.2f} · VIX3M/VIX6M {R3_6:.3f}")
    if np.isfinite(V1Y): st.write(f"VIX1Y {V1Y:.2f} · VIX6M/VIX1Y {R6_1Y:.3f}")
    if np.isfinite(VIX_RV_RATIO): st.write(f"VIX intraday RV acceleration {VIX_RV_RATIO:.2f}x")
    if np.isfinite(DSPX): st.write(f"DSPX {DSPX:.2f} · 5D {DSPX5:+.2f}%")
    if np.isfinite(V1): st.write(f"VIX1D {V1:.2f} · VIX1D/VIX {R1:.3f} · VIX1D/VIX9D {R1_9:.3f}")
    if np.isfinite(VV): st.write(f"VVIX {VV:.2f} · VVIX/VIX z-score {VV_Z:+.2f} · Relative 5D momentum {VV_REL:+.2f}pp")
    if np.isfinite(SKEW): st.write(f"SKEW {SKEW:.2f} · 5D {SKEW5:+.2f}% · z-score {SKEW_Z:+.2f}")
    if np.isfinite(COR1M): st.write(f"COR1M {COR1M:.2f} · 5D {COR5:+.2f}% · z-score {COR_Z:+.2f}")
    st.write(f"כיסוי נתונים: {data_coverage:.0f}% · Institutional availability {institutional_available:.1f}/{institutional_total:.1f}")
    st.write(f"ציון נטו: {score:+.2f} → עוצמת איתות {signal_score:.2f}/10 · כיוון: {'SHORT' if signed_signal>0 else ('LONG' if signed_signal<0 else 'NEUTRAL')}")
    st.write("**ציוני קטגוריות:** "+" · ".join([f"{k}: {v:+.2f}" for k,v in category_scores.items()]))
    for pts,txt,cat in sorted(reasons,key=lambda z:abs(z[0]),reverse=True): st.write(f"**{pts:+.2f}** — {txt} · _{cat}_")

with st.expander("מקורות נתונים / גיבוי"):
    st.write(f"VIX Daily: **{source_name(v)}**")
    st.write(f"VIX9D Daily: **{source_name(v9)}**")
    st.write(f"VIX3M Daily: **{source_name(v3)}**")
    st.write(f"VVIX Daily: **{source_name(vv)}**")
    st.write(f"VIX1D Daily: **{source_name(v1)}**")
    st.write(f"SKEW Daily: **{source_name(skew)}**")
    st.write(f"COR1M Daily: **{source_name(cor1m)}**")
    st.write(f"VIX6M Daily: **{source_name(v6)}**")
    st.write(f"VIX1Y Daily: **{source_name(v1y)}**")
    st.write(f"DSPX Daily: **{source_name(dspx)}**")
    st.write(f"VIX Close 60m: **{source_name(v_intra)}**")
    st.write(f"VIX OHLC 60m: **{source_name(vo_intra)}**")
    st.write(f"QQQ OHLC 60m (Fast Execution only): **{source_name(qqq_intra)}**")
    st.caption("גיבוי אמיתי: Yahoo/yfinance → Yahoo Chart API ישיר → Cboe הרשמי למדדי תנודתיות/אופציות. FRED משמש ל-VIX/VIX3M במידת הצורך. רכיב Institutional הוא אופציונלי: מקור חסר מוריד Data Coverage ואינו מוחלף בנתון מומצא.")

st.caption(f"עודכן {pd.Timestamp.now(tz='Asia/Jerusalem').strftime('%H:%M')} · v11.4 Fast Continuation / Late Entry · שעון ישראל · Multi-Source + Retry פעיל · כלי מחקרי, לא ייעוץ השקעות")
