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


def fresh_history(obj, interval, now=None):
    if obj is None or obj.empty:
        return False
    if interval == '1d':
        return obj.index[-1].tz_localize(None).normalize() == latest_daily_session(now)
    now = utc_now(now)
    sched = schedule(now)
    available = sched[sched['open'] + pd.Timedelta(hours=1, minutes=15) <= now]
    row = available.iloc[-1]
    # Timestamp of the most recent fully available hourly bar in regular hours.
    starts = pd.date_range(row['open'], row['close'], freq='1h', inclusive='left')
    ends = pd.DatetimeIndex([min(t + pd.Timedelta(hours=1), row['close']) for t in starts])
    expected = starts[ends + pd.Timedelta(minutes=15) <= now][-1]
    idx = obj.index
    idx = idx.tz_localize(TZ) if idx.tz is None else idx.tz_convert(TZ)
    return idx[-1].tz_convert('UTC') >= expected


def session_bars(obj, hours, now=None):
    """Regular-session 4h buckets or a whole session (legacy '12H').

    Require every expected hourly start. Reject unfinished/incomplete groups.
    A 4h second bucket is 2.5h on a normal day; shortened sessions use the
    exchange calendar. This is deliberately not a TradingView 12h candle.
    """
    x = clean_history(obj)
    if x is None or hours not in (4,12):
        return None
    now = utc_now(now)
    x.index = x.index.tz_localize(TZ) if x.index.tz is None else x.index.tz_convert(TZ)
    x.index = x.index.tz_convert('UTC')
    sched = schedule(now)
    sched = sched[(sched['close'] >= x.index.min()) & (sched['open'] <= now)]
    rows, labels = [], []
    for _, day in sched.iterrows():
        op, cl = day['open'], day['close']
        starts = pd.date_range(op, cl, freq='1h', inclusive='left')
        for group in range(0, len(starts), hours):
            expected = starts[group:group+hours]
            end = min(expected[-1] + pd.Timedelta(hours=1), cl)
            if end + pd.Timedelta(minutes=15) > now or not expected.isin(x.index).all():
                continue
            part = x.loc[expected]
            labels.append(expected[0])
            if isinstance(x, pd.DataFrame):
                rows.append({'Open':part.Open.iloc[0], 'High':part.High.max(),
                             'Low':part.Low.min(), 'Close':part.Close.iloc[-1]})
            else:
                rows.append(part.iloc[-1])
    if not rows:
        return None
    idx = pd.DatetimeIndex(labels).tz_convert(TZ)
    out = pd.DataFrame(rows,index=idx) if isinstance(x,pd.DataFrame) else pd.Series(rows,index=idx)
    out.attrs = obj.attrs.copy()
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


def latest_bucket_start(hours, now=None):
    now=utc_now(now)
    sched=schedule(now)
    sched=sched[(sched['open'] <= now) & (sched['close'] >= now-pd.Timedelta(days=10))]
    result=None
    for _,day in sched.iterrows():
        starts=pd.date_range(day['open'],day['close'],freq=f'{hours}h',inclusive='left')
        for start in starts:
            end=min(start+pd.Timedelta(hours=hours),day['close'])
            if end+pd.Timedelta(minutes=15)<=now:
                result=start.tz_convert(TZ)
    return result
