# VIX Swing v10.3 — Resilient + VIX S/R 12H

מבוסס על v10.2 Resilient FIXED.

## חדש ב-v10.3
- Support / Resistance נפרד ל-VIX על גרף 12H, מבוסס OHLC 60m שעובר resample ל-12H לפי סשן ניו-יורק.
- ניקוד VIX S/R:
  - תגובה/קרבה לתמיכה: +0.55 לכיוון SHORT ב-QQQ/Nasdaq.
  - דחייה מהתנגדות: -0.55 לכיוון LONG.
  - פריצה מאושרת מעל התנגדות: +0.70 SHORT.
  - שבירה מאושרת מתחת לתמיכה: -0.70 LONG.
- תצוגה נפרדת של Market S/R ושל VIX S/R.
- רמות התמיכה וההתנגדות המדויקות מופיעות בפירוט החישוב.

## נשמר מ-v10.2
- Yahoo/yfinance עם retry.
- Yahoo Chart API ישיר כגיבוי.
- Cboe/FRED ל-VIX family Daily לפי הצורך.
- Stooq ל-NDX/SPX Daily לפי הצורך.
- בניית 4H/12H מעוגנת לסשן ארה"ב.
- RSI Divergence ב-VIX 4H/12H.
- ניקוד Momentum, Term Structure, VIX9D/VIX, EMA9/26 ועוד.

הכלי מחקרי בלבד ואינו ייעוץ השקעות.
