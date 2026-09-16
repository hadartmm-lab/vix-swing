# VIX Swing v10.5 — Final Calibrated

מבוסס על v10.4 Institutional Volatility Pressure, לאחר בדיקת robustness ובק־טסט אחרון על רכיבים שניתנים לאימות היסטורי בצורה נקייה.

## משקלי המודל הסופיים
- **VIX Divergence + Reversal Structure — 45%**
  - RSI Divergence 12H: עד ±1.90
  - RSI Divergence 4H: עד ±1.00
  - סנכרון 4H+12H: ±0.45
  - Higher High + rejection ב-VIX: עד -0.65 לכיוון LONG Nasdaq
  - Lower Low + rebound ב-VIX: עד +0.30 לכיוון SHORT Nasdaq
  - VIX Support/Resistance 12H: עד ±0.50
- **Institutional Volatility Pressure — 25%**
  - VVIX relative pressure: עד ±0.90
  - VIX1D short-end curve: עד ±0.80
  - SKEW tail-risk pressure: ±0.40
  - COR1M implied correlation: ±0.40
- **Volatility Curve / VIX Impulse — 15%**
  - VIX9D/VIX, VIX/VIX3M, תנועת VIX יומית/5D ורמת VIX.
- **Nasdaq/S&P Confirmation — 10%**
  - Momentum 2D/5D קיבל משקל קטן מאוד לאחר שלא הראה יתרון יציב בבק־טסט.
  - Support/Resistance 12H הוא עיקר הקטגוריה.
- **EMA9/26 — 5% בלבד**
  - נשאר מסנן מגמה קטן בלבד.

## כיול Swing סופי
Swing exhaustion דורש כעת לפחות **1.5%** יצירת שיא/שפל חדש וגם **1.5% rejection/rebound** לפני קבלת ניקוד. כך שיא עולה או שפל יורד לבדם אינם מתפרשים בטעות כהיפוך.

הבק־טסט האחרון הצביע על אסימטריה: Higher High + rejection ב-VIX היה שימושי יותר לזיהוי אפשרות לירידת VIX / LONG Nasdaq מאשר Lower Low + rebound לזיהוי SHORT. לכן המשקלים אינם סימטריים.

## Institutional Volatility Pressure
- **VVIX** — תנודתיות צפויה של VIX / לחץ convexity.
- **VIX1D** — לחץ מיידי בקצה הקצר מול VIX9D/VIX.
- **SKEW** — ביקוש ל-tail risk.
- **COR1M** — implied correlation / herd behavior.

הרכיבים משתמשים בשינויים יחסיים וב-z-score כדי להתאים לרג'ימים שונים.

## גיבויי נתונים
- Yahoo/yfinance עם retry.
- Yahoo Chart API ישיר.
- Cboe הרשמי למדדי VIX / VVIX / VIX1D / SKEW / COR1M כאשר זמין.
- FRED לגיבוי VIX/VIX3M.
- Stooq לגיבוי NDX/SPX Daily.
- אם רכיב Institutional חסר, האפליקציה לא ממציאה נתון אלא מורידה Data Coverage.

## ממשק
- Signal Score 0–10.
- Top Drivers — שלושת הגורמים החזקים ביותר כרגע.
- Data Coverage.
- Institutional Vol Pressure בשורה ברורה.
- Divergence 12H מסומן כ-High Weight, EMA9/26 כ-Low Weight.

הכלי מחקרי בלבד ואינו ייעוץ השקעות.
