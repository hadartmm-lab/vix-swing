# VIX Tactical v10.8 — Short Trade

גרסה זו נבנתה על v10.7 Smart Fib ומותאמת לעסקאות קצרות יותר, לאו דווקא Swing.

## מה הוסר
- אין יותר Market Confirmation של Nasdaq 100 / S&P 500 בציון.
- אין יותר EMA 9/26 או Golden/Death Cross בציון ובממשק.

## היררכיית הטיימפריימים
- 1H = טריגר מהיר.
- 4H = האישור המרכזי.
- 12H = בונוס / הקשר בלבד.

## משקלי Divergence
- RSI Divergence ב-VIX 1H: עד ±2.00.
- RSI Divergence ב-VIX 4H: עד ±2.20.
- סנכרון 1H + 4H: עד ±0.80.
- RSI Divergence ב-VIX 12H: בונוס עד ±0.50.
- STRONG / VERY STRONG דורש Divergence מתאים לפחות ב-1H או ב-4H.

Bullish divergence ב-VIX מחזק SHORT ב-QQQ/Nasdaq. Bearish divergence ב-VIX מחזק LONG.

## Smart Fib + תמיכה/התנגדות
Smart Fib נבדק גם ב-4H וגם ב-12H באזור 0.50–0.618.

Fib לא מקבל ניקוד רק בגלל מיקום המחיר. כדי לקבל ניקוד, אזור ה-Fib צריך לחפוף לאזור תמיכה/התנגדות שנבדק במספר נרות/פיבוטים — לפחות 3 נגיעות — ובנוסף נדרש אישור מכיוון Divergence או תבנית.

משקל:
- 4H: Watch עד ±0.45, תגובה/שבירה מאושרת עד ±0.75.
- 12H: Watch עד ±0.25, תגובה/שבירה מאושרת עד ±0.45.

## תבניות מחיר ב-VIX
נבדקות ב-1H וב-4H:
- W מאושר = שורי ל-VIX -> מחזק SHORT במדד.
- M מאושר = דובי ל-VIX -> מחזק LONG במדד.
- Bullish / Falling Wedge מאושר = שורי ל-VIX -> מחזק SHORT.
- Bearish / Rising Wedge מאושר = דובי ל-VIX -> מחזק LONG.

משקל:
- Pattern 1H: עד ±0.55.
- Pattern 4H: עד ±0.90.
- סנכרון Pattern 1H + 4H: עד ±0.35.

## אזורי S/R רב-נגיעות
- 4H: עד ±0.45.
- 12H: בונוס עד ±0.25.
- נדרש Cluster של לפחות 3 נגיעות כדי שהאזור ייחשב משמעותי.

## שכבות שנשארו
Institutional Volatility Pressure נשאר כשכבת הקשר: VVIX, VIX1D, SKEW, COR1M.
Volatility Curve נשארת: VIX9D/VIX, VIX/VIX3M ו-Impulse יומי/5D.

## ספי איתות
- LONG / SHORT: החל מ-4.0/10.
- STRONG: החל מ-6.5/10.
- VERY STRONG: החל מ-8.0/10.

הציון הוא סכום משקלים ולא אחוז הצלחה. האפליקציה משתמשת בנרות סגורים בלבד; נר 1H שעדיין נבנה לא נכנס לחישוב.

## התקנה
```bash
pip install -r requirements.txt
streamlit run app.py
```

זהו כלי מחקרי ואינו מבצע עסקאות אוטומטית.


## v10.9 Reference Setup upgrade
- 4H = primary scalp / short-trade Setup.
- 1H = timing / fast trigger. Missing 1H divergence does not invalidate a complete 4H setup.
- 12H = direction/context bonus. 12H divergence is never mandatory.
- 12H directional context bonus: ±0.35.
- 4H Reference Setup bonus: ±1.20 when these align:
  4H RSI divergence + confirmed W/M or wedge + Smart Fib 0.50–0.618 reaction/break at repeated S/R.
- Symmetric logic: bearish VIX setup -> LONG Nasdaq/QQQ bias; bullish VIX setup -> SHORT Nasdaq/QQQ bias.
- Closed candles only.

## v10.10 Institutional Fast Pressure
- Scalp-oriented institutional confirmation; it cannot create a trade by itself.
- VIX1D immediate pressure: 35% of institutional layer (max ±0.875).
- VVIX acceleration versus VIX: 30% (max ±0.75), emphasizing 1D/3D relative acceleration.
- Front-end VIX1D/VIX9D acceleration: 25% (max ±0.625).
- SKEW + COR1M: only 10% combined (max ±0.25), context rather than trigger.
- Refresh pulls the newest observation exposed by each public source.
- Technical 1H/4H/12H logic remains closed-candle only; not tick-by-tick.
- UI now explicitly shows Last data update.
