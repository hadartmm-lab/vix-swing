# VIX Tactical v11.2 — 12H Reversal Confluence

גרסה זו משדרגת את v11.1 Precision Continuation כדי לזהות מוקדם יותר Reversal ב-VIX שמאותת על כיוון הפוך ב-QQQ/Nasdaq.

## מה חדש ב-v11.2
- 12H כבר אינו רק בונוס: נוסף **12H Reversal Confluence Engine**.
- WATCH מוקדם כאשר מתקיימים יחד: אזור S/R עם לפחות 3 נגיעות + Smart Fib 0.50–0.618 + Rising/Falling Wedge פעילה.
- זיהוי יתד 12H גם כשהיא עדיין בבנייה, ולא רק לאחר breakout מלא.
- נשמר אירוע rejection מה-S/R/Fib גם לאחר שהמחיר כבר התרחק מהאזור, כדי לא לאבד את הסטאפ מאוחר מדי.
- נוסף **4H Early RSI Divergence** רגיש יותר. הוא משמש רק כחיזוק כאשר 12H Confluence כבר קיים, ולכן לא מייצר עסקה לבדו.
- סטטוסים חדשים: **WATCH → DEVELOPING → ENTRY READY**.
- DEVELOPING מתקבל כאשר WATCH מקבל rejection, שבירת wedge או Divergence מתאים ב-4H.
- ENTRY READY דורש בנוסף סגירת 4H והמשכיות/מבנה מאשר; WATCH לבדו לעולם אינו Entry.
- הלוגיקה סימטרית: VIX bearish reversal → QQQ/Nasdaq LONG; VIX bullish reversal → QQQ/Nasdaq SHORT.

## מה נשאר מ-v11.1
- 4H הוא ה-Setup המרכזי ו-1H משמש לתזמון.
- Smart Fib נשאר כלי retracement/continuation בלבד; מעבר 0.618 אינו breakout בפני עצמו.
- Institutional Fast Pressure: VIX1D, VVIX acceleration, VIX1D/VIX9D acceleration, SKEW/COR1M.
- נרות סגורים בלבד לטריגרים הטכניים.
- Multi-source + retry וגיבויי נתונים.

## התקנה
```bash
pip install -r requirements.txt
streamlit run app.py
```

זהו כלי מחקרי ואינו מבצע עסקאות אוטומטית.
