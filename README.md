# VIX Tactical v11.3 — Fast Continuation / Late Entry

גרסה זו ממשיכה את v11.2 12H Reversal Confluence ומוסיפה שכבת כניסה שנייה לעסקאות קצרות כאשר ה-Entry הראשי כבר התפספס אבל כיוון ה-VIX עדיין ברור.

## מה חדש ב-v11.3
- נוסף **FAST CONTINUATION / LATE ENTRY** — מצב משני בלבד; הוא לא יוצר כיוון חדש ולא משנה את ציון ה-VIX הראשי.
- הכיוון ממשיך להגיע מה-VIX. רק אחרי `DEVELOPING / ENTRY READY` או איתות ליבה שעבר את סף העסקה, המצב המהיר נפתח.
- QQQ 4H/1H משמש רק לתזמון ביצוע: מחפשים Pullback קטן ולאחריו חידוש מומנטום בכיוון שכבר אושר.
- במקביל נדרש אישור הפוך ב-VIX 1H, כדי לא להיכנס כאשר ה-VIX מתחיל להסתובב נגד העסקה.
- נוסף **No-Chase Gate**: אם נר החידוש כבר נמתח מדי או שהסטופ המבני גדול מדי ביחס ל-ATR, מתקבל `NO CHASE` ומחכים ל-Pullback נוסף.
- כאשר מתקבל `FAST CONTINUATION · ENTRY READY`, האפליקציה מציגה QQQ Entry, Stop, TP1, TP2, אחוזי סיכון/יעד ו-R:R.
- היעד של המצב הזה הוא עסקה קצרה של כמה שעות ועד יום מסחר אחד, לא ניסיון לתפוס מחדש את כל המהלך.
- אם נתוני QQQ אינם זמינים, מנוע ה-VIX הראשי ממשיך לעבוד כרגיל; רק שכבת Fast Continuation מסומנת כלא זמינה.

## מה נשאר מ-v11.2
- **12H Reversal Confluence Engine**: S/R רב-נגיעות + Smart Fib 0.50–0.618 + Rising/Falling Wedge.
- סטטוסים: **WATCH → DEVELOPING → ENTRY READY**.
- 4H Early RSI Divergence כחיזוק בלבד כאשר קיים 12H Confluence.
- 4H הוא ה-Setup המרכזי, 1H הוא תזמון, ו-12H נותן כיוון/Reversal מוקדם.
- Institutional Fast Pressure: VIX1D, VVIX acceleration, VIX1D/VIX9D acceleration, SKEW/COR1M.
- נרות סגורים בלבד לטריגרים הטכניים.
- Multi-source + retry וגיבויי נתונים.

## התקנה
```bash
pip install -r requirements.txt
streamlit run app.py
```

זהו כלי מחקרי ואינו מבצע עסקאות אוטומטית.
