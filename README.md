# VIX Swing v10.1 — Multi-Source Resilient

גרסה מחוזקת של VIX Swing עם מנגנון גיבוי למקורות נתונים.

## מה השתנה
- Retry אוטומטי למשיכות Yahoo/yfinance.
- מסלול גיבוי ישיר דרך Yahoo Chart API (query1/query2), במקרה שה-wrapper של yfinance נכשל.
- גיבוי Daily רשמי של Cboe עבור VIX / VIX9D / VIX3M / VVIX.
- גיבוי Daily עצמאי של Stooq עבור Nasdaq 100 / S&P 500.
- אבחון ברור: האפליקציה מציגה איזה feed נכשל ומאיזה מקור כל סדרה התקבלה.
- נתוני 12H נשארים קשיחים: אם אין נתוני 60m אמינים, האפליקציה לא מחליפה אותם ב-Daily כדי לא לשנות את לוגיקת השיטה.

## Streamlit Cloud
העלה את שלושת הקבצים (`app.py`, `requirements.txt`, `README.md`) לאותו repository והפעל את `app.py` כרגיל.

אין צורך ב-API key לגרסה הזאת.
