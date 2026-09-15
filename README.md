# VIX Swing — Simple Score UI

גרסה פשוטה ומעודכנת של אפליקציית VIX Swing.

## עיקרי הלוגיקה
- VIX מול EMA9
- EMA9 מול EMA26 של VIX
- Golden / Death Cross + מד קרבה 1–10
- VIX9D / VIX
- שינוי VIX יום אחד ו-5 ימים
- Term Structure דרך VIX / VIX3M
- Nasdaq/S&P מול EMA26 + מומנטום 2/5 ימים
- RSI של VIX אינו מקבל ניקוד ישיר
- RSI משמש רק ל-Divergence של VIX ב-4H וב-12H
- Support / Resistance עד כחודש אחורה
- סף אות: ±3.5
- 15m אינו חלק מהאפליקציה; אישור סופי 1H

## הרצה
```bash
pip install -r requirements.txt
streamlit run app.py
```
