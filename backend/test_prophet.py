from prophet import Prophet
import pandas as pd

df = pd.DataFrame({
    'ds': pd.date_range('2026-01-01', periods=30, freq='D'),
    'y': range(30)
})

m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=5)
forecast = m.predict(future)
print(forecast[['ds', 'yhat']].tail())
print("Prophet is working correctly!")