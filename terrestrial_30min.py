
import pandas as pd
from darts.models import ExponentialSmoothing, Prophet, AutoARIMA, Theta
from utils.forecast_utils import forecast_each, submit
from darts import TimeSeries


targets = pd.read_csv("https://data.ecoforecast.org/targets/terrestrial_30min/terrestrial_30min-targets.csv.gz")

#datetime_series = pd.to_datetime(df['time'], infer_datetime_format=True, utc=True)
#datetime_index = pd.DatetimeIndex(datetime_series.values)
#targets=df.set_index(datetime_index)
#targets.drop('time', axis=1,inplace=True)

horizon = 35 * 2 * 24
variables = ["nee", "le"]
model = Prophet()
full = forecast_each(model, targets, variables, horizon, freq = None)
submit(full, "terrestrial_30min", "cb_prophet")


