import pandas as pd
from darts.models import ExponentialSmoothing, Prophet, AutoARIMA, Theta
from utils.forecast_utils import forecast_each, submit
from darts import TimeSeries

targets = pd.read_csv("https://data.ecoforecast.org/neon4cast-targets/aquatics/aquatics-targets.csv.gz")

horizon = 35
variables = ["temperature", "oxygen", "chla"]
model = Prophet()
full = forecast_each(model, targets, variables, horizon, freq = "D")
submit(full, "aquatics", "cb_prophet")

