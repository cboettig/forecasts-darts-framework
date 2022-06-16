import pandas as pd
from darts.models import ExponentialSmoothing, Prophet, AutoARIMA, Theta
from utils.forecast_utils import forecast_each, submit
from darts import TimeSeries

## Needs targets averaged over depths
targets = pd.read_csv("aquatics-targets.csv.gz")
horizon = 14
variables = ["temperature", "oxygen", "chla"]
model = Prophet()
full = forecast_each(model, targets, variables, horizon, freq = "D")
submit(full, "aquatics", "cb_prophet")

