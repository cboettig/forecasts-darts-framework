import pandas as pd
from darts.models import ExponentialSmoothing, Prophet, AutoARIMA, Theta
from utils.forecast_utils import forecast_each, submit

horizon = 35

targets = pd.read_csv("https://data.ecoforecast.org/targets/terrestrial_daily/terrestrial_daily-targets.csv.gz")
variables = ["nee", "le"]
model = Prophet()
full = forecast_each(model, targets, variables, horizon)
submit(full, "terrestrial_daily", "cb_prophet")


