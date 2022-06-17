import pandas as pd
from darts.models import Prophet
from utils.forecast_utils import forecast_each, submit

horizon = 35

targets = pd.read_csv("https://data.ecoforecast.org/targets/phenology/phenology-targets.csv.gz")
variables = ["gcc_90", "rcc_90"]
model = Prophet()
full = forecast_each(model, targets, variables, horizon)
submit(full, "phenology", "cb_prophet")



