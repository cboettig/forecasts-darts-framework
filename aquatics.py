
import os
import numpy as np
import pandas as pd
from darts.models import ExponentialSmoothing, Prophet, AutoARIMA, Theta
from utils.forecast_utils import ts_parser, efi_format, forecast_csv

# import matplotlib.pyplot as plt
# GPU configuration, optional
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#pl_trainer_kwargs={"accelerator": "gpu", "gpus": 1, "auto_select_gpus": True} 

# Set to True to generate a forecast of the last 25% of target data using first 75%
dev_test = True

# Forecast horizon
horizon = 14

target = pd.read_csv("https://data.ecoforecast.org/targets/aquatics/aquatics-targets.csv.gz")
variables = ["temperature", "oxygen", "chla"]
sites = target["siteID"].unique()

full = pd.DataFrame()

# Note: still specific to old format targets file
for variable in variables:
  for site_id in sites:
    train = ts_parser(target, site_id, variable)
    
    # development purposes only, production runs will use all available data
    if dev_test:
      train, val = train.split_before(0.75)
      horizon = len(val)
    
    # Other options include: ExponentialSmoothing(), AutoARIMA(), Theta() 
    model = Prophet()
    model.fit(train)
    forecast = model.predict(horizon, num_samples=500)
    df = efi_format(forecast, site_id, variable)
    full = pd.concat([full,df])

#forecast_csv(full, theme = "aquatics", team = "cb_prophet")
#submit(full, theme = "aquatics", team = "cb_prophet")



