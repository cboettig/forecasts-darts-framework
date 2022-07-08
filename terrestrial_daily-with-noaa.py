import pandas as pd
from darts.models import Prophet
from utils.forecast_utils import forecast_each, submit

horizon = 35
targets = pd.read_csv("https://data.ecoforecast.org/targets/terrestrial_daily/terrestrial_daily-targets.csv.gz")
variables = ["nee", "le"]


import pyarrow.dataset as ds
from pyarrow import fs

s3 = fs.S3FileSystem(endpoint_override = "data.ecoforecast.org", anonymous = True)

dataset = ds.dataset(
    "drivers/noaa/neon/gefs",
    format="parquet",
    filesystem=s3,
    partitioning="hive"
)
expression = ((ds.field("variable") == "RH") & 
              (ds.field("site_id") == "BART"))
              
bart_rh = dataset.to_table(filter=expression)
bart_rh



