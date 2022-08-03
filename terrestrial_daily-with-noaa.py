from darts import TimeSeries
import pandas as pd
import pyarrow.dataset as ds
from pyarrow import fs


s3 = fs.S3FileSystem(endpoint_override = "data.ecoforecast.org", anonymous = True)
dataset = ds.dataset(
    "neon4cast-drivers/noaa/gefs-v12/stage1",
    format="parquet",
    filesystem=s3,
    partitioning=["start_date", "cycle"]
)
expression = (
              (ds.field("site_id") == "BART") &
              (ds.field("variable") == "TMP") &
              (ds.field("start_date") == "2022-07-01") &
              (ds.field("cycle") == 0) &
              (ds.field("ensemble")  == 1)
              )
ex = dataset.to_table(filter=expression)
predicted = ex.to_pandas()[['time', 'predicted']]



neon_temp = ds.dataset(
    "neon4cast-targets/neon/TAAT_30min-basic-DP1.00003.001",
    format="parquet",
    filesystem=s3)
expression = (ds.field("site_id") == "BART")
historical = neon_temp.to_table(filter=expression)

df = ex.to_pandas()[['time', 'predicted']]


## See  darts.TimeSeries.from_group_dataframe to build a LIST of timeseries from a GROUPED dataframe

# ICK this throws: `Error: Cannot interpret 'datetime64[ns, UTC]' as a data type`
#series = TimeSeries.from_dataframe(df, time_col = 'time', value_cols = 'predicted', fill_missing_dates = True)

# Manually parse the datetime64[ns, UTC] date into a pandas index:
datetime_series = pd.to_datetime(df['time'], infer_datetime_format=True, utc=True)
datetime_index = pd.DatetimeIndex(datetime_series.values)
values=df.set_index(datetime_index)
values.drop('time', axis=1,inplace=True)
series = TimeSeries.from_dataframe(values, value_cols = 'predicted', fill_missing_dates = True, freq = '3H')






horizon = 35
targets = pd.read_csv("https://data.ecoforecast.org/targets/terrestrial_daily/terrestrial_daily-targets.csv.gz")
variables = ["nee", "le"]

