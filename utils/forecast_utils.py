import os
import pandas as pd
from datetime import date, datetime
from darts import TimeSeries
from pyarrow import fs
import pyarrow as pa
from pyarrow import csv

# DEVNOTE: can darts.TimeSeries have multiple variables? multiple sites?
# See darts.TimeSeries.from_group_dataframe to build a LIST of timeseries from a GROUPED dataframe

def ts_parser(df, site_id, variable, freq=None, reference_datetime = date.today()):
  df = df[df["site_id"] == site_id]
  df = df[df["variable"] == variable]
  df = df[df["datetime"]  <= str(reference_datetime)]
  if len(df.index) == 0:
    return(None)
  datetime_series = pd.to_datetime(df["datetime"], infer_datetime_format=True, utc=True)
  datetime_index = pd.DatetimeIndex(datetime_series.values)
  targets=df.set_index(datetime_index)
  targets.drop("datetime", axis=1,inplace=True)
  series = TimeSeries.from_dataframe(targets, value_cols = "observation", fill_missing_dates = True, freq = freq)
  return(series)

def efi_format(pred, site_id, variable, reference_datetime = date.today()):
  nd = pred.all_values() # numpy array: time x variables x replicates
  var1 = nd[:,0,:] # index the first (only) variable
  df = pd.DataFrame(var1)
  df["datetime"] = pred.time_index
  reftime = datetime.combine(reference_datetime, datetime.min.time())
  df = df[ (df['datetime'] >= reftime) ]
  # pivot longer, ensemble as id, not as column name
  df = df.melt(id_vars="datetime", var_name="parameter", value_name="prediction")
  df["variable"] = variable # "temperature"
  df["site_id"] = site_id # "BART"
  df["family"] = "ensemble"
  df["reference_datetime"] = reference_datetime
  df = df[["datetime", "site_id", "variable",
           "prediction", "parameter", "family"]]
  return(df)


def fc_name(theme, team, pub_time = date.today()):
  filename = theme + "-" + pub_time.strftime("%Y-%m-%d") + "-" + team +  ".csv"
  return(filename)

def submit(forecast_df, theme, team, pub_time = date.today()):
  os.environ["AWS_EC2_METADATA_DISABLED"] = "TRUE"
  os.environ["AWS_DEFAULT_REGION"] = ""
  os.environ["AWS_S3_ENDPOINT"] = ""
  os.environ["AWS_ACCESS_KEY_ID"] = ""
  os.environ["AWS_SECRET_ACCESS_KEY"] = ""
  s3, path = fs.FileSystem.from_uri("s3://neon4cast-submissions?endpoint_override=data.ecoforecast.org")
  filename = fc_name(theme = theme, team = team, pub_time = pub_time)
  where = path + "/" + filename
  table = pa.Table.from_pandas(forecast_df, preserve_index=False)
  with s3.open_output_stream(where) as file:
    csv.write_csv(table, file)

def isNaN(train):
  x = train.values()
  return all(x != x)


def forecast_each(model, targets, variables, horizon, freq = "D",
                  num_samples = 30, interp=None, scaler = None,
                  reference_datetime = date.today()):
  full = pd.DataFrame()
  sites = targets["site_id"].unique()

  ## Seriously learn to parallelize this 
  for variable in variables:
    print(variable)
    for site_id in sites:
      print(site_id)
      
      train = ts_parser(targets, site_id, variable, freq = freq,
                        reference_datetime = reference_datetime)
      if train is None:
        continue                        
      if isNaN(train):
        continue
      if interp is not None:
        train = interp.filter(train)
      if scaler is not None:
        train = scaler.fit_transform(train)
      
      ## FIXME -- support additional model kwargs, e.g. epochs  
      model.fit(train)
      forecast = model.predict(horizon, num_samples=num_samples)
      if scaler is not None:
        forecast = scaler.inverse_transform(forecast)
      
      df = efi_format(forecast, site_id, variable, reference_datetime)
      full = pd.concat([full,df])
  return(full)



