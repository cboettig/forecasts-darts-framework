import os
import pandas as pd
from datetime import date
from darts import TimeSeries

from pyarrow import fs
import pyarrow as pa
from pyarrow import csv

#FIXME can darts.TimeSeries have multiple variables? multiple sites?

def ts_parser(df, siteID, variable, freq=None):
  df = df[df["siteID"] == siteID]
  df = df[['time', variable]]
  datetime_series = pd.to_datetime(df['time'], infer_datetime_format=True, utc=True)
  datetime_index = pd.DatetimeIndex(datetime_series.values)
  targets=df.set_index(datetime_index)
  targets.drop('time', axis=1,inplace=True)
  series = TimeSeries.from_dataframe(targets, value_cols = variable, fill_missing_dates = True, freq = freq)
  return(series)

#FIXME can we handle multiple variables?
def efi_format(pred, site_id, variable):
  nd = pred.all_values() # numpy nd-array
  nd.shape # time x variables x replicates
  # index the first variable
  var1 = nd[:,0,:]
  var1.shape # time x replicate -- 2D array can be converted to DataFrame
  
  df = pd.DataFrame(var1)
  df['time'] = pred.time_index
  # pivot longer, ensemble as id, not as column name
  df = df.melt(id_vars="time", var_name="ensemble", value_name="predicted")
  
  df['variable'] = variable # "temperature"
  df['site_id'] = site_id # "BART"
  df = df[['time', 'site_id', 'variable', 'predicted', 'ensemble']]
  return(df)


def fc_name(theme, team, pub_time = date.today()):
  filename = theme + "-" + pub_time.strftime("%Y-%m-%d") + "-" + team +  ".csv.gz"
  return(filename)


def submit(forecast_df, theme, team = "cb_prophet", pub_time = date.today()):
    
  os.environ["AWS_EC2_METADATA_DISABLED"] = "TRUE"
  os.environ["AWS_DEFAULT_REGION"] = ""
  os.environ["AWS_S3_ENDPOINT"] = ""
  os.environ["AWS_ACCESS_KEY_ID"] = ""
  os.environ["AWS_SECRET_ACCESS_KEY"] = ""
  
  s3, path = fs.FileSystem.from_uri("s3://submissions?endpoint_override=data.ecoforecast.org")
  filename = fc_name(theme = theme, team = team, pub_time = pub_time)
  where = path + "/" + filename
  table = pa.Table.from_pandas(forecast_df)
  
  with s3.open_output_stream(where) as file:
    csv.write_csv(table, file)

# Model options include: Prophet(), ExponentialSmoothing(), AutoARIMA(), Theta() 

def forecast_each(model, targets, variables, horizon, freq = "D"):
  full = pd.DataFrame()
  sites = targets["siteID"].unique()

  for variable in variables:
    print(variable)
    for site_id in sites:
      print(site_id)
      train = ts_parser(targets, site_id, variable, freq = freq)
      model.fit(train)
      forecast = model.predict(horizon, num_samples=500)
      df = efi_format(forecast, site_id, variable)
      full = pd.concat([full,df])
  return(full)





# local csv writer
def forecast_csv(df, theme, team, pub_time = date.today()):
  filename = fc_name(theme, team, pub_time)
  df.to_csv(filename, index=False)


def ts_parser_orig(df, siteID, variable, freq=None):
  df = df[df["siteID"] == siteID]
  df = df[['time', variable]]
  series = TimeSeries.from_dataframe(df, time_col = "time", value_cols = variable, fill_missing_dates = True, freq = freq)
  return(series)
