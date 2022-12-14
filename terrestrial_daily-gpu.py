import pandas as pd
from darts.models import GaussianProcessFilter, TCNModel
from darts.utils.likelihood_models import LaplaceLikelihood
from darts.dataprocessing.transformers import Scaler
from sklearn.gaussian_process.kernels import RBF
from utils.forecast_utils import submit, ts_parser, efi_format

# read the forecasting data
targets = pd.read_csv("https://data.ecoforecast.org/neon4cast-targets/terrestrial_daily/terrestrial_daily-targets.csv.gz")
horizon = 35
variables = ["nee", "le"]

## Define model
model = TCNModel(
    input_chunk_length=365,  # pts_in_yr
    output_chunk_length=horizon, # forecast horizon
    random_state=42,
    likelihood=LaplaceLikelihood(),
    pl_trainer_kwargs={"accelerator": "gpu", "gpus": 1, "auto_select_gpus": True} 
    )


# Most NN models need interpolation of missing values first.
# We set up a GP to filter the data:
gpf = GaussianProcessFilter(kernel=RBF(), alpha=0.1, normalize_y=True)
## Most NN models work best when data is rescaled as well.
## We set up a "scaler" to do this:
scaler = Scaler()


## Now for any timeseries (site, variable), we can:
# 1. subset that timeseries from the target
# 2. interpolate via GP
# 3. re-scale data
# 4. fit the NN model! (slowest step, best with GPU!)
# 5. generate predicted forecast (100 reps) from model
# 6. transform back to original units
# 7. Convert from xarray to EFI standard format
full = pd.DataFrame()
sites = targets["site_id"].unique()

for variable in variables:
    print(variable)
    for site_id in sites:
      print(site_id)
      train = ts_parser(targets, site_id, variable, freq = "D")
      filtered_series = gpf.filter(train, num_samples=10) 
      train_scaled = scaler.fit_transform(filtered_series)
      model.fit(train_scaled, epochs=100)
      pred = model.predict(n=horizon, num_samples=100)
      forecast = scaler.inverse_transform(pred)
      df = efi_format(forecast, site_id, variable)
      full = pd.concat([full,df])

import pyarrow as pa
import pyarrow.parquet as pq
table = pa.Table.from_pandas(full)
pq.write_table(table, 'example.parquet')

submit(table, "terrestrial_daily", "cb_tcnmodel")


#import matplotlib.pyplot as plt
#plt.cla()
#pred.plot(low_quantile=0.2, high_quantile=0.8, label="20-80th percentiles")
#plt.show()

