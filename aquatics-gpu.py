import pandas as pd
from utils.forecast_utils import forecast_each, submit
from darts import TimeSeries
from darts.models import TCNModel
from darts.utils.likelihood_models import LaplaceLikelihood
from darts.dataprocessing.transformers import Scaler


horizon = 35
variables = ["temperature", "oxygen", "chla"]

model = TCNModel(
    input_chunk_length=400,  # pts_in_yr
    output_chunk_length=horizon,
    random_state=42,
    likelihood=LaplaceLikelihood(),
    pl_trainer_kwargs={"accelerator": "gpu", "gpus": 1, "auto_select_gpus": True} 

)

targets = pd.read_csv("https://data.ecoforecast.org/neon4cast-targets/aquatics/aquatics-targets.csv.gz")


from darts.models import GaussianProcessFilter
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
kernel = RBF(length_scale=1) + WhiteKernel(noise_level=1)
gpf = GaussianProcessFilter(kernel=kernel, normalize_y=True)
scaler = Scaler()


full = forecast_each(model, targets, variables, horizon, freq = "D",
                     num_samples=30, interp=gpf, scaler = scaler)

submit(full, "aquatics", "cb_TCN")

