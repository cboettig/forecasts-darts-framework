#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.datasets import AirPassengersDataset


# In[2]:


series = AirPassengersDataset().load()
series.plot()


# In[3]:


series1, series2 = series.split_before(0.75)
series1.plot()
series2.plot()


# In[4]:


train, val = series.split_before(pd.Timestamp("19580101"))
train.plot(label="training")
val.plot(label="validation")



# In[5]:


from darts.models import NaiveSeasonal

naive_model = NaiveSeasonal(K=1)
naive_model.fit(train)
naive_forecast = naive_model.predict(36)

series.plot(label="actual")
naive_forecast.plot(label="naive forecast (K=1)")


# In[6]:


seasonal_model = NaiveSeasonal(K=12)
seasonal_model.fit(train)
seasonal_forecast = seasonal_model.predict(36)

series.plot(label="actual")
seasonal_forecast.plot(label="naive forecast (K=12)")



# In[7]:


from darts.models import NaiveDrift

drift_model = NaiveDrift()
drift_model.fit(train)
drift_forecast = drift_model.predict(36)

combined_forecast = drift_forecast + seasonal_forecast - train.last_value()

series.plot()
combined_forecast.plot(label="combined")
drift_forecast.plot(label="drift")



# In[8]:


import darts.metrics

print(
    "Mean absolute percentage error for the combined naive drift + seasonal: {:.2f}%.".format(
        darts.metrics.mape(series, combined_forecast)
    )
)



# # Probablistic Forecasts

# In[9]:


from darts.models import ExponentialSmoothing, Prophet, AutoARIMA, Theta
model_es = ExponentialSmoothing()
model_es.fit(train)
probabilistic_forecast = model_es.predict(len(val), num_samples=500)

series.plot(label="actual")
probabilistic_forecast.plot(label="probabilistic forecast")
plt.legend()
plt.show()



# In[10]:


from darts.models import TCNModel
from darts.utils.likelihood_models import LaplaceLikelihood


model = TCNModel(
    input_chunk_length=24,
    output_chunk_length=12,
    random_state=42,
    likelihood=LaplaceLikelihood()
)
model.trainer_params


# In[11]:


from darts.dataprocessing.transformers import Scaler
scaler = Scaler()
train_scaled = scaler.fit_transform(train)


# In[12]:


model.fit(train_scaled, epochs=400, verbose=True)


# In[16]:


pred = model.predict(n=36, num_samples=500)


# In[17]:


pred.plot(low_quantile=0.01, high_quantile=0.99, label="1-99th percentiles")
pred.plot(low_quantile=0.2, high_quantile=0.8, label="20-80th percentiles")


# In[18]:


train.plot(label="training")
val.plot(label="validation")
pred = scaler.inverse_transform(pred)
pred.plot(low_quantile=0.01, high_quantile=0.99, label="1-99th percentiles")
pred.plot(low_quantile=0.2, high_quantile=0.8, label="20-80th percentiles")

