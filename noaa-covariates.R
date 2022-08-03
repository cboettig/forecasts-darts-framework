library(arrow)
library(dplyr)
library(readr)
library(tsibble)
library(fable)

## Daily.  we use daily drivers,
target <- read_csv("https://data.ecoforecast.org/targets/terrestrial_daily/terrestrial_daily-targets.csv.gz") |>
  rename(site_id = siteID) |>
  as_tsibble(index=time, key=site_id)



## NEON based access
neon <- s3_bucket("neon4cast-targets/neon",
                  endpoint_override = "data.ecoforecast.org",
                  anonymous = TRUE)
# list tables with `neon$ls()`

## Triple-aspirated temperature:
neon_temp <- open_dataset(neon$path("TAAT_30min-basic-DP1.00003.001")) |>
  mutate(time = as.Date(startDateTime)) |>
  group_by(siteID, time) |>
  summarise(mean_tmp = mean(tempTripleMean, na.rm = TRUE),
            min_tmp = min(tempTripleMinimum, na.rm = TRUE),
            max_tmp = max(tempTripleMaximum, na.rm = TRUE)) |>
  collect() |>
  rename(site_id = siteID) |>
  as_tsibble(index=time, key=site_id)


# Relative Humidity (also contains tempRH and dewTemp)
rh <- open_dataset(neon$path("RH_30min-basic-DP1.00098.001"))  |>
  mutate(time = as.Date(startDateTime)) |>
  group_by(siteID, time) |>
  summarise(mean_rh = mean(RHMean, na.rm = TRUE),
            min_rh = min(RHMinimum, na.rm = TRUE),
            max_rh = max(RHMaximum, na.rm = TRUE)) |>
  collect() |>
  rename(site_id = siteID) |>
  as_tsibble(index=time, key=site_id)

# Precipitation (priPrecipBulk)
precip <- open_dataset(neon$path("PRIPRE_30min-basic-DP1.00006.001")) |>
  mutate(time = as.Date(startDateTime)) |>
  group_by(siteID, time) |>
  summarise(sum_precip = sum(priPrecipBulk, na.rm = TRUE)) |>
  collect() |>
  rename(site_id = siteID) |>
  as_tsibble(index=time, key=site_id)

precip |> filter(sum_precip > 0)

precip

## Build a data.frame with additional columns for additional predictor variables
matrix <- target |> left_join(neon_temp) |> left_join(rh) |> left_join(precip)



## Add lags?

fit <- matrix |> fill_gaps() |>
  model(tslm = TSLM(nee ~ mean_tmp + min_tmp +
                      max_tmp + mean_rh + min_rh + max_rh))

## Precip data is really lacking, left_join introduces too many NAs
matrix2 <- target |> left_join(neon_temp) |> left_join(rh) |> inner_join(precip)
fit2 <- matrix2 |>
  model(tslm = TSLM(nee ~ mean_tmp + min_tmp +
                      max_tmp + mean_rh + min_rh + max_rh + sum_precip))




###  NOAA access

# PRES Pressure
# TMP Temperature (C)
# RH Relative Humidity (%)
# UGRD U-component of wind
# VGRD V-component of wind
# APCP Total precip (kg/m^2 in 3 or 6-hr interval)
# DSWRF Downward shortwave radiation flux
# DLWRF Downward longwave radiation flux

## Note: kg/m^2 of rain (NOAA units) is the same thing as 1mm (NEON units) of rain!

noaa <- s3_bucket("neon4cast-drivers/noaa/gefs-v12/stage1",
                  endpoint_override = "data.ecoforecast.org",
                  anonymous = TRUE)

## daily mean, min, max.  For simplicity, we summarize over ensemble
noaa_forecast <-
  open_dataset(noaa, partitioning = c("start_date", "cycle")) |>
  filter(cycle == 0,
         variable %in% c("RH", "TMP"),
         start_date == "2022-06-01") |>
  mutate(day = horizon %/% 24) |>
  group_by(start_date, variable, day, site_id) |>
  summarise(mean = mean(predicted, na.rm = TRUE),
            min = min(predicted, na.rm = TRUE),
            max = max(predicted, na.rm = TRUE))|>
  collect() |>
  mutate(time = as.Date(start_date) + day)

## Forecast precipitation as sum
noaa_precip <-
  open_dataset(noaa, partitioning = c("start_date", "cycle")) |>
  filter(cycle == 0,
         variable == "APCP",
         start_date == "2022-06-01") |>
  mutate(day = horizon %/% 24) |>
  group_by(start_date, variable, day, site_id) |>
  summarise(sum_precip = sum(predicted, na.rm = TRUE)) |>
  collect() |>
  mutate(time = as.Date(start_date) + day) |>
  select(-variable)

noaa_fc <- noaa_forecast |>
  as_tsibble(index=time, key=c(site_id, variable)) |>
  pivot_wider(names_from = variable, values_from = c(mean, min, max)) |>
  rename_all(tolower) |>
  left_join(noaa_precip)





## Forecasting requires new data:
fit |> forecast(h = "35 days", new_data = noaa_fc)

