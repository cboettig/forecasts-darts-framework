library(arrow)
library(dplyr)
library(readr)




## Daily.  we use daily drivers,

target <- read_csv("https://data.ecoforecast.org/targets/terrestrial_daily/terrestrial_daily-targets.csv.gz")

noaa <- s3_bucket("neon4cast-drivers/noaa/gefs-v12/stage1",
                endpoint_override = "data.ecoforecast.org",
                anonymous = TRUE)
ds <- open_dataset(noaa, partitioning = c("start_date", "cycle"))


## daily mean, min, max
ex <- ds |> filter(cycle == 0,
                   variable == "TMP",
                   start_date == "2022-04-01") |>
  mutate(day = horizon %/% 24) |>
  group_by(start_date, variable, day, ensemble, site_id) |>
  summarise(mean = mean(predicted), min = min(predicted), max = max(predicted))
ex |> collect()


## NEON based access
neon <- s3_bucket("neon4cast-targets/neon",
                  endpoint_override = "data.ecoforecast.org",
                  anonymous = TRUE)
# list tables with `neon$ls()`
rh <- open_dataset(neon$path("RH_30min-basic-DP1.00098.001"))     # Relative Humidity
tmp <- open_dataset(neon$path("TAAT_30min-basic-DP1.00003.001"))  # Temperature
precip <- open_dataset(neon$path("PRIPRE_30min-basic-DP1.00006.001")) # Precipitation


# PRES Pressure
# TMP Temperature
# RH Relative Humidity
# UGRD U-component of wind
# VGRD V-component of wind
# APCP Total precip (kg/m^2 in 3 or 6-hr interval)
# DSWRF Downward shortwave radiation flux
# DLWRF Downward longwave radiation flux




noaa <- s3_bucket("neon4cast-drivers/noaa/gefs-v12/stage1",
                  endpoint_override = "data.ecoforecast.org",
                  anonymous = TRUE)
ds <- open_dataset(noaa, partitioning = c("start_date", "cycle"))


## daily mean, min, max
ex <- ds |> filter(cycle == 0,
                   variable == "TMP",
                   ensemble == 1,
                   site_id == "BART",
                   start_date > "2022-01-01",
                   start_date < "2022-07-20",
                   ) |>
  mutate(day = horizon %/% 24) |>
  group_by(start_date, variable, day, ensemble, site_id) |>
  summarise(mean = mean(predicted), min = min(predicted), max = max(predicted))
ex |> collect()


