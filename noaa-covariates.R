library(arrow)
library(dplyr)
s3 <- s3_bucket("neon4cast-drivers/noaa/gefs-v12/stage1",
                endpoint_override = "data.ecoforecast.org",
                anonymous = TRUE)
ds <- open_dataset(s3, partitioning = c("start_date", "cycle"))

# 393 ms

ex <- ds |> filter(cycle == 0,
                   variable == "TMP",
                   start_date == "2022-04-01",
                   site_id == "BART",
                   ensemble == 1)
ex |> collect()
