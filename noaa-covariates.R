library(arrow)
library(dplyr)

peek <- function(df) collect(head(df))
s3 <- s3_bucket("drivers/noaa/neon/gefs", endpoint_override = "data.ecoforecast.org", anonymous = TRUE)
ds <- open_dataset(s3)

latest <- ds |> summarise(x = max(start_time)) |> pull(x)

ex <- ds |> filter(site_id == "BART", variable == "RH", start_time == latest) |> collect()
ex |> count(horizon, sort=TRUE)
