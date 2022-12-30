## simplify the data (could be done in python)
library(tidyverse)
aquatics <- read_csv("https://data.ecoforecast.org/neon4cast-targets/aquatics/aquatics-targets.csv.gz")
aquatics <- aquatics |>
  select(!contains(c("depth", "sd"))) |>
  group_by(time, siteID) |>
  summarise(across(!any_of(c("time", "siteID")), .fns = mean, na.rm=TRUE),
            .groups="drop")


## consider GP-fill since python is failing to do that?

write_csv(aquatics, "aquatics-targets.csv.gz")