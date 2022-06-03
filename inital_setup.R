## Not needed once the project is using renv, this will all be setup.

#install.packages("renv")
renv::activate()
renv::use_python()
install.packages(c("reticulate", "yaml"))
library(reticulate)

## requirements.txt does not capture that numpy must be build from source
reticulate::py_install("numpy",
                       pip_options="--no-binary=':all:'",
                       ignore_installed=TRUE)

reticulate::py_install(c("darts","jupyter", "plotly"))
renv::snapshot()


np <- reticulate::import("numpy")
np$show_config()
