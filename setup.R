#reticulate::install_miniconda()
# renv::use_python(type="conda")

library(reticulate)
reticulate::use_condaenv("renv/python/condaenvs/renv-python/")
reticulate::py_install(c("pytorch", "u8darts-all", "prophet", "pyarrow"))
