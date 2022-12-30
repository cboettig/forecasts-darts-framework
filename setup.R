#reticulate::install_miniconda()
#renv::use_python(type="conda")

library(reticulate)
#reticulate::use_condaenv("renv/python/condaenvs/renv-python/")
reticulate::virtualenv_create("./venv")
reticulate::use_virtualenv("./venv")
reticulate::py_install(c("pytorch", "u8darts-all", "prophet", "pyarrow"))
