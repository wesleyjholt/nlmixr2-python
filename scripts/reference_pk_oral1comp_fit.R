suppressPackageStartupMessages({
  library(nlmixr2)
  library(nlmixr2data)
})

source("reference_fit_helpers.R")

oral_data <- subset(
  Oral_1CPT,
  SD == 1 & SS == 99,
  select = c(ID, TIME, DV, AMT, EVID, CMT)
)

oral_pk <- function() {
  ini({
    tka <- log(1.0)
    tcl <- log(5.0)
    tv <- log(90.0)
    eta.ka ~ 0.2
    eta.cl ~ 0.2
    eta.v ~ 0.1
    prop.sd <- 0.15
  })
  model({
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v <- exp(tv + eta.v)
    d/dt(depot) = -ka * depot
    d/dt(center) = ka * depot - cl / v * center
    cp = center / v
    cp ~ prop(prop.sd)
  })
}

fit <- suppressMessages(
  nlmixr2(
    oral_pk,
    oral_data,
    est = "focei",
    control = foceiControl(print = 0)
  )
)

write_reference_artifact(
  fit = fit,
  data = oral_data,
  estimator = "focei",
  dataset_name = "Oral_1CPT",
  model_name = "oral_pk",
  output_path = "reference-pk-oral1comp-fit.json"
)
