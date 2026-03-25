suppressPackageStartupMessages({
  library(nlmixr2)
  library(nlmixr2data)
})

source("reference_fit_helpers.R")

warfarin_pk_data <- subset(
  warfarin,
  select = c(id, time, amt, dv, evid, wt, age, sex)
)

warfarin_pk <- function() {
  ini({
    tka <- log(1.0)
    tcl <- log(0.08)
    tv <- log(8.0)
    eta.ka ~ 0.3
    eta.cl ~ 0.2
    eta.v ~ 0.1
    prop.sd <- 0.15
    add.sd <- 0.02
  })
  model({
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v <- exp(tv + eta.v)
    d/dt(depot) = -ka * depot
    d/dt(center) = ka * depot - cl / v * center
    cp = center / v
    cp ~ prop(prop.sd) + add(add.sd)
  })
}

fit <- suppressMessages(
  nlmixr2(
    warfarin_pk,
    warfarin_pk_data,
    est = "focei",
    control = foceiControl(print = 0)
  )
)

write_reference_artifact(
  fit = fit,
  data = warfarin_pk_data,
  estimator = "focei",
  dataset_name = "warfarin",
  model_name = "warfarin_pk",
  output_path = "reference-warfarin-foce-fit.json"
)
