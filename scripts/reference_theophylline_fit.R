suppressPackageStartupMessages({
  library(jsonlite)
  library(nlmixr2)
})

one.cmt <- function() {
  ini({
    tka <- 0.45
    tcl <- 1
    tv <- 3.45
    eta.ka ~ 0.6
    eta.cl ~ 0.3
    eta.v ~ 0.1
    add.sd <- 0.7
  })
  model({
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v <- exp(tv + eta.v)
    d/dt(depot) = -ka * depot
    d/dt(center) = ka * depot - cl / v * center
    cp = center / v
    cp ~ add(add.sd)
  })
}

fit <- suppressMessages(
  nlmixr2(
    one.cmt,
    theo_sd,
    est = "focei",
    control = foceiControl(print = 0)
  )
)

extract_scalar <- function(frame, column) {
  if (!column %in% names(frame)) {
    return(NULL)
  }
  values <- frame[[column]]
  if (length(values) == 0L) {
    return(NULL)
  }
  out <- suppressWarnings(as.numeric(values[[1]]))
  if (is.na(out)) {
    return(NULL)
  }
  unname(out)
}

par_fixed <- as.data.frame(fit$parFixedDf)
estimate_col <- if ("Est." %in% names(par_fixed)) {
  "Est."
} else if ("Estimate" %in% names(par_fixed)) {
  "Estimate"
} else {
  numeric_cols <- names(par_fixed)[vapply(par_fixed, is.numeric, logical(1))]
  if (length(numeric_cols) == 0L) stop("Unable to locate estimate column in parFixedDf")
  numeric_cols[[1]]
}
se_col <- if ("SE" %in% names(par_fixed)) "SE" else NA_character_
rse_col <- if ("%RSE" %in% names(par_fixed)) "%RSE" else NA_character_
par_fixed_rows <- list()
for (idx in seq_len(nrow(par_fixed))) {
  row <- par_fixed[idx, , drop = FALSE]
  row_name <- rownames(par_fixed)[idx]
  name <- if ("Parameter" %in% names(row)) {
    as.character(row[["Parameter"]])
  } else if (!is.null(row_name) && nzchar(row_name)) {
    row_name
  } else {
    paste0("param_", idx)
  }
  if (length(name) == 0L || !nzchar(name)) {
    next
  }
  par_fixed_rows[[name]] <- list(
    estimate = unname(as.numeric(row[[estimate_col]])),
    se = if (!is.na(se_col)) unname(as.numeric(row[[se_col]])) else NA_real_,
    rse = if (!is.na(rse_col)) unname(as.numeric(row[[rse_col]])) else NA_real_
  )
}

obj_df <- as.data.frame(fit$objDF)
payload <- list(
  artifact_version = 1L,
  source = list(
    tool = "nlmixr2",
    version = as.character(utils::packageVersion("nlmixr2")),
    estimator = "focei",
    dataset = "theo_sd",
    model = "one.cmt"
  ),
  run = list(
    n_observations = nrow(theo_sd),
    n_ids = length(unique(theo_sd$ID)),
    columns = names(theo_sd),
    objective = unname(as.numeric(fit$objf)),
    aic = extract_scalar(obj_df, "AIC"),
    bic = extract_scalar(obj_df, "BIC"),
    log_likelihood = extract_scalar(obj_df, "Log-likelihood")
  ),
  parameters = par_fixed_rows
)

jsonlite::write_json(
  payload,
  path = "reference-theophylline-fit.json",
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null",
  digits = 12
)

cat("wrote reference-theophylline-fit.json\n")
