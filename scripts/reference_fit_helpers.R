suppressPackageStartupMessages({
  library(jsonlite)
})

extract_scalar <- function(frame, column) {
  if (is.null(frame) || !column %in% names(frame)) {
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

extract_named_value <- function(row, candidates, fallback = NULL) {
  for (candidate in candidates) {
    if (candidate %in% names(row)) {
      value <- suppressWarnings(as.numeric(row[[candidate]]))
      if (!is.na(value)) {
        return(unname(value))
      }
    }
  }
  fallback
}

extract_parameter_name <- function(row, idx) {
  if ("Parameter" %in% names(row)) {
    value <- as.character(row[["Parameter"]][[1]])
    if (!is.na(value) && nzchar(value)) {
      return(value)
    }
  }
  row_name <- rownames(row)
  if (length(row_name) > 0L && nzchar(row_name[[1]])) {
    return(row_name[[1]])
  }
  paste0("param_", idx)
}

extract_parameters <- function(fit) {
  par_fixed <- as.data.frame(fit$parFixedDf)
  rows <- list()
  for (idx in seq_len(nrow(par_fixed))) {
    row <- par_fixed[idx, , drop = FALSE]
    name <- extract_parameter_name(row, idx)
    rows[[name]] <- list(
      estimate = extract_named_value(row, c("Est.", "Estimate")),
      se = extract_named_value(row, c("SE")),
      rse = extract_named_value(row, c("%RSE"))
    )
  }
  rows
}

write_reference_artifact <- function(
  fit,
  data,
  estimator,
  dataset_name,
  model_name,
  output_path
) {
  obj_df <- as.data.frame(fit$objDF)
  id_column <- names(data)[tolower(names(data)) == "id"]
  payload <- list(
    artifact_version = 1L,
    source = list(
      tool = "nlmixr2",
      version = as.character(utils::packageVersion("nlmixr2")),
      estimator = estimator,
      dataset = dataset_name,
      model = model_name
    ),
    run = list(
      n_observations = nrow(data),
      n_ids = if (length(id_column) > 0L) length(unique(data[[id_column[[1]]]])) else NULL,
      columns = names(data),
      objective = unname(as.numeric(fit$objf)),
      aic = extract_scalar(obj_df, "AIC"),
      bic = extract_scalar(obj_df, "BIC"),
      log_likelihood = extract_scalar(obj_df, "Log-likelihood")
    ),
    parameters = extract_parameters(fit)
  )

  jsonlite::write_json(
    payload,
    path = output_path,
    auto_unbox = TRUE,
    pretty = TRUE,
    null = "null",
    digits = 12
  )

  cat("wrote", output_path, "\n")
}
