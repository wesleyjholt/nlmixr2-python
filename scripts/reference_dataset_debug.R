suppressPackageStartupMessages({
  library(nlmixr2)
  library(nlmixr2data)
})

dump_dataset <- function(name, data) {
  cat("dataset=", name, "\n", sep = "")
  cat("class=", paste(class(data), collapse = ","), "\n", sep = "")
  cat("nrow=", nrow(data), "\n", sep = "")
  cat("names=", paste(names(data), collapse = ","), "\n", sep = "")
  for (column in c("DVID", "CMT", "EVID", "dose", "DOSE", "SS", "SD", "OCC")) {
    if (column %in% names(data)) {
      values <- unique(data[[column]])
      cat(column, "=", paste(values, collapse = ","), "\n", sep = "")
    }
  }
  print(utils::head(data))
  cat("\n")
}

dump_dataset("warfarin", warfarin)
dump_dataset("Oral_1CPT", Oral_1CPT)
