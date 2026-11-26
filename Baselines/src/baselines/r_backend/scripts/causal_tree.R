script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
source(file.path(script_dir, "common.R"))
job <- load_backend_job()
runtime <- require_locked_package("causalTree", job$config$source)
suppressPackageStartupMessages(library(rpart))
set.seed(as.integer(job$config$seed))

if (job$args$mode == "fit") {
  X <- design_matrix(job$train)
  data <- data.frame(y = job$train$`_outcome`, X, check.names = FALSE)
  formula <- stats::as.formula(paste("y ~", paste(colnames(X), collapse = " + ")))
  tree <- causalTree::causalTree(
    formula, data = data, treatment = job$train$`_treatment`,
    split.Rule = job$config$split_rule, cv.option = job$config$cv_option,
    split.Honest = FALSE, cv.Honest = FALSE,
    xval = job$config$cross_validation_folds, cp = job$config$cp,
    minsize = job$config$minsize, propensity = mean(job$train$`_treatment`)
  )
  selected_cp <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]
  fit <- rpart::prune(tree, cp = selected_cp)
  saveRDS(fit, artifact_fit_path(job))
  write_runtime_provenance(runtime, job$args$`artifact-dir`)
} else {
  fit <- readRDS(artifact_fit_path(job))
}

prediction <- as.numeric(stats::predict(fit, newdata = data.frame(design_matrix(job$predict))))
write_cate_output(job$args$output, job$predict$`_row_id`, cate = prediction)
