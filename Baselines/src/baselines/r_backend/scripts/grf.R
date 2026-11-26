script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
source(file.path(script_dir, "common.R"))
job <- load_backend_job()
runtime <- require_locked_package("grf", job$config$source)
set.seed(as.integer(job$config$seed))

if (job$args$mode == "fit") {
  fit <- grf::causal_forest(
    design_matrix(job$train), job$train$`_outcome`, job$train$`_treatment`,
    num.trees = as.integer(job$config$n_estimators),
    Y.hat = NULL, W.hat = NULL,
    sample.fraction = as.numeric(job$config$sample_fraction),
    mtry = as.integer(job$config$mtry),
    min.node.size = as.integer(job$config$min_node_size),
    honesty = isTRUE(job$config$honesty),
    honesty.fraction = as.numeric(job$config$honesty_fraction),
    honesty.prune.leaves = isTRUE(job$config$honesty_prune_leaves),
    alpha = as.numeric(job$config$alpha),
    imbalance.penalty = as.numeric(job$config$imbalance_penalty),
    stabilize.splits = isTRUE(job$config$stabilize_splits),
    ci.group.size = as.integer(job$config$ci_group_size),
    tune.parameters = as.character(job$config$tune_parameters),
    compute.oob.predictions = isTRUE(job$config$compute_oob_predictions),
    num.threads = as.integer(job$config$thread_count),
    seed = as.integer(job$config$seed)
  )
  saveRDS(fit, artifact_fit_path(job))
  write_runtime_provenance(runtime, job$args$`artifact-dir`)
} else {
  fit <- readRDS(artifact_fit_path(job))
}

prediction <- stats::predict(
  fit, design_matrix(job$predict),
  estimate.variance = isTRUE(job$config$estimate_variance)
)
output <- data.frame(
  `_row_id` = job$predict$`_row_id`, cate = prediction$predictions,
  check.names = FALSE
)
if (isTRUE(job$config$estimate_variance)) {
  standard_error <- sqrt(pmax(prediction$variance.estimates, 0))
  critical <- stats::qnorm(1 - interval_alpha(job$config) / 2)
  output$cate_lower <- prediction$predictions - critical * standard_error
  output$cate_upper <- prediction$predictions + critical * standard_error
}
write.csv(output, job$args$output, row.names = FALSE, quote = FALSE)
