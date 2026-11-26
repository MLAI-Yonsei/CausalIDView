parse_backend_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  flags <- c("--mode", "--train", "--predict", "--config", "--output", "--artifact-dir")
  result <- list()
  for (flag in flags) {
    position <- match(flag, args)
    if (is.na(position) || position == length(args)) stop(paste("missing argument", flag))
    result[[substring(flag, 3)]] <- args[[position + 1]]
  }
  result
}

load_backend_job <- function() {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("R package 'jsonlite' is required by the CausalArena R protocol")
  }
  args <- parse_backend_args()
  list(
    args = args,
    train = read.csv(args$train, check.names = FALSE),
    predict = read.csv(args$predict, check.names = FALSE),
    config = jsonlite::fromJSON(args$config, simplifyVector = TRUE)
  )
}

design_matrix <- function(table) {
  columns <- grep("^x[0-9]+$", names(table), value = TRUE)
  if (length(columns) == 0) stop("input table has no x columns")
  matrix(as.numeric(as.matrix(table[, columns, drop = FALSE])),
         nrow = nrow(table), ncol = length(columns),
         dimnames = list(NULL, columns))
}

require_locked_package <- function(package, source) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(sprintf("required R package '%s' is not installed in R_LIBS_USER", package))
  }
  description <- utils::packageDescription(package)
  runtime_version <- as.character(utils::packageVersion(package))
  locked_version <- as.character(source$package_version)
  version_matches <- identical(runtime_version, locked_version) ||
    startsWith(locked_version, paste0(runtime_version, "+git."))
  if (!version_matches) {
    stop(sprintf(
      "R package '%s' version mismatch: runtime=%s locked=%s",
      package, runtime_version, locked_version
    ))
  }
  remote_sha <- if (is.null(description$RemoteSha)) NA_character_ else description$RemoteSha
  locked_commit <- as.character(source$commit_sha)
  if (grepl("^[0-9a-f]{40}$", locked_commit) &&
      (is.na(remote_sha) || !identical(remote_sha, locked_commit))) {
    stop(sprintf(
      "R package '%s' source commit mismatch: runtime=%s locked=%s",
      package, remote_sha, locked_commit
    ))
  }
  runtime <- list(
    package = package,
    version = runtime_version,
    remote_sha = remote_sha,
    locked_commit = locked_commit,
    locked_version = locked_version
  )
  runtime
}

write_runtime_provenance <- function(runtime, artifact_dir) {
  jsonlite::write_json(runtime, file.path(artifact_dir, "runtime_provenance.json"),
                       auto_unbox = TRUE, pretty = TRUE, null = "null")
}

write_cate_output <- function(path, row_ids, draws = NULL, cate = NULL,
                              alpha = 0.05, mu0 = NULL, mu1 = NULL) {
  if (!is.null(draws)) {
    draws <- as.matrix(draws)
    cate <- colMeans(draws)
    lower <- apply(draws, 2, stats::quantile, probs = alpha / 2, names = FALSE)
    upper <- apply(draws, 2, stats::quantile, probs = 1 - alpha / 2, names = FALSE)
  } else {
    lower <- upper <- NULL
  }
  output <- data.frame(`_row_id` = row_ids, cate = as.numeric(cate), check.names = FALSE)
  if (!is.null(mu0)) output$mu0 <- as.numeric(mu0)
  if (!is.null(mu1)) output$mu1 <- as.numeric(mu1)
  if (!is.null(lower)) output$cate_lower <- as.numeric(lower)
  if (!is.null(upper)) output$cate_upper <- as.numeric(upper)
  write.csv(output, path, row.names = FALSE, quote = FALSE)
}

artifact_fit_path <- function(job) file.path(job$args$`artifact-dir`, "fit.rds")

interval_alpha <- function(config) {
  if (is.null(config$interval_alpha)) 0.05 else as.numeric(config$interval_alpha)
}
