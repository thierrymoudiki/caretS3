#' Fit a caret model with fixed hyperparameters (no resampling)
#'
#' @param x Matrix or data frame of predictors.
#' @param y Response vector (numeric for regression, factor for classification).
#' @param method Any model name accepted by \code{caret::train} (e.g. "glmnet", "rf", "svmRadial").
#' @param ... Hyperparameters for \code{method}, passed as a one-row \code{tuneGrid}
#'   (e.g. \code{alpha = 0.5, lambda = 0.1} for "glmnet").
#' @param train_args Named list of extra arguments for \code{caret::train}
#'   (e.g. \code{list(preProcess = c("center", "scale"))}).
#' @return An object of class \code{"caretS3"}.
#' @export
#'
#' @examples
#'
#' data(mtcars)
#'
#' # Prepare data
#' X_reg <- mtcars[, -1]  # All except mpg
#' y_reg <- mtcars$mpg     # Target variable
#'
#' # Split into train/test
#' set.seed(123)
#' idx_reg <- sample(nrow(X_reg), 0.7 * nrow(X_reg))
#' X_train <- X_reg[idx_reg, ]
#' y_train <- y_reg[idx_reg]
#' X_test <- X_reg[-idx_reg, ]
#' y_test <- y_reg[-idx_reg]
#'
#' mod <- caretS3::caretS3(X_train, y_train, method = "glmnet")
#' (pred <- predict(mod, X_test))
#'
#' mod <- caretS3::caretS3(X_train, y_train, method = "rf")
#' (pred <- predict(mod, X_test))
#'
caretS3 <- function(x, y, method = "glmnet", ..., train_args = list())
{
  x <- as.data.frame(x)
  if (is.null(colnames(x)) || any(colnames(x) == ""))
    colnames(x) <- paste0("X", seq_len(ncol(x)))
  x_names <- colnames(x)

  hyper <- list(...)
  tune_grid <- if (length(hyper)) data.frame(hyper, stringsAsFactors = FALSE) else NULL

  args <- c(list(x = x, y = y,
                 method = method,
                 trControl = caret::trainControl(method = "none")),
            if (!is.null(tune_grid)) list(tuneGrid = tune_grid),
            train_args)
  fit <- do.call(caret::train, args)

  structure(list(fit = fit,
                 method = method,
                 hyperparameters = hyper,
                 x_names = x_names,
                 type = if (is.factor(y)) "classification" else "regression",
                 levels = if (is.factor(y)) levels(y) else NULL,
                 call = match.call()),
            class = "caretS3")
}

#' Predict from a caretS3 model
#'
#' @param object A \code{caretS3} object.
#' @param newx Matrix or data frame of new predictors, same columns as training.
#' @param type "raw" (predictions/classes) or "prob" (class probabilities).
#' @param ... Unused.
#' @export
predict.caretS3 <- function(object, newx, type = c("raw", "prob"), ...)
{
  type <- match.arg(type)
  newx <- as.data.frame(newx)
  if (ncol(newx) != length(object$x_names))
    stop("`newx` has ", ncol(newx), " columns; the model expects ",
         length(object$x_names), ".", call. = FALSE)
  if (is.null(colnames(newx)) || !all(object$x_names %in% colnames(newx)))
    colnames(newx) <- object$x_names   # positional match, as in the original code
  newx <- newx[, object$x_names, drop = FALSE]
  caret::predict.train(object$fit, newdata = newx, type = type)
}

#' @export
print.caretS3 <- function(x, ...)
{
  cat("caretS3 model\n")
  cat("  method:     ", x$method, "\n", sep = "")
  cat("  task:       ", x$type, "\n", sep = "")
  cat("  predictors: ", length(x$x_names), "\n", sep = "")
  if (length(x$hyperparameters)) {
    hp <- paste(names(x$hyperparameters), unlist(x$hyperparameters),
                sep = " = ", collapse = ", ")
    cat("  tuning:     ", hp, "\n", sep = "")
  }
  invisible(x)
}

#' @export
coef.caretS3 <- function(object, ...)
{
  fm <- object$fit$finalModel
  if (object$method == "glmnet")
    return(stats::coef(fm, s = object$fit$bestTune$lambda))
  stats::coef(fm, ...)
}
