# caretS3

```R
data(mtcars)

# Prepare data
X_reg <- mtcars[, -1]  # All except mpg
y_reg <- mtcars$mpg     # Target variable

# Split into train/test
set.seed(123)
idx_reg <- sample(nrow(X_reg), 0.7 * nrow(X_reg))
X_train <- X_reg[idx_reg, ]
y_train <- y_reg[idx_reg]
X_test <- X_reg[-idx_reg, ]
y_test <- y_reg[-idx_reg]

mod <- caretS3::caretS3(X_train, y_train, method = "glmnet")
(pred <- predict(mod, X_test))

mod <- caretS3::caretS3(X_train, y_train, method = "rf")
(pred <- predict(mod, X_test))
```
