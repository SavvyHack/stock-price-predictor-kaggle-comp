### Read in the data
# This assumes that train.csv and test.csv are in the current working directory
train <- read.csv("train.csv")
test <- read.csv("test.csv")

### Create a hold-out set
# 80-20 split
holdout.ID <- sample(train$ID, 0.2 * nrow(train))
# Holdout set
holdout <- train[holdout.ID, ]
# Remaining training data
train <- train[-holdout.ID, ]

### Train a model on the training data
# This is a logistic regression on the raw training data
# It is intentionally not meant to be an example of a good model
response <- train$p40 > train$p50
mod <- glm(response ~ . - ID, family = "binomial", data = train[, names(train) != "p50"])

### Make predictions on the hold-out set
holdout.sell <- as.integer(predict(mod, newdata = holdout, type = "response") > 0.5)

### Evaluate the predictions on the hold-out set
# To simulate the constraint, we can require that no more than 40% of the hold-out
# stocks can be sold
# The above predictions do not explicitly account for this constraint, so we may
# need to randomly subsample
score <- function(holdout, holdout.sell) {
  # Check the constraint
  K <- 0.4 * nrow(holdout)
  n.sell <- sum(holdout.sell)
  if (n.sell > K) {
    # It has been violated, randomly subsample
    cat("Subsampling", n.sell, "stocks down to", K, "\n")
    ones <- which(holdout.sell == 1)
    keep <- sample(ones, K)
    holdout.sell <- holdout.sell * 0
    holdout.sell[keep] <- 1
  }
  
  R <- sum(holdout.sell * (holdout$p40 - holdout$p50))
  R
}

score(holdout, holdout.sell)

### Make predictions on the test set
test.sell <- as.integer(predict(mod, newdata = test, type = "response") > 0.5)
# Note again that these predictions will not necessarily follow the constraint
# The code below checks if we are following the constraint
n.sell <- sum(test.sell); n.sell
n.sell <= 4000

### Create submission file
submission.df <- as.data.frame(cbind(ID=test$ID, sell=test.sell))
write.csv(submission.df, "skeleton.csv", row.names = FALSE)
# The created file can be submitted to Kaggle



