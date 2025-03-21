% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PTree.R
\name{PTree}
\alias{PTree}
\title{AP-Tree to generate stochastic discount factor}
\usage{
PTree(
  R,
  Y,
  X,
  Z,
  H,
  portfolio_weight,
  loss_weight,
  stocks,
  months,
  first_split_var,
  second_split_var,
  num_stocks,
  num_months,
  min_leaf_size,
  max_depth,
  num_iter,
  num_cutpoints,
  eta = 1,
  equal_weight = FALSE,
  no_H = FALSE,
  abs_normalize = FALSE,
  weighted_loss = FALSE,
  lambda_mean = 0,
  lambda_cov = 0,
  early_stop = FALSE,
  stop_threshold = 0.95,
  lambda_ridge = 0,
  a1=0, 
  a2=0, 
  list_K, 
  random_split = FALSE
)
}
\arguments{
\item{R}{A vector of all individual stock returns, pooled of cross section and time series. The data do not need to be balanced.}

\item{Y}{Auxiliary vector, used in the split criterion of Y ~ Z * F, for boosting.}

\item{X}{Matrix of firm characteristics, should match each row of \eqn{R}.}

\item{Z}{Matrix of macro economic variables, should match each row of \eqn{R}.}

\item{portfolio_weight}{A vector of weight (e.g., value weight or equal weight) for each individual stock return when generate the leaf basis portfolio, should match each row of \eqn{R}.}

\item{loss_weight}{A vector of weight for each individual stock return in the loss function (split criterion), should match each row of \eqn{R}.}

\item{stocks}{A vector of index of stocks, should match each row of \eqn{R}.}

\item{months}{A vector of index of months, should match each row of \eqn{R}.}

\item{first_split_var}{A vector of indicies of variables being considered as the first split, corresponding to column index in \eqn{X} (count starting from zero).}

\item{second_split_var}{A vector of indicies of variables being considered as the second split, corresponding to column index in \eqn{X} (count starting from zero).}

\item{num_stocks}{Integer, number of stocks.}

\item{num_months}{Integer, number of months.}

\item{min_leaf_size}{Integer, minimal number of stocks in the leaf node, as stopping condition.}

\item{max_depth}{Integer, maximum depth of the tree, as stopping condition.}

\item{num_iter}{Integer, maximum number of iterations of the algorithm.}

\item{num_cutpoints}{Integer, number of cutpoint candidates for each characteristics, defined by quantiles of \eqn{X} variables.}

\item{eta}{Scalar, tuning parameter. When construct SDF, w <- w * eta + (1 - w) * equalweight. For regularization to equal weight.}

\item{equal_weight}{Bool, if TRUE, equal weight portfolios.}

\item{no_H}{Bool, if TRUE, no Ht in the split criterion, otherwise it is Y ~ Z * F + H.}

\item{abs_normalize}{Bool if TRUE, normalize leaf basis portfolio weight when constructing SDF by w <- w / sum(abs(w)).}

\item{weighted_loss}{Bool, if TRUE, the loss function (split criterion) has weights for each return data, work wth loss_weight input.}

\item{lambda_mean}{Scalar, the SDF is generated by (mu + lambda_mean) * (Sigma + lambda_cov)^-1.}

\item{lambda_cov}{Scalar, see above.}

\item{early_stop}{Bool, if TRUE, allow early stop if (split criterion after the new split) >= stopthreshold * (split criterion before the new split).}

\item{stop_threshold}{Scalar, should be smaller than 1, for early stop option.}

\item{lambda_ridge}{Scalar, optional if the split criterion is ridge regression.}

\item{a1}{Scalar, regularization for tree structure, use default, not activated.}

\item{a2}{Scalar, regularization for tree structure, use default, not activated.}

\item{list_K}{Matrix, regularization for tree structure, use default, not activated.}

\item{random_split}{Bool, default FALSE, use Sharpe ratio criteria; if TRUE, use random split criteria.}
}

\description{
This is the main function of AP-Tree cross sectional split, which paritions the universe of individual stock returns to generate leaf basis portfolios and stochastic discount factors.
}
