#include "common.h"

// overload to print vectors and vector<vector>

std::ostream &operator<<(std::ostream &out, const std::vector<double> &v)
{
    size_t last = v.size() - 1;
    for (size_t i = 0; i < v.size(); ++i)
    {
        out << v[i];
        if (i != last)
            out << ", ";
    }
    return out;
}

std::ostream &operator<<(std::ostream &out, const std::vector<bool> &v)
{
    size_t last = v.size() - 1;
    for (size_t i = 0; i < v.size(); ++i)
    {
        out << v[i];
        if (i != last)
            out << ", ";
    }
    return out;
}

std::ostream &operator<<(std::ostream &out, const std::vector<size_t> &v)
{
    size_t last = v.size() - 1;
    for (size_t i = 0; i < v.size(); ++i)
    {
        out << v[i];
        if (i != last)
            out << ", ";
    }
    return out;
}

std::ostream &operator<<(std::ostream &out, const std::vector<std::vector<double>> &v)
{
    // size_t last = v.size() - 1;
    for (size_t i = 0; i < v.size(); ++i)
    {
        out << v[i] << endl;
    }
    return out;
}

std::ostream &operator<<(std::ostream &out, const std::vector<std::vector<size_t>> &v)
{
    // size_t last = v.size() - 1;
    for (size_t i = 0; i < v.size(); ++i)
    {
        out << v[i] << endl;
    }
    return out;
}

// fast implementation of OLS, returns sum of squared error

double fastLm(const arma::vec &y, const arma::mat &X)
{
    // this function calculate sum of residual squares for OLS
    size_t n = X.n_rows;
    size_t k = X.n_cols;

    arma::colvec coef = arma::solve(X, y);
    arma::colvec resid = y - X * coef;

    double sig2 = arma::as_scalar(arma::trans(resid) * resid / (n - k));
    arma::colvec stderrest =
        arma::sqrt(sig2 * arma::diagvec(arma::inv(arma::trans(X) * X)));

    arma::colvec temp = arma::pow(resid, 2);

    double output = arma::accu(temp);
    return output;
}

// fast implementation of OLS, returns sum of squared error

double fastLm(const arma::mat &y, const arma::mat &X)
{
    // this function calculate sum of residual squares for OLS
    size_t n = X.n_rows;
    size_t k = X.n_cols;

    arma::colvec coef = arma::solve(X, y);
    arma::colvec resid = y - X * coef;

    double sig2 = arma::as_scalar(arma::trans(resid) * resid / (n - k));
    arma::colvec stderrest =
        arma::sqrt(sig2 * arma::diagvec(arma::inv(arma::trans(X) * X)));

    arma::colvec temp = arma::pow(resid, 2);

    double output = arma::accu(temp);
    return output;
}

// fast implementation of OLS, returns out of sample sum of squared error

double fastLm_OOS(const arma::mat &y_train, const arma::mat &X_train, const arma::mat &y_test, const arma::mat &X_test)
{
    // this function calculate out of sample sum of residual squares for OLS

    // arma::colvec coef = arma::solve(X_train, y_train);
    arma::mat coef = inv(X_train.t() * X_train) * X_train.t() * y_train;

    arma::mat resid = y_test - X_test * coef;

    arma::mat temp = arma::pow(resid, 2);

    double output = arma::accu(temp);

    return output;
}

// fast implementation of weighted OLS, returns sum of squared error

double fastLm_weighted(const arma::vec &y, const arma::mat &X, const arma::vec &weight)
{
    // this function calculate sum of residual squares for OLS
    size_t n = X.n_rows;
    size_t k = X.n_cols;

    arma::colvec coef = arma::solve(X, y);
    arma::colvec resid = y - X * coef;

    double sig2 = arma::as_scalar(arma::trans(resid) * resid / (n - k));
    arma::colvec stderrest =
        arma::sqrt(sig2 * arma::diagvec(arma::inv(arma::trans(X) * X)));

    arma::colvec temp = arma::pow(resid, 2) % weight;

    double output = arma::accu(temp);

    return output;
}

bool sum(std::vector<bool> &v)
{
    bool output = false;
    for (size_t i = 0; i < v.size(); i++)
    {
        output = output + v[i];
    }
    return output;
}

double log_normal_density(arma::vec &R, arma::mat &cov)
{
    double output = 0.0;

    return output;
}

// implement Lasso estimation

double soft_c(double a, double lambda)
{
    // soft threshold
    if (a > lambda)
    {
        return (a - lambda);
    }
    else if (a < lambda)
    {
        return (a + lambda);
    }
    else
    {
        return 0.0;
    }
}

// Lasso loss function

double lasso_loss(const arma::mat &X, const arma::mat &Y, const arma::vec &beta, double lambda)
{
    size_t n = X.n_rows;
    double output = accu(square(Y - X * beta) / (2 * n)) + lambda * accu(abs(beta));
    return output;
}

// main lasso fit function, returns estimated beta

arma::vec lasso_fit_standardized(const arma::mat &X, const arma::mat &Y, double lambda, const arma::vec &beta_ini, double eps = 0.0001)
{
    // solve Lasso by coordinate descent method
    // not the closed form solution
    size_t n = X.n_rows;
    size_t p = X.n_cols;
    arma::vec beta_last = beta_ini;
    arma::vec beta_new = beta_ini;

    double loss_diff = 100.00;

    arma::vec r = Y - X * beta_ini;

    double loss_old;

    while (loss_diff >= eps)
    {
        beta_last = beta_new;

        loss_old = lasso_loss(X, Y, beta_last, lambda);

        for (size_t i = 0; i < p; i++)
        {
            beta_new(i) = soft_c(arma::as_scalar(beta_last(i) + X.col(i).t() * r / n), lambda);

            r = r + X.col(i) * (beta_last(i) - beta_new(i));
        }

        double loss_new = lasso_loss(X, Y, beta_new, lambda);
        loss_diff = loss_old - loss_new;
    }

    return beta_new;
}


void int_to_bin(size_t num, std::vector<size_t> &s)
{
    size_t p = s.size();
    // i has to be int type here, cannot be size_t
    // otherwise it overflow to large positive number, the loop cannot stop
    for (int i = (p - 1); i >= 0; i--)
    {
        if (num & 1)
        {
            s.at(i) = 1;
        }
        else
        {
            s.at(i) = 0;
        }
        num = num >> 1;
    }
    return;
}

double univariate_normal_density(double x, double mu, double sig2)
{
    // density of univariate normal distribution
    double sig = sqrt(sig2);
    double output = 1.0 / sig / sqrt(2.0 * CONSTPI) * exp(-0.5 * pow(x - mu, 2) / sig2);
    return output;
}


double normalCDF(double x) // Phi(-∞, x) aka N(x)
{
    return std::erfc(-x / std::sqrt(2.0)) / 2.0;
}
