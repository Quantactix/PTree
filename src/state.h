#ifndef GUARD_state_h
#define GUARD_state_h
#include "common.h"

class State
{
public:
    arma::mat *X; // pointer to the charateristics matrix
    arma::vec *Y;
    arma::vec *R;         // pointer to the return vector
    arma::mat *market;    // pointer to market, no spike and slab selection
    arma::mat *R_mat;     // pointer to the return matrix, for TSTree only
    arma::mat *Z;         // placeholder
    arma::mat *F;         // for Bayes tree
    arma::mat *regressor; // for Bayes tree
    arma::mat *H;         // placeholder
    arma::vec *weight;
    arma::vec *loss_weight;
    arma::vec *stocks; // pointer to the index of stocks, same number of rows as X
    arma::vec *months; // months indicator
    arma::vec *unique_months;
    arma::vec *first_split_var;
    arma::vec *second_split_var;
    arma::vec *future_split_var;
    // the two vectors below are for the APTree model 2, first cut at Macro variable
    arma::vec *third_split_var;
    arma::vec *deep_split_var;
    arma::mat *first_split_mat; // for APTree model 2 only
    arma::mat *split_candidate_mat;
    arma::mat *pi_vec;
    std::map<size_t, size_t> *months_list; // list of UNIQUE months

    size_t num_obs_all;
    size_t num_stocks;
    size_t num_months;
    size_t first_test;       // for rolling window OLS, first month of training data (in the original label)
    size_t min_leaf_size;    // mainly used in PTree, minimal data points
    size_t min_leaf_size_CS; // for cross section splits in BayesTree, minimal number of stocks for a month
    size_t min_leaf_size_TS; // for time series splits, minimal number of months on one side
    size_t max_depth;
    size_t num_cutpoints;
    size_t num_regressors; // for Bayes tree
    size_t p;              // number of charateristics
    std::vector<double> split_candidates;
    bool equal_weight;
    bool no_H;
    bool abs_normalize;
    bool weighted_loss;
    bool range_constraint;
    double upper_limit; // constraint for weight optimization
    double lower_limit;
    double sum_constraint;
    double gamma_optimize;
    double overall_loss;
    arma::vec current_ft;  // the current factor by PTree
    arma::vec current_ft2; // the current factor of PTree + H
    double sigma;
    double tau;
    double lambda;
    double lambda_mean;
    double lambda_cov;
    double lambda_mean_factor;
    double lambda_cov_factor;
    double eta;
    double stop_threshold; // for APTree
    double lambda_ridge;   // for ridge pool regression
    bool flag_first_cut;
    bool no_X;         // remove X main effect
    bool no_Z;         // remove Z main effect, macro variables
    bool no_X_F;       // remove X * F interactions
    bool no_market;    // remove extra market term
    bool no_intercept; // remove intercept
    bool parallel;
    bool parallel_per_cutpoint;
    bool early_stop;
    bool rolling;
    // for APTree
    double a1;         // for penalized criteria
    double a2;         // for penalized criteria
    arma::mat *list_K; // for penalized criteria
    // prior parameters for the Bayes tree
    arma::mat *X_reg;
    double a;
    double b;
    double xi_normal;
    arma::vec xi_spike;
    arma::vec xi_slab;
    size_t p_normal_prior;
    size_t p_spike_slab;
    size_t p_market;
    bool discrete; // inidicating delta function spike or normal spike
    size_t X_reg_cols;
    size_t Z_cols;
    arma::mat *split_candidates_mat;   // matrix of split point candidates for all variables. row for candidate, column for variable
    std::vector<bool> *split_var_type; // vector of indicator for split variable type, 1 for cross sectional, 0 for time series
    
    // for randome split.
    bool random_split;

    // parameter for XBART tree regularization prior
    double alpha;
    double beta;
    
    // state for APTree model
    State(arma::mat &X, arma::vec &Y, arma::vec &R, arma::mat &Z, arma::mat &H, arma::vec &portfolio_weight, arma::vec &loss_weight, arma::vec &stocks, arma::vec &months, arma::vec &first_split_var, arma::vec &second_split_var, size_t &num_months, std::map<size_t, size_t> &months_list, size_t &num_stocks, size_t &min_leaf_size, size_t &max_depth, size_t &num_cutpoints, bool &equal_weight, bool &no_H, bool &abs_normalize, bool &weighted_loss, double &eta, double &lambda_mean, double &lambda_cov, double &lambda_mean_factor, double &lambda_cov_factor, bool &early_stop, double &stop_threshold, double &lambda_ridge, double &a1, double &a2, arma::mat &list_K, bool &random_split)
    {
        this->X = &X;
        this->Y = &Y;
        this->R = &R;
        this->Z = &Z;
        this->H = &H;
        this->weight = &portfolio_weight;
        this->loss_weight = &loss_weight;
        this->stocks = &stocks;
        this->months = &months;
        this->months_list = &months_list;
        this->first_split_var = &first_split_var;
        this->second_split_var = &second_split_var;
        this->third_split_var = 0;
        this->deep_split_var = 0;
        this->num_months = num_months;
        this->num_stocks = num_stocks;
        this->min_leaf_size = min_leaf_size;
        this->max_depth = max_depth;
        this->num_cutpoints = num_cutpoints;
        this->split_candidates.resize(num_cutpoints);
        this->p = X.n_cols;
        this->num_obs_all = X.n_rows;
        this->equal_weight = equal_weight;
        this->no_H = no_H;
        this->abs_normalize = abs_normalize;
        this->weighted_loss = weighted_loss;
        this->overall_loss = std::numeric_limits<double>::max(); // start from the largest criterion value
        this->random_split = random_split;

        // Initialize the current_ft as zeros
        this->current_ft = arma::vec(H.n_rows, arma::fill::zeros);

        // Initialize the current_ft2 as the MVE of H + current_ft (zeros)

        arma::mat Hmu;
        arma::mat Hsigma;
        arma::mat Hweight;
        arma::mat Hft;
        double Hweight_sum;

        if (H.n_cols == 1)
        {
            this->current_ft2 = H * 1.0;
        }
        else
        {
            Hmu = arma::mean(H, 0);
            Hmu = arma::trans(Hmu);
            Hsigma = arma::cov(H);
            Hweight = arma::inv(Hsigma + lambda_cov_factor * arma::eye(H.n_cols, H.n_cols)) * (Hmu + lambda_mean_factor * arma::ones(Hmu.n_rows, Hmu.n_cols));
            arma::vec Hequal_weight(H.n_cols);
            Hequal_weight.fill(1.0 / H.n_cols);
            Hweight = Hweight * eta + (1.0 - eta) * Hequal_weight;

            if (abs_normalize)
            {
                Hweight_sum = arma::accu(arma::abs(Hweight));
            }
            else
            {
                Hweight_sum = arma::accu((Hweight));
            }
            Hweight = Hweight / Hweight_sum;

            // mean variance efficient portfolio
            this->current_ft2 = H * Hweight;
        }

        // double initial_sharpe = arma::mean(arma::mean(this->current_ft2)) / arma::mean(arma::stddev(this->current_ft2));

        this->sigma = 0.0;
        this->tau = 0.0;
        this->lambda = 0.0;
        this->eta = eta;
        this->first_split_mat = 0;
        this->num_regressors = 0;
        this->lambda_mean = lambda_mean;
        this->lambda_cov = lambda_cov;
        this->lambda_mean_factor = lambda_mean_factor;
        this->lambda_cov_factor = lambda_cov_factor;
        this->stop_threshold = stop_threshold;
        this->early_stop = early_stop;
        this->lambda_ridge = lambda_ridge;
        this->a1 = a1;
        this->a2 = a2;

        this->list_K = &list_K;

        for (size_t i = 0; i < num_cutpoints; i++)
        {
            split_candidates[i] = 2.0 / (num_cutpoints + 1) * (i + 1) - 1;
        }

        cout << "The split value candidates are " << split_candidates << endl;
    }

};

#endif