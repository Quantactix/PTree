#ifndef GUARD_model_h
#define GUARD_model_h
#include "state.h"

class tree;
class APTree;
class BayesTree;
class OLSTree;
class TSTree;

class Model
{
public:
    double lambda;

    Model(double lambda) { this->lambda = lambda; }

    virtual double criterion(State &state, leaf_data &data) { return 0.0; };

    virtual void update_leaf_theta(State &state, arma::umat &Xorder, tree *leaf_node) { return; };

    virtual void calculate_criterion(State &state, arma::umat &Xorder, size_t &split_var, size_t &split_point, size_t &num_obs_left, size_t &num_obs_right, tree *tree_pointer, bool &splitable) { return; };

    virtual double calculate_criterion_one_candidate(State &state, arma::umat &Xorder, size_t var, size_t ind, size_t num_obs) { return 0.0; };
};

class CARTModel : public Model
{
public:
    CARTModel(double lambda) : Model(1.0) { this->lambda = lambda; }

    double criterion(State &state, leaf_data &data);

    void update_leaf_theta(State &state, arma::umat &Xorder, tree *leaf_node);

    using Model::calculate_criterion;
    void calculate_criterion(State &state, arma::umat &Xorder, size_t &split_var, size_t &split_point, size_t &num_obs_left, size_t &num_obs_right, tree *tree_pointer);

    using Model::calculate_criterion_one_candidate;
    double calculate_criterion_one_candidate(State &stat, arma::umat &Xorder, size_t var, size_t ind, size_t num_obs);
};

class APTreeModel : public Model
{
public:
    arma::mat regressor;

    APTreeModel(double lambda) : Model(1.0) { this->lambda = lambda; }

    void calculateSecondMomentMatrix(arma::mat &R, arma::mat &second_moment_matrix);

    void check_node_splitability(State &state, std::vector<APTree *> &bottom_nodes_vec, std::vector<bool> &node_splitability);

    using Model::calculate_criterion;
    void calculate_criterion(State &state, std::vector<APTree *> &bottom_nodes_vec, std::vector<bool> &node_splitability, size_t &split_node, size_t &split_var, size_t &split_point, bool &splitable, std::vector<double> &criterion_values);

    void split_node(State &state, APTree *node, size_t split_var, size_t split_point);

    void initialize_portfolio(State &state, APTree *node);

    void initialize_regressor_matrix(State &state);

    void predict_AP(arma::mat &X, APTree &root, arma::vec &months, arma::vec &leaf_index);

    void calculate_criterion_one_variable(State &state, size_t var, std::vector<APTree *> &bottom_nodes_vec, size_t node_ind, std::vector<double> &output, arma::vec &weighted_return_all, arma::vec &cumu_weight_all, arma::vec &num_stocks_all);

    void calculate_criterion_one_variable_ft(State &state, size_t var, std::vector<APTree *> &bottom_nodes_vec, size_t node_ind, std::vector<double> &output, arma::vec &weighted_return_all, arma::vec &cumu_weight_all, arma::vec &num_stocks_all, arma::vec &proposed_ft, arma::vec &proposed_ft2, size_t &ind_cutpoint);

    void node_sufficient_stat(State &state, arma::umat &Xorder, arma::vec &weighted_return_all, arma::vec &cumu_weight_all, arma::vec &num_stocks_all);

    void calculate_factor(APTree &root, arma::vec &leaf_node_index, arma::mat &all_leaf_portfolio, arma::mat &leaf_weight, arma::mat &ft, arma::mat &ft_benchmark, State &state);

};


#endif