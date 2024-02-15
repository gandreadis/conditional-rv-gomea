/**
 *
 * Fitness-based Conditional Real-Valued Gene-pool Optimal Mixing Evolutionary Algorithm
 *
 * Copyright (c) 2024 by Georgios Andreadis, Tanja Alderliesten, Peter A.N. Bosman, Anton Bouter, and Chantal Olieman
 * This code is licensed under CC BY-NC-ND 4.0. A copy of the license is included in the LICENSE file.
 *
 * If you use this software for any purpose, please cite the most recent pre-print titled:
 * "Fitness-based Linkage Learning and Maximum-Clique Conditional Linkage Modelling for Gray-box Optimization
 *  with RV-GOMEA", by Georgios Andreadis, Tanja Alderliesten, and Peter A.N. Bosman. 2024.
 *
 * IN NO EVENT WILL THE AUTHOR OF THIS SOFTWARE BE LIABLE TO YOU FOR ANY
 * DAMAGES, INCLUDING BUT NOT LIMITED TO LOST PROFITS, LOST SAVINGS, OR OTHER
 * INCIDENTIAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR THE INABILITY
 * TO USE SUCH PROGRAM, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGES, OR FOR ANY CLAIM BY ANY OTHER PARTY. THE AUTHOR MAKES NO
 * REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE
 * AUTHOR SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY ANYONE AS A RESULT OF
 * USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
 *
 */

#pragma once

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "tools.h"
#include "optimization.h"
#include "partial_solution.h"
#include "solution.h"

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

class distribution_t {
public:
    virtual ~distribution_t();

    // Parameter settings and default values
    double st_dev_ratio_threshold = 1.0;
    double distribution_multiplier_decrease = 0.9;
    double distribution_multiplier_increase = 1.0 / 0.9; // 1.0/c_dec

    // Variables
    double distribution_multiplier = 1.0;
    int samples_drawn = 0;
    int out_of_bounds_draws = 0;

    std::vector<int> variables;

    void adaptDistributionMultiplier(partial_solution_t **partial_solutions, int num_solutions);

    void adaptDistributionMultiplierMaximumStretch(partial_solution_t **partial_solutions, int num_solutions);

    virtual short
    generationalImprovementForOnePopulationForFOSElement(partial_solution_t **partial_solutions, int num_solutions,
                                                         double *st_dev_ratio) = 0;

    double estimateMean(int var, solution_t **selection, int selection_size);

    double estimateCovariance(int vara, int varb, solution_t **selection, int selection_size);

    vec estimateMeanVectorML(std::vector<int> variables, solution_t **selection, int selection_size);

    mat estimateRegularCovarianceMatrixML(std::vector<int> variables, vec mean_vector, solution_t **selection,
                                          int selection_size);

    mat estimateCovarianceMatrixML(std::vector<int> variables, solution_t **selection, int selection_size);

    mat estimateUnivariateCovarianceMatrixML(std::vector<int> variables, solution_t **selection, int selection_size);

    bool regularizeCovarianceMatrix(mat &cov_mat, vec &mean_vector, solution_t **selection, int selection_size);

    mat pseudoInverse(const mat &matrix);

    mat choleskyDecomposition(const mat &matrix);

    int linpackDCHDC(double a[], int lda, int p, double work[], int ipvt[]);

    void blasDSCAL(int n, double sa, double x[], int incx);

    int blasDAXPY(int n, double da, double *dx, int incx, double *dy, int incy);

    int blasDSWAP(int n, double *dx, int incx, double *dy, int incy);

    virtual void updateConditionals(const std::map<int, std::set<int>> &variable_interaction_graph, int visited[]);

    virtual void setOrder(const std::vector<int> &order);

    virtual void estimateDistribution(solution_t **selection, int selection_size,
                                      vec_t<vec_t<double>> fitness_dependency_matrix) = 0;

    virtual partial_solution_t *generatePartialSolution(solution_t *parent) = 0;

    virtual void print();
};

class normal_distribution_t : public distribution_t {
public:
    normal_distribution_t(std::vector<int> variables);

    vec mean_vector;
    mat covariance_matrix;
    mat cholesky_decomposition;

    void estimateDistribution(solution_t **selection, int selection_size,
                              vec_t<vec_t<double>> fitness_dependency_matrix);

    partial_solution_t *generatePartialSolution(solution_t *parent = NULL);

    short
    generationalImprovementForOnePopulationForFOSElement(partial_solution_t **partial_solutions, int num_solutions,
                                                         double *st_dev_ratio);
};

class conditional_distribution_t : public distribution_t {
public:
    conditional_distribution_t();

    conditional_distribution_t(const std::vector<int> &variables, const std::vector<int> &conditioned_variables);

    conditional_distribution_t(const std::vector<int> &variables, const std::set<int> &conditioned_variables);

    std::vector<int> order;
    std::vector <std::vector<int>> variable_groups;
    std::vector <std::vector<int>> variables_conditioned_on;
    std::vector <std::vector<int>> index_in_var_array;

    std::vector <vec> mean_vectors;
    std::vector <vec> mean_vectors_conditioned_on;
    std::vector <mat> covariance_matrices;
    std::vector <mat> rho_matrices;
    std::vector <mat> cholesky_decompositions;

    void addGroupOfVariables(const std::vector<int> &indices, const std::set<int> &indices_cond);

    void addGroupOfVariables(std::vector<int> indices, std::vector<int> indices_cond);

    void estimateDistribution(solution_t **selection, int selection_size,
                              vec_t<vec_t<double>> fitness_dependency_matrix);

    void setOrder(const std::vector<int> &order);

    void updateConditionals(const std::map<int, std::set<int>> &variable_interaction_graph, int visited[]);

    partial_solution_t *generatePartialSolution(solution_t *solution_conditioned_on = NULL);

    short
    generationalImprovementForOnePopulationForFOSElement(partial_solution_t **partial_solutions, int num_solutions,
                                                         double *st_dev_ratio);

private:
    void initializeMemory();

    void estimateConditionalGaussianML(int variable_group_index, solution_t **selection, int selection_size,
                                       vec_t<vec_t<double>> fitness_dependency_matrix);

    void print();
};

