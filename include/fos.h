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
#include "solution.h"
#include "distribution.h"
#include "partial_solution.h"

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

class fos_t {

public:
    fos_t();

    fos_t(int element_size);

    fos_t(double **covariance_matrix);

    fos_t(vec_t<vec_t<double>> fitness_dependency_matrix, bool is_marginal);

    fos_t(FILE *file);

    fos_t(const std::map<int, std::set<int>> &variable_interaction_graph,
          vec_t<vec_t<double>> *fitness_dependency_matrix);

    fos_t(const fos_t &f);

    ~fos_t();

    int getLength();

    void deriveTree(double **MI_matrix);

    std::vector<int> getSet(int element_index);

    int getSetLength(int element_index);

    double getAcceptanceRate();

    double getDistributionMultiplier(int element_index);

    void addGroup(int var_index);

    void addGroup(const std::set<int> &group);

    void addGroup(std::vector<int> group);

    void addGroup(distribution_t *dist);

    void addConditionedGroup(std::vector<int> variables);

    void addConditionedGroup(std::vector<int> variables, std::set<int> conditioned_variables);

    void randomizeOrder();

    void randomizeOrder(const std::map<int, std::set<int>> &variable_interaction_graph);

    std::vector<int> getVIGOrderBreadthFirst(const std::map<int, std::set<int>> &variable_interaction_graph);

    double getSimilarity(int a, int b, int *mpm_num_ind);

    double **computeMIMatrix(double **covariance_matrix, int n);

    int *hungarianAlgorithm(int **similarity_matrix, int dim);

    void hungarianAlgorithmAddToTree(int x, int prevx, short *S, int *prev, int *slack, int *slackx, int *lx, int *ly,
                                     int **similarity_matrix, int dim);

    int determineNearestNeighbour(int index, int *mpm_num_ind, int mpm_length);

    partial_solution_t *generatePartialSolution(int FOS_index, solution_t *solution_conditioned_on);

    void estimateDistributions(solution_t **selection, int selection_size,
                               vec_t<vec_t<double>> fitness_dependency_matrix);

    void estimateDistribution(int FOS_index, solution_t **selection, int selection_size,
                              vec_t<vec_t<double>> fitness_dependency_matrix);

    void adaptDistributionMultiplier(int FOS_index, partial_solution_t **solutions, int num_solutions);

    std::vector<distribution_t *> distributions;
    int no_improvement_stretch = 0;
    int maximum_no_improvement_stretch = 100;

    double p_accept = 0.0;
    std::vector <std::vector<int>> sets;
    std::vector <uvec> variables_conditioned_on;

    void print();

    uvec order;
    int *next_variable_to_sample = NULL;

    double **S_matrix;
    double *S_vector;                             /* Avoids quadratic memory requirements when a linkage tree is learned based on a random distance measure. */
};

/*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
extern int max_clique_size;
extern bool include_cliques_as_fos_elements;
extern bool include_full_fos_element;
extern bool learn_conditional_linkage_tree;
extern int seed_cliques_per_variable;
extern bool use_conditional_sampling;
extern int FOS_element_ub,                       /* Cut-off value for bounded fixed linkage tree (BFLT). */
prune_linkage_tree,
        learn_linkage_tree,                   /* Whether the FOS is learned at the start of each generation. */
static_linkage_tree,                  /* Whether the FOS is fixed throughout optimization. */
random_linkage_tree,                  /* Whether the fixed linkage tree is learned based on a random distance measure. */
FOS_element_size;                     /* If positive, the size of blocks of consecutive variables in the FOS. If negative, determines specific kind of linkage tree FOS. */
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
