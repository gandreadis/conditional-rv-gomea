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
#include "fitness.h"
#include "fos.h"
#include "solution.h"

#include <deque>

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

class population_t {
public:
    /*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
    population_t(fitness_t *fitness, int population_size, double lower_init, double upper_init);

    ~population_t();

    void runGeneration();

    void makeSelection();

    void updateElitist();

    void computeRanks();

    void makeSelectionUsingDiversityOnRank0();

    void estimateDistribution(int FOS_index);

    void estimateDistribution();

    double estimateMean(int var);

    void updateAMSMeans();

    void copyBestSolutionsToPopulation();

    void getBestInPopulation(int *individual_index);

    void evaluateCompletePopulation();

    void generateAndEvaluateNewSolutions();

    void insertImprovement(solution_t *solution, partial_solution_t *part);

    short checkForImprovement(solution_t *solution, partial_solution_t *part);

    void applyPartialAMS(partial_solution_t *solution, double cmul);

    short applyAMS(int individual_index);

    void applyForcedImprovements(int individual_index, int donor_index);

    double getFitnessMean();

    double getFitnessVariance();

    void initializeDefaultParameters();

    void initializeNewPopulationMemory();

    void initializeFOS();

    void initializeParameterRangeBounds(double lower_user_range, double upper_user_range);

    void initializePopulationAndFitnessValues();

    std::map<int, std::set<int>> buildVariableInteractionGraphBasedOnFitnessDependencies();

    void initializeFitnessDependencyMatrix();

    void updateFitnessDependencyMatrix();

    int computeFitnessDependency(int k, solution_t *individual_to_compare);

    void printFitnessDependencyMatrix();

    void initializeFitnessDependencyMonitoringFile();

    void writeFitnessDependencyMonitoringToFile();

    void computeMinMaxBoundsOfCurrentPopulation(double *min, double *max);

    /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

    /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
    double eta_cov,
            tau,
            st_dev_ratio_threshold,
            distribution_multiplier_increase,
            distribution_multiplier_decrease,
            *lower_init_ranges,
            *upper_init_ranges;
    int maximum_no_improvement_stretch,
            num_elitists_to_copy = 1;
    double delta_AMS;
    /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
    partial_solution_t ***sampled_solutions;
    int number_of_generations,
            population_size,                                /* The size of the first population in the multi-start scheme. */
    selection_size,                                     /* The size of the selection for each population. */
    *individual_NIS;                                      /* The number of generations a solution has not improved. */
    solution_t **individuals,
            **selection;                                          /* Selected solutions, one for each population. */
    fitness_t *fitness;
    double *ranks,                                               /* Ranks of population members. */
    objective_value_elitist,                         /* Objective values of selected solutions. */
    constraint_value_elitist,                        /* Sum of all constraint violations of selected solutions. */
    *mean_shift_vector,                                   /* The mean vectors of the previous generation, one for each population. */
    *prev_mean_vector;                                   /* The mean vectors of the previous generation, one for each population. */
    short population_terminated;
    fos_t *linkage_model = NULL;

    // Fitness dependencies
    vec_t<vec_t<double>> fitness_dependency_matrix;
    mat fitness_dependency_pairs;
    solution_t *first_individual_for_fitness_comparison = NULL;
    solution_t *second_individual_for_fitness_comparison = NULL;
    double *fitnesses_of_first_individual_variants = NULL;

    int minimal_fitness_dependencies_per_iteration = 2;
    int number_of_fitness_dependency_pairs = -1;
    int number_of_checked_fitness_dependency_pairs = -1;
    int fitness_dependency_pairs_to_check_per_iteration = -1;
    int total_fitness_dependencies_found = -1;
    int fitness_dependency_check_iteration = -1;
    int current_fitness_dependency_waiting_position = -1;
    int number_of_fitness_dependency_waiting_cycles = -1;
    /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
};

