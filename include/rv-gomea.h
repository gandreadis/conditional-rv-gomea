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
#include "population.h"
#include "tools.h"
#include "fitness.h"
#include "optimization.h"
#include "fos.h"

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

class rvg_t {
public:
    /*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
    rvg_t(int argc, char **argv);

    ~rvg_t();

    void run(void);

    void runAllPopulations();

    void generationalStepAllPopulations();

    void generationalStepAllPopulationsRecursiveFold(int population_index_smallest, int population_index_biggest);

    void parseCommandLine(int argc, char **argv);

    void parseOptions(int argc, char **argv, int *index);

    void parseFOSElementSize(int *index, int argc, char **argv);

    void parseSeed(int *index, int argc, char **argv);

    void optionError(char **argv, int index);

    void parseParameters(int argc, char **argv, int *index);

    void printUsage(void);

    void checkOptions(void);

    void printVerboseOverview(void);

    void initialize(void);

    void initializeNewPopulation(void);

    void initializeProblem(int problem_index, double vtr);

    void restartLargestPopulation();

    void writeGenerationalStatisticsForOnePopulation(int population_index);

    void writeGenerationalSolutions(short final);

    void writeGenerationalSolutionsBest(short final);

    short checkTerminationCondition(void);

    short checkSubgenerationTerminationConditions();

    short checkPopulationTerminationConditions(int population_index);

    short checkTimeLimitTerminationCondition(void);

    short checkNumberOfEvaluationsTerminationCondition(void);

    short checkVTRTerminationCondition(void);

    void checkAverageFitnessTerminationConditions(void);

    void determineBestSolutionInCurrentPopulations(int *population_of_best, int *index_of_best);

    short checkFitnessVarianceTermination(int population_index);

    short checkDistributionMultiplierTerminationCondition(int population_index);
    /*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

    /*-=-=-=-=-=-=-=-=-=-=-=- Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
    std::vector<population_t *> populations;
    fitness_t *fitness;
    int total_number_of_writes = 0.0;                              /* Total number of times a statistics file has been written. */
    int total_number_of_generations = 0;

    /*-=-=-=-=-=-=-=-=-=-=-=- Options -=-=-=-=-=-=-=-=-=-=-=-=-*/
    short print_verbose_overview,                              /* Whether to print a overview of settings (0 = no). */
    use_guidelines,                                      /* Whether to override parameters with guidelines (for those that exist). */
    fix_seed = 0,                                            /* Whether a fixed seed is used. */
    use_vtr,
            black_box_evaluations,                         /* Whether full (black-box) evaluations must always be performed. */
    write_generational_statistics,                 /* Whether to compute and write statistics every generation (0 = no). */
    write_generational_solutions;                  /* Whether to write the population every generation (0 = no). */
    int base_population_size,                                /* The size of the first population in the multi-start scheme. */
    maximum_number_of_populations,                       /* The maximum number of populations in the multi-start scheme. */
    number_of_subgenerations_per_population_factor,      /* The subgeneration factor in the multi-start scheme. */
    maximum_no_improvement_stretch,                      /* The maximum number of subsequent generations without an improvement while the distribution multiplier is <= 1.0. */
    use_conditional_sampling;
    double maximum_number_of_evaluations,                       /* The maximum number of evaluations. */
    maximum_number_of_seconds,                           /* The maximum number of seconds. */
    tau,                                                 /* The selection truncation percentile (in [1/population_size,1]). */
    distribution_multiplier_increase,                    /* The multiplicative distribution multiplier increase. */
    distribution_multiplier_decrease,                    /* The multiplicative distribution multiplier decrease. */
    st_dev_ratio_threshold,                              /* The maximum ratio of the distance of the average improvement to the mean compared to the distance of one standard deviation before triggering AVS (SDR mechanism). */
    fitness_variance_tolerance,                          /* The minimum fitness variance level that is allowed. */
    eta_ams = 1.0,
            eta_cov = 1.0;
    double vtr,                                           /* The value-to-reach (function value of best solution that is feasible). */
    rotation_angle,                                /* The angle of rotation to be applied to the problem. */
    lower_user_range,                              /* The initial lower range-bound indicated by the user (same for all dimensions). */
    upper_user_range;                              /* The initial upper range-bound indicated by the user (same for all dimensions). */
    short use_FOS_parallelization = 1;
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
};
