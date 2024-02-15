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

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "population.h"

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

population_t::population_t(fitness_t *fitness, int population_size, double lower_init, double upper_init) {
    this->population_size = population_size;
    this->fitness = fitness;

    initializeDefaultParameters();

    initializeParameterRangeBounds(lower_init, upper_init);

    initializeNewPopulationMemory();

    initializePopulationAndFitnessValues();


}

population_t::~population_t() {
    for (int j = 0; j < population_size; j++)
        delete (individuals[j]);
    free(individuals);

    free(selection);
    free(ranks);

    free(mean_shift_vector);
    free(prev_mean_vector);

    free(individual_NIS);

    free(lower_init_ranges);
    free(upper_init_ranges);

    for (int j = 0; j < linkage_model->getLength(); j++)
        free(sampled_solutions[j]);
    free(sampled_solutions);

    if (similarity_measure == 'F') {
        free(fitnesses_of_first_individual_variants);
        delete first_individual_for_fitness_comparison;
        delete second_individual_for_fitness_comparison;
    }

    delete linkage_model;
}

void population_t::runGeneration() {
    if (population_terminated)
        return;

    makeSelection();

    updateElitist();

    estimateDistribution();

    copyBestSolutionsToPopulation();

    generateAndEvaluateNewSolutions();

    number_of_generations++;
}

void population_t::updateElitist() {
    solution_t *best_so_far = individuals[0];
    for (int i = 1; i < population_size; i++) {
        if (fitness->betterFitness(individuals[i], best_so_far))
            best_so_far = individuals[i];
    }
    objective_value_elitist = best_so_far->objective_value;
    constraint_value_elitist = best_so_far->constraint_value;
}

void population_t::makeSelection() {
    computeRanks();
    int *sorted = mergeSort(ranks, population_size);

    if (ranks[sorted[selection_size - 1]] == 0)
        makeSelectionUsingDiversityOnRank0();
    else {
        for (int i = 0; i < selection_size; i++)
            selection[i] = individuals[sorted[i]];
    }

    free(sorted);
}

void population_t::makeSelectionUsingDiversityOnRank0() {
    int number_of_rank0_solutions = 0;
    for (int i = 0; i < population_size; i++) {
        if (ranks[i] == 0)
            number_of_rank0_solutions++;
    }

    int *preselection_indices = (int *) Malloc(number_of_rank0_solutions * sizeof(int));
    int k = 0;
    for (int i = 0; i < population_size; i++) {
        if (ranks[i] == 0) {
            preselection_indices[k] = i;
            k++;
        }
    }

    int index_of_farthest = 0;
    double distance_of_farthest = individuals[preselection_indices[0]]->objective_value;
    for (int i = 1; i < number_of_rank0_solutions; i++) {
        if (individuals[preselection_indices[i]]->objective_value > distance_of_farthest) {
            index_of_farthest = i;
            distance_of_farthest = individuals[preselection_indices[i]]->objective_value;
        }
    }

    int number_selected_so_far = 0;
    int *selection_indices = (int *) Malloc(selection_size * sizeof(int));
    selection_indices[number_selected_so_far] = preselection_indices[index_of_farthest];
    preselection_indices[index_of_farthest] = preselection_indices[number_of_rank0_solutions - 1];
    number_of_rank0_solutions--;
    number_selected_so_far++;

    double *nn_distances = (double *) Malloc(number_of_rank0_solutions * sizeof(double));
    for (int i = 0; i < number_of_rank0_solutions; i++)
        nn_distances[i] = distanceEuclidean(individuals[preselection_indices[i]]->variables,
                                            individuals[selection_indices[number_selected_so_far - 1]]->variables);

    while (number_selected_so_far < selection_size) {
        index_of_farthest = 0;
        distance_of_farthest = nn_distances[0];
        for (int i = 1; i < number_of_rank0_solutions; i++) {
            if (nn_distances[i] > distance_of_farthest) {
                index_of_farthest = i;
                distance_of_farthest = nn_distances[i];
            }
        }

        selection_indices[number_selected_so_far] = preselection_indices[index_of_farthest];
        preselection_indices[index_of_farthest] = preselection_indices[number_of_rank0_solutions - 1];
        nn_distances[index_of_farthest] = nn_distances[number_of_rank0_solutions - 1];
        number_of_rank0_solutions--;
        number_selected_so_far++;

        for (int i = 0; i < number_of_rank0_solutions; i++) {
            double value = distanceEuclidean(individuals[preselection_indices[i]]->variables,
                                             individuals[selection_indices[number_selected_so_far - 1]]->variables);
            if (value < nn_distances[i])
                nn_distances[i] = value;
        }
    }

    for (int i = 0; i < selection_size; i++)
        selection[i] = individuals[selection_indices[i]];

    free(nn_distances);
    free(selection_indices);
    free(preselection_indices);
}

void population_t::computeRanks() {
    std::vector<int> sorted(population_size);
    for (int i = 0; i < population_size; i++)
        sorted[i] = i;
    std::sort(sorted.begin(), sorted.end(),
              [&](int x, int y) { return fitness->betterFitness(individuals[x], individuals[y]); });

    int rank = 0;
    ranks[sorted[0]] = rank;
    for (int i = 1; i < population_size; i++) {
        if (individuals[sorted[i]]->objective_value != individuals[sorted[i - 1]]->objective_value)
            rank++;

        ranks[sorted[i]] = rank;
    }
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

void population_t::estimateDistribution() {
    if (!use_conditional_sampling) {
        linkage_model->randomizeOrder();
    }

    if (similarity_measure == 'F') {
        if (current_fitness_dependency_waiting_position == 0) {
            // Learn dependencies and linkage model
            updateFitnessDependencyMatrix();

            initializeFOS();
        } else {
            current_fitness_dependency_waiting_position--;
        }

        if (write_fitness_dependencies) {
            writeFitnessDependencyMonitoringToFile();
        }
    }

    for (int i = 0; i < linkage_model->getLength(); i++)
        estimateDistribution(i);
    updateAMSMeans();
}

void population_t::estimateDistribution(int FOS_index) {
    linkage_model->estimateDistribution(FOS_index, selection, selection_size, fitness_dependency_matrix);
}


void population_t::computeMinMaxBoundsOfCurrentPopulation(double *min, double *max) {
    assert(fitness->number_of_parameters >= 0);

    for (int k = 0; k < fitness->number_of_parameters; k++) {
        min[k] = individuals[0]->variables[k];
        max[k] = individuals[0]->variables[k];
    }

    for (int i = 0; i < population_size; i++) {
        for (int k = 0; k < fitness->number_of_parameters; k++) {
            if (individuals[i]->variables[k] < min[k]) {
                min[k] = individuals[i]->variables[k];
            }
            if (individuals[i]->variables[k] > max[k]) {
                max[k] = individuals[i]->variables[k];
            }
        }
    }
}

double population_t::estimateMean(int var) {
    double mean = 0.0;
    for (int j = 0; j < selection_size; j++)
        mean += selection[j]->variables[var];
    mean /= (double) selection_size;
    return (mean);
}

void population_t::updateAMSMeans() {
    for (int i = 0; i < fitness->number_of_parameters; i++) {
        double new_mean = estimateMean(i);
        if (number_of_generations > 0)
            mean_shift_vector[i] = new_mean - prev_mean_vector[i];

        prev_mean_vector[i] = new_mean;
    }
}

/*void population_t::printCovarianceMatrices()
{
	// First do the maximum-likelihood estimate from data
	for(int i = 0; i < linkage_model->getLength(); i++ )
	{
		printf("F = [");
		for(int j = 0; j < linkage_model->getSetLength(i); j++ )
		{
			printf("%d",linkage_model->sets[i][j]);
			if( j < linkage_model->getSetLength(i)-1 )
				printf(",");
		}
		printf("]\n");
		printf("Cov[%d] = \n",i);
		for(int j = 0; j < linkage_model->getSetLength(i); j++ )
		{
			for(int k = 0; k < linkage_model->getSetLength(i); k++ )
				printf("%10.3e ",covariance_matrices[i](j,k));
			printf("\n");
		}
	}
}*/

void population_t::copyBestSolutionsToPopulation() {
    assert(num_elitists_to_copy ==
           1); // elitists to be copied should first be copied to avoid overwriting them beforehand
    for (int i = 0; i < num_elitists_to_copy; i++) {
        for (int k = 0; k < fitness->number_of_parameters; k++)
            individuals[i]->variables[k] = selection[i]->variables[k];

        individuals[i]->objective_value = selection[i]->objective_value;
        individuals[i]->constraint_value = selection[i]->constraint_value;
    }
}

void population_t::getBestInPopulation(int *individual_index) {
    *individual_index = 0;
    for (int i = 0; i < population_size; i++)
        if (fitness->betterFitness(individuals[i]->objective_value, individuals[i]->constraint_value,
                                   individuals[*individual_index]->objective_value,
                                   individuals[*individual_index]->constraint_value))
            *individual_index = i;
}

void population_t::evaluateCompletePopulation() {
    for (int j = 0; j < population_size; j++)
        fitness->evaluate(individuals[j]);
}

void population_t::generateAndEvaluateNewSolutions() {
    if (!fitness->black_box_optimization && (number_of_generations + 1) % 50 == 0)
        evaluateCompletePopulation();

    short *individual_improved = (short *) Malloc(population_size * sizeof(short));
    for (int k = num_elitists_to_copy; k < population_size; k++)
        individual_improved[k] = 0;

    double alpha_AMS = 0.5 * tau * (((double) population_size) / ((double) (population_size - 1)));
    int number_of_AMS_solutions = (int) (alpha_AMS * (population_size - 1));

    if (perform_factorized_gom) {
        linkage_model->randomizeOrder();
        for (int g = 0; g < linkage_model->getLength(); g++) {
            int FOS_index = linkage_model->order[g];
            double t = getTimer();

            if (selection_during_gom) {
                makeSelection();
                estimateDistribution(FOS_index);
            }
            if (update_elitist_during_gom) {
                updateElitist();
            }

            for (int k = num_elitists_to_copy; k < population_size; k++) {
                sampled_solutions[FOS_index][k] = linkage_model->generatePartialSolution(FOS_index, individuals[k]);
            }

            if (number_of_generations > 0) {
                for (int k = num_elitists_to_copy; k <= number_of_AMS_solutions; k++)
                    applyPartialAMS(sampled_solutions[FOS_index][k],
                                    linkage_model->getDistributionMultiplier(FOS_index));
            }

            for (int k = num_elitists_to_copy; k < population_size; k++) {
                fitness->evaluatePartialSolution(individuals[k], sampled_solutions[FOS_index][k]);
            }

            int num_improvements = 0;
            short *accept_improvement = (short *) Malloc(population_size * sizeof(short));
            for (int k = num_elitists_to_copy; k < population_size; k++) {
                accept_improvement[k] = checkForImprovement(individuals[k], sampled_solutions[FOS_index][k]);
                individual_improved[k] = accept_improvement[k];
                if (accept_improvement[k]) num_improvements++;
            }

            for (int k = num_elitists_to_copy; k < population_size; k++) {
                if (accept_improvement[k] || randomRealUniform01() < linkage_model->getAcceptanceRate()) {
                    sampled_solutions[FOS_index][k]->is_accepted = 1;
                    insertImprovement(individuals[k], sampled_solutions[FOS_index][k]);
                } else {
                    for (int i = 0; i < sampled_solutions[FOS_index][k]->num_touched_variables; i++) {
                        int ind = sampled_solutions[FOS_index][k]->touched_indices[i];
                        sampled_solutions[FOS_index][k]->touched_variables[i] = individuals[k]->variables[ind];
                    }
                    sampled_solutions[FOS_index][k]->objective_value = individuals[k]->objective_value;
                    sampled_solutions[FOS_index][k]->constraint_value = individuals[k]->constraint_value;
                }

                if (fitness->betterFitness(sampled_solutions[FOS_index][k]->objective_value,
                                           sampled_solutions[FOS_index][k]->constraint_value, objective_value_elitist,
                                           constraint_value_elitist))
                    sampled_solutions[FOS_index][k]->improves_elitist = 1;
            }
            free(accept_improvement);

            gomtime += getTimer() - t;
            linkage_model->adaptDistributionMultiplier(FOS_index, &sampled_solutions[FOS_index][num_elitists_to_copy],
                                                       population_size - num_elitists_to_copy);
        }

        for (int g = 0; g < linkage_model->getLength(); g++)
            for (int k = num_elitists_to_copy; k < population_size; k++)
                delete (sampled_solutions[g][k]);
    }

    if (perform_eda_gom) {
        solution_t **full_solutions = (solution_t **) Malloc(population_size * sizeof(solution_t * ));
        for (int k = num_elitists_to_copy; k < population_size; k++) {
            full_solutions[k] = new solution_t(fitness->number_of_parameters);

            for (int i = 0; i < fitness->number_of_parameters; i++) {
                full_solutions[k]->variables[i] = -1e16;
            }
        }

        linkage_model->randomizeOrder();
        for (int g = 0; g < linkage_model->getLength(); g++) {
            int FOS_index = linkage_model->order[g];

            for (int k = num_elitists_to_copy; k < population_size; k++) {
                partial_solution_t *partial_solution = linkage_model->generatePartialSolution(FOS_index,
                                                                                              individuals[k]);

                for (int i = 0; i < partial_solution->num_touched_variables; i++) {
                    full_solutions[k]->variables[partial_solution->touched_indices[i]] = partial_solution->touched_variables[i];
                }

                delete partial_solution;
            }
        }

        for (int k = num_elitists_to_copy; k < population_size; k++) {
            // Sanity check
            for (int i = 0; i < fitness->number_of_parameters; i++) {
                assert(full_solutions[k]->variables[i] != -1e16);
            }

            fitness->evaluate(full_solutions[k]);

            if (fitness->betterFitness(full_solutions[k], individuals[k])) {
                for (int i = 0; i < fitness->number_of_parameters; i++) {
                    individuals[k]->variables[i] = full_solutions[k]->variables[i];
                }
                individuals[k]->objective_value = full_solutions[k]->objective_value;
                individuals[k]->constraint_value = full_solutions[k]->constraint_value;
                individual_improved[k] = 1;
            }
        }
    }


    if (number_of_generations > 0) {
        for (int k = num_elitists_to_copy; k <= number_of_AMS_solutions; k++)
            individual_improved[k] |= applyAMS(k);
    }

    short generational_improvement = 0;
    for (int i = num_elitists_to_copy; i < population_size; i++) {
        if (!individual_improved[i])
            individual_NIS[i]++;
        else {
            individual_NIS[i] = 0;
            generational_improvement = 1;
        }
    }

    int best_individual_index;
    getBestInPopulation(&best_individual_index);
    for (int k = num_elitists_to_copy; k < population_size; k++)
        if (individual_NIS[k] > maximum_no_improvement_stretch)
            applyForcedImprovements(k, best_individual_index);

    if (generational_improvement)
        linkage_model->no_improvement_stretch = 0;
    else {
        short all_multipliers_leq_one = 1;
        for (int j = 0; j < linkage_model->getLength(); j++)
            if (linkage_model->getDistributionMultiplier(j) > 1.0) {
                all_multipliers_leq_one = 0;
                break;
            }

        if (all_multipliers_leq_one)
            linkage_model->no_improvement_stretch++;
    }

    /*printf("[%d] NIS = ",number_of_generations);
    for(int i = 0; i < population_size; i++ )
        printf("%3d ",individual_NIS[i]);
    printf("\n");
    printf("[%d] MUL = ",number_of_generations);
    for(int i = 0; i < linkage_model->getLength(); i++ )
        printf("%6.2lf ",distribution_multipliers[i]);
    printf("\n");*/

    free(individual_improved);
}

void population_t::applyPartialAMS(partial_solution_t *solution, double cmul) {
    short out_of_range = 1;
    double shrink_factor = 2;
    double *result = (double *) Malloc(solution->num_touched_variables * sizeof(double));
    while ((out_of_range == 1) && (shrink_factor > 1e-10)) {
        shrink_factor *= 0.5;
        out_of_range = 0;
        for (int m = 0; m < solution->num_touched_variables; m++) {
            int im = solution->touched_indices[m];
            result[m] = solution->touched_variables[m] + shrink_factor * delta_AMS * cmul * (mean_shift_vector[im]);
            if (!fitness->isParameterInRangeBounds(result[m], im)) {
                out_of_range = 1;
                break;
            }
        }
    }
    if (!out_of_range) {
        for (int m = 0; m < solution->num_touched_variables; m++) {
            int im = solution->touched_indices[m];
            solution->touched_variables[m] = result[m];
        }
    }
    free(result);
}

short population_t::checkForImprovement(solution_t *solution, partial_solution_t *part) {
    return (fitness_t::betterFitness(part->objective_value, part->constraint_value, solution->objective_value,
                                     solution->constraint_value));
}

void population_t::insertImprovement(solution_t *solution, partial_solution_t *part) {
    for (int j = 0; j < part->num_touched_variables; j++) {
        int ind = part->touched_indices[j];
        solution->variables[ind] = part->touched_variables[j];
    }
    solution->objective_value = part->objective_value;
    solution->constraint_value = part->constraint_value;
}

short population_t::applyAMS(int individual_index) {
    short out_of_range = 1;
    short improvement = 0;
    double delta_AMS = 2;
    double shrink_factor = 2;
    solution_t *solution_AMS = new solution_t(fitness->number_of_parameters);
    while ((out_of_range == 1) && (shrink_factor > 1e-10)) {
        shrink_factor *= 0.5;
        out_of_range = 0;
        for (int m = 0; m < fitness->number_of_parameters; m++) {
            solution_AMS->variables[m] =
                    individuals[individual_index]->variables[m] + shrink_factor * delta_AMS * (mean_shift_vector[m]);
            if (!fitness->isParameterInRangeBounds(solution_AMS->variables[m], m)) {
                out_of_range = 1;
                break;
            }
        }
    }
    if (!out_of_range) {
        short improvement;
        fitness->evaluate(solution_AMS);
        improvement = fitness->betterFitness(solution_AMS->objective_value, solution_AMS->constraint_value,
                                             individuals[individual_index]->objective_value,
                                             individuals[individual_index]->constraint_value);
        //if( improvement )
        if (randomRealUniform01() < linkage_model->getAcceptanceRate() || improvement) // BLA
        {
            individuals[individual_index]->objective_value = solution_AMS->objective_value;
            individuals[individual_index]->constraint_value = solution_AMS->constraint_value;
            for (int m = 0; m < fitness->number_of_parameters; m++)
                individuals[individual_index]->variables[m] = solution_AMS->variables[m];
            improvement = 1;
        }
    }
    delete (solution_AMS);

    return (improvement);
}

void population_t::applyForcedImprovements(int individual_index, int donor_index) {
    double obj_val, cons_val;
    short improvement = 0;
    double alpha = 1.0;

    while (alpha >= 0.01) {
        alpha *= 0.5;
        for (int io = 0; io < linkage_model->getLength(); io++) {
            int i = linkage_model->order[io];
            std::vector<int> touched_indices = linkage_model->sets[i];
            int num_touched_indices = linkage_model->getSetLength(i);

            vec FI_vars = vec(num_touched_indices, fill::none);
            for (int j = 0; j < num_touched_indices; j++) {
                FI_vars[j] = alpha * individuals[individual_index]->variables[touched_indices[j]] +
                             (1 - alpha) * individuals[donor_index]->variables[touched_indices[j]];
            }
            partial_solution_t *FI_solution = new partial_solution_t(FI_vars, touched_indices);
            fitness->evaluatePartialSolution(individuals[individual_index], FI_solution);
            improvement = fitness->betterFitness(FI_solution->objective_value, FI_solution->constraint_value,
                                                 individuals[individual_index]->objective_value,
                                                 individuals[individual_index]->constraint_value);

            if (improvement) {
                for (int j = 0; j < num_touched_indices; j++)
                    individuals[individual_index]->variables[touched_indices[j]] = FI_solution->touched_variables[j];
                individuals[individual_index]->objective_value = FI_solution->objective_value;
                individuals[individual_index]->constraint_value = FI_solution->constraint_value;
            }
            delete FI_solution;

            if (improvement)
                break;
        }
        if (improvement)
            break;
    }

    if (!improvement) {
        for (int i = 0; i < fitness->number_of_parameters; i++)
            individuals[individual_index]->variables[i] = individuals[donor_index]->variables[i];
        individuals[individual_index]->objective_value = individuals[donor_index]->objective_value;
        individuals[individual_index]->constraint_value = individuals[donor_index]->constraint_value;
    }
}

double population_t::getFitnessMean() {
    double objective_avg = 0.0;
    for (int i = 0; i < population_size; i++)
        objective_avg += individuals[i]->objective_value;
    objective_avg = objective_avg / ((double) population_size);

    return (objective_avg);
}

double population_t::getFitnessVariance() {
    double objective_avg = getFitnessMean();

    double objective_var = 0.0;
    for (int i = 0; i < population_size; i++)
        objective_var +=
                (individuals[i]->objective_value - objective_avg) * (individuals[i]->objective_value - objective_avg);
    objective_var = objective_var / ((double) population_size);

    if (objective_var <= 0.0)
        objective_var = 0.0;
    return (objective_var);
}

void population_t::initializeDefaultParameters() {
    eta_cov = 1.0;
    tau = 0.35;
    st_dev_ratio_threshold = 1.0;
    distribution_multiplier_decrease = 0.9;
    distribution_multiplier_increase = 1.0 / distribution_multiplier_decrease;
    maximum_no_improvement_stretch = 100;
    delta_AMS = 2.0;
    selection_size = (int) (tau * population_size);
}

void population_t::initializeNewPopulationMemory() {
    individuals = (solution_t **) Malloc(population_size * sizeof(solution_t * ));
    for (int j = 0; j < population_size; j++)
        individuals[j] = new solution_t(fitness->number_of_parameters);

    ranks = (double *) Malloc(population_size * sizeof(double));

    selection = (solution_t **) Malloc(selection_size * sizeof(solution_t * ));

    mean_shift_vector = (double *) Malloc(fitness->number_of_parameters * sizeof(double));
    prev_mean_vector = (double *) Malloc(fitness->number_of_parameters * sizeof(double));

    individual_NIS = (int *) Malloc(population_size * sizeof(int));

    initializeFOS();

    population_terminated = 0;

    number_of_generations = 0;
}

/**
 * Initializes the linkage tree
 */
void population_t::initializeFOS() {
    fos_t *new_FOS;

    similarity_measure = 'M';
    learn_linkage_tree = 0;
    static_linkage_tree = 0;
    random_linkage_tree = 0;
    prune_linkage_tree = 0;
    FOS_element_ub = number_of_parameters;
    fitness_based_ordering = 0;

    assert(FOS_element_size != 0);

    if (FOS_element_size > 0) {
        new_FOS = new fos_t(FOS_element_size);

    } else if (FOS_element_size == -1) {
        FOS_element_size = number_of_parameters;

        new_FOS = new fos_t();
        std::vector<int> full_fos_element;
        for (int i = 0; i < fitness->number_of_parameters; i++) {
            full_fos_element.push_back(i);
        }
        new_FOS->addGroup(full_fos_element);

    } else if (FOS_element_size == -4) {
        static_linkage_tree = 1;
        FOS_element_ub = 100;
        new_FOS = new fos_t((double **) NULL);

    } else if (FOS_element_size == -5) {
        static_linkage_tree = 0;
        learn_linkage_tree = 1;
        similarity_measure = 'F';
        FOS_element_ub = 100;
        prune_linkage_tree = 0;

        if (linkage_model == NULL) {
            initializeFitnessDependencyMatrix();
            updateFitnessDependencyMatrix();
        }

        new_FOS = new fos_t(fitness_dependency_matrix, false);

    } else if (FOS_element_size == -6) {
        static_linkage_tree = 0;
        learn_linkage_tree = 1;
        similarity_measure = 'F';
        FOS_element_ub = 100;
        prune_linkage_tree = 1;

        if (linkage_model == NULL) {
            initializeFitnessDependencyMatrix();
            updateFitnessDependencyMatrix();
        }

        new_FOS = new fos_t(fitness_dependency_matrix, false);

    } else if (FOS_element_size == -7) { // mp-fb-online-gg
        static_linkage_tree = 0;
        learn_linkage_tree = 1;
        similarity_measure = 'F';
        FOS_element_ub = 100;
        prune_linkage_tree = 1;
        perform_factorized_gom = 0;
        perform_eda_gom = 1;

        if (linkage_model == NULL) {
            initializeFitnessDependencyMatrix();
            updateFitnessDependencyMatrix();
        }

        new_FOS = new fos_t(fitness_dependency_matrix, true);

    } else if (FOS_element_size == -8) { // mp-fb-online-fg
        static_linkage_tree = 0;
        learn_linkage_tree = 1;
        similarity_measure = 'F';
        FOS_element_ub = 100;
        prune_linkage_tree = 1;
        perform_factorized_gom = 1;
        perform_eda_gom = 0;

        if (linkage_model == NULL) {
            initializeFitnessDependencyMatrix();
            updateFitnessDependencyMatrix();
        }

        new_FOS = new fos_t(fitness_dependency_matrix, true);

    } else if (FOS_element_size == -9) { // mp-fb-online-hg
        static_linkage_tree = 0;
        learn_linkage_tree = 1;
        similarity_measure = 'F';
        FOS_element_ub = 100;
        prune_linkage_tree = 1;
        perform_factorized_gom = 1;
        perform_eda_gom = 1;

        if (linkage_model == NULL) {
            initializeFitnessDependencyMatrix();
            updateFitnessDependencyMatrix();
        }

        new_FOS = new fos_t(fitness_dependency_matrix, true);

    } else if (FOS_element_size <= -10) {
        int id = -1 * FOS_element_size;

        use_set_cover = (id % 10) > 0;
        id /= 10;

        use_conditional_sampling = (id % 10) > 0;
        id /= 10;

        seed_cliques_per_variable = id % 10;
        id /= 10;

        int is_fitness_based = (id % 10) > 0;
        id /= 10;

        if (use_set_cover) {
            assert(is_fitness_based == 0);
        }

        if (is_fitness_based) {
            similarity_measure = 'F';

            if (is_fitness_based == 1) {
                fitness_based_ordering = 1;
            }

            if (linkage_model == NULL) {
                initializeFitnessDependencyMatrix();
                updateFitnessDependencyMatrix();
            }
            fitness->variable_interaction_graph = buildVariableInteractionGraphBasedOnFitnessDependencies();
        }

        include_full_fos_element = (id % 10) == 1;
        id /= 10;

        include_cliques_as_fos_elements = (id % 10) == 1;
        id /= 10;

        max_clique_size = id;

        if (!include_full_fos_element && !include_cliques_as_fos_elements) {
            // LT-conditional FOS detected
            include_full_fos_element = 1;
            include_cliques_as_fos_elements = 1;

            learn_conditional_linkage_tree = true;
            if (is_fitness_based) {
                learn_linkage_tree = true;
            }

            if (seed_cliques_per_variable == 3) {
                prune_linkage_tree = 1;
                seed_cliques_per_variable = 0;
            }
        }

        if (max_clique_size == 1) {
            seed_cliques_per_variable = false;
        }

        // Make sure that, if at all, either set cover or clique seeding is used as strategy, not both
        assert(!(use_set_cover && seed_cliques_per_variable));

        if (!use_conditional_sampling) {
            // In non-conditional setting, generational, factorized and hybrid GOM have a different connotation
            perform_factorized_gom = include_cliques_as_fos_elements;
            perform_eda_gom = include_full_fos_element;

            include_cliques_as_fos_elements = true;
            include_full_fos_element = false;
        }

        new_FOS = new fos_t(fitness->variable_interaction_graph,
                            ((is_fitness_based && fitness_based_ordering) ? &fitness_dependency_matrix : NULL));
    }

    if (linkage_model != NULL) {
        // Inherit distribution multipliers
        int **FOS_element_similarity_matrix = (int **) Malloc((linkage_model->getLength()) * sizeof(int *));
        for (int i = 0; i < linkage_model->getLength(); i++) {
            FOS_element_similarity_matrix[i] = (int *) Malloc((new_FOS->getLength()) * sizeof(int));
        }

        for (int i = 0; i < linkage_model->getLength(); i++) {
            for (int j = 0; j < new_FOS->getLength(); j++) {
                int a = 0;
                int b = 0;
                int matches = 0;
                while (a < linkage_model->sets[i].size() && b < new_FOS->sets[j].size()) {
                    if (linkage_model->sets[i][a] < new_FOS->sets[j][b]) {
                        a++;
                    } else if (linkage_model->sets[i][a] > new_FOS->sets[j][b]) {
                        b++;
                    } else {
                        a++;
                        b++;
                        matches++;
                    }
                }
                FOS_element_similarity_matrix[i][j] =
                        (int) 10000 * (2.0 * matches / (linkage_model->sets[i].size() + new_FOS->sets[j].size()));
            }
        }

        for (int i = 0; i < new_FOS->getLength(); i++) {
            int max_index = 0;
            int max_similarity = -1;
            for (int j = 0; j < linkage_model->getLength(); j++) {
                if (FOS_element_similarity_matrix[j][i] > max_similarity) {
                    max_index = j;
                    max_similarity = FOS_element_similarity_matrix[j][i];
                }
            }
            new_FOS->distributions[i]->distribution_multiplier = linkage_model->distributions[max_index]->distribution_multiplier;
        }

        for (int i = 0; i < linkage_model->getLength(); i++) {
            free(FOS_element_similarity_matrix[i]);
        }
        free(FOS_element_similarity_matrix);

        for (int j = 0; j < linkage_model->getLength(); j++)
            free(sampled_solutions[j]);
        free(sampled_solutions);

        delete linkage_model;
    }

    linkage_model = new_FOS;

    sampled_solutions = (partial_solution_t ***) Malloc(linkage_model->getLength() * sizeof(partial_solution_t * *));
    for (int j = 0; j < linkage_model->getLength(); j++)
        sampled_solutions[j] = (partial_solution_t **) Malloc(population_size * sizeof(partial_solution_t * ));
}

std::map<int, std::set<int>> population_t::buildVariableInteractionGraphBasedOnFitnessDependencies() {
    std::map<int, std::set<int>> vig;

    for (int i = 0; i < number_of_parameters; i++) {
        vig[i] = std::set<int>();
    }
    for (int i = 0; i < number_of_parameters; i++) {
        for (int j = 0; j < number_of_parameters; j++) {
            if (i == j) {
                continue;
            }
            if (fitness_dependency_matrix[i][j] != 0.0) {
                vig[i].insert(j);
            }
        }
    }
    assert(vig.size() == number_of_parameters);
    return vig;
}

/**
 * Initializes the parameter range bounds.
 */
void population_t::initializeParameterRangeBounds(double lower_user_range, double upper_user_range) {
    lower_init_ranges = (double *) Malloc(fitness->number_of_parameters * sizeof(double));
    upper_init_ranges = (double *) Malloc(fitness->number_of_parameters * sizeof(double));

    for (int i = 0; i < fitness->number_of_parameters; i++) {
        lower_init_ranges[i] = lower_user_range;
        if (lower_user_range < fitness->getLowerRangeBound(i))
            lower_init_ranges[i] = fitness->getLowerRangeBound(i);
        if (lower_user_range > fitness->getUpperRangeBound(i))
            lower_init_ranges[i] = fitness->getLowerRangeBound(i);

        upper_init_ranges[i] = upper_user_range;
        if (upper_user_range > fitness->getUpperRangeBound(i))
            upper_init_ranges[i] = fitness->getUpperRangeBound(i);
        if (upper_user_range < fitness->getLowerRangeBound(i))
            upper_init_ranges[i] = fitness->getUpperRangeBound(i);
    }
}

void population_t::initializePopulationAndFitnessValues() {
    for (int j = 0; j < population_size; j++) {
        individual_NIS[j] = 0;
        for (int k = 0; k < fitness->number_of_parameters; k++)
            individuals[j]->variables[k] =
                    lower_init_ranges[k] + (upper_init_ranges[k] - lower_init_ranges[k]) * randomRealUniform01();

        fitness->evaluate(individuals[j]);
    }
    //for(int k = 0; k < number_of_parameters; k++ )
    //individuals[0]->variables[k] = k;
    //individuals[0]->variables[0] = 1;
    /*fitness->evaluate( individuals[0] );
    printf("f = %10.10e\n", individuals[0]->objective_value);
    exit(0);*/
    /*individuals[0]->variables[0] = 0;
    individuals[0]->variables[1] = 1;
    fitness->evaluate( individuals[0] );
    printf("f = %10.10e\n", individuals[0]->objective_value);
    individuals[0]->variables[1] = 0;
    individuals[0]->variables[2] = 1;
    fitness->evaluate( individuals[0] );
    printf("f = %10.10e\n", individuals[0]->objective_value);
    exit(0);*/
}

void population_t::initializeFitnessDependencyMatrix() {
    number_of_checked_fitness_dependency_pairs = 0;
    total_fitness_dependencies_found = 0;
    fitness_dependency_check_iteration = 0;
    current_fitness_dependency_waiting_position = 0;
    number_of_fitness_dependency_waiting_cycles = 2;

    assert(number_of_parameters >= 0);
    fitness_dependency_matrix.resize(number_of_parameters);
    for (int i = 0; i < number_of_parameters; i++) {
        fitness_dependency_matrix[i].resize(number_of_parameters);
    }

    number_of_fitness_dependency_pairs = (int) round(
            ((number_of_parameters * number_of_parameters) - number_of_parameters) / 2.0);
    fitness_dependency_pairs = mat(number_of_fitness_dependency_pairs, 2);

    int counter = 0;
    for (int i = 0; i < number_of_parameters; i++) {
        for (int j = i + 1; j < number_of_parameters; j++) {
            assert(counter <= number_of_fitness_dependency_pairs);

            fitness_dependency_pairs(counter, 0) = i;
            fitness_dependency_pairs(counter, 1) = j;
            counter++;
        }
    }
    assert(counter == number_of_fitness_dependency_pairs);

    // Shuffle the order of fitness dependency pairs
    counter = number_of_fitness_dependency_pairs;
    for (int i = counter - 1; i >= 0; --i) {
        int j = randomInt(i + 1);
        assert(j <= i);

        // Swap the last element with element at random index
        int temp0 = fitness_dependency_pairs(i, 0);
        int temp1 = fitness_dependency_pairs(i, 1);
        fitness_dependency_pairs(i, 0) = fitness_dependency_pairs(j, 0);
        fitness_dependency_pairs(i, 1) = fitness_dependency_pairs(j, 1);
        fitness_dependency_pairs(j, 0) = temp0;
        fitness_dependency_pairs(j, 1) = temp1;
    }

    first_individual_for_fitness_comparison = new solution_t(number_of_parameters);
    second_individual_for_fitness_comparison = new solution_t(number_of_parameters);

    fitnesses_of_first_individual_variants = (double *) Malloc(number_of_parameters * sizeof(double));

    for (int i = 0; i < number_of_parameters; i++) {
        for (int j = 0; j < number_of_parameters; j++) {
            fitness_dependency_matrix[i][j] = 0.0;
        }
    }
    number_of_checked_fitness_dependency_pairs = 0;
    fitness_dependency_pairs_to_check_per_iteration = number_of_parameters;

    if (write_fitness_dependencies) {
        initializeFitnessDependencyMonitoringFile();
    }
}

void population_t::updateFitnessDependencyMatrix() {
    // Compute minima and maxima
    double *population_min = (double *) Malloc(number_of_parameters * sizeof(double));
    double *population_max = (double *) Malloc(number_of_parameters * sizeof(double));
    computeMinMaxBoundsOfCurrentPopulation(population_min, population_max);

    assert(number_of_parameters >= 0);

    solution_t *individual_to_compare = new solution_t(number_of_parameters);

    assert(number_of_checked_fitness_dependency_pairs >= 0);

    // In case a cycle is about to start
    if (number_of_checked_fitness_dependency_pairs == 0) {
        // Initialize first and second individuals to representative positions in population
        for (int k = 0; k < number_of_parameters; k++) {
            double min = population_min[k];
            double max = population_max[k];

            if (nround(min, 2) == nround(max, 2)) {
                max = upper_init_ranges[k];
            }
            first_individual_for_fitness_comparison->variables[k] = min + ((max - min) * 0.35);
            double parameter_diff = (max - min) * 0.35;
            second_individual_for_fitness_comparison->variables[k] =
                    parameter_diff + first_individual_for_fitness_comparison->variables[k];
            individual_to_compare->variables[k] = first_individual_for_fitness_comparison->variables[k];
        }

        // Evaluate first individual
        fitness->evaluate(first_individual_for_fitness_comparison);

        for (int k = 0; k < number_of_parameters; k++) {
            vec_t<double> touched_variables;
            touched_variables.push_back(second_individual_for_fitness_comparison->variables[k]);
            vec_t<int> touched_indices;
            touched_indices.push_back(k);

            partial_solution_t *partial_solution = new partial_solution_t(touched_variables, touched_indices);

            fitness->evaluatePartialSolution(first_individual_for_fitness_comparison, partial_solution);
            fitnesses_of_first_individual_variants[k] = partial_solution->objective_value;

            delete partial_solution;
        }

        // Shuffle the order of fitness dependency pairs
        int counter = number_of_fitness_dependency_pairs;
        assert(number_of_fitness_dependency_pairs >= 0);
        for (int i = counter - 1; i >= 0; --i) {
            int j = randomInt(i + 1);
            assert(j <= i);

            // Swap the last element with element at random index
            int temp0 = fitness_dependency_pairs(i, 0);
            int temp1 = fitness_dependency_pairs(i, 1);
            assert(temp0 >= 0);
            assert(temp0 < number_of_parameters);
            assert(temp1 >= 0);
            assert(temp1 < number_of_parameters);

            fitness_dependency_pairs(i, 0) = fitness_dependency_pairs(j, 0);
            fitness_dependency_pairs(i, 1) = fitness_dependency_pairs(j, 1);
            assert(fitness_dependency_pairs(j, 0) >= 0);
            assert(fitness_dependency_pairs(j, 1) < number_of_parameters);
            assert(fitness_dependency_pairs(j, 0) >= 0);
            assert(fitness_dependency_pairs(j, 1) < number_of_parameters);

            fitness_dependency_pairs(j, 0) = temp0;
            fitness_dependency_pairs(j, 1) = temp1;
        }
    } else {
        for (int k = 0; k < number_of_parameters; k++) {
            individual_to_compare->variables[k] = first_individual_for_fitness_comparison->variables[k];
        }
    }

    fitness_dependency_check_iteration++;
    int max_index = number_of_checked_fitness_dependency_pairs + fitness_dependency_pairs_to_check_per_iteration;
    if (max_index >= number_of_fitness_dependency_pairs) {
        max_index = number_of_fitness_dependency_pairs;
    }

    // Copy over first individual to the comparison individual
    for (int k = 0; k < number_of_parameters; k++) {
        individual_to_compare->variables[k] = first_individual_for_fitness_comparison->variables[k];
    }

    int found_fitness_dependencies = 0;
    double max_dependency = 0.0;

    assert(number_of_checked_fitness_dependency_pairs >= 0);
    assert(max_index >= 0);
    assert(max_index <= number_of_fitness_dependency_pairs);

    for (int k = number_of_checked_fitness_dependency_pairs; k < max_index; k++) {
        found_fitness_dependencies += computeFitnessDependency(k, individual_to_compare);
    }

    total_fitness_dependencies_found += found_fitness_dependencies;
    number_of_checked_fitness_dependency_pairs += fitness_dependency_pairs_to_check_per_iteration;

    int found_dependencies_per_iteration = total_fitness_dependencies_found / fitness_dependency_check_iteration;

    bool low_dependency_check_performance = ((found_fitness_dependencies == 0) && (found_dependencies_per_iteration <
                                                                                   minimal_fitness_dependencies_per_iteration));
    bool all_dependency_pairs_checked = (number_of_checked_fitness_dependency_pairs >=
                                         number_of_fitness_dependency_pairs);

//    if (low_dependency_check_performance || all_dependency_pairs_checked) {
    if (all_dependency_pairs_checked) {
        current_fitness_dependency_waiting_position = number_of_fitness_dependency_waiting_cycles;
        number_of_checked_fitness_dependency_pairs = 0;
        number_of_fitness_dependency_waiting_cycles *= 2;
        fitness_dependency_check_iteration = 0;
        total_fitness_dependencies_found = 0;
    }

    delete individual_to_compare;
    free(population_min);
    free(population_max);
}

int population_t::computeFitnessDependency(int k, solution_t *individual_to_compare) {
    assert(k < number_of_fitness_dependency_pairs);

    int i = fitness_dependency_pairs(k, 0);
    int j = fitness_dependency_pairs(k, 1);

    assert(number_of_parameters >= 0);

    assert(i >= 0);
    assert(i < number_of_parameters);
    assert(j >= 0);
    assert(j < number_of_parameters);

    double change_i = fitnesses_of_first_individual_variants[i];
    double change_j = fitnesses_of_first_individual_variants[j];

    vec_t<double> touched_variables;
    touched_variables.push_back(second_individual_for_fitness_comparison->variables[i]);
    touched_variables.push_back(second_individual_for_fitness_comparison->variables[j]);
    vec_t<int> touched_indices;
    touched_indices.push_back(i);
    touched_indices.push_back(j);

    partial_solution_t *partial_solution = new partial_solution_t(touched_variables, touched_indices);

    fitness->evaluatePartialSolution(first_individual_for_fitness_comparison, partial_solution);
    double change_i_j = partial_solution->objective_value;

    delete partial_solution;

    double original_objective = first_individual_for_fitness_comparison->objective_value;
    change_i = change_i / original_objective;
    change_j = change_j / original_objective;
    change_i_j = change_i_j / original_objective;

    double delta_i = fabs(1.0 - change_i);
    double delta_ij = fabs(change_j - change_i_j);
    delta_i = nround(delta_i, 12);
    delta_ij = nround(delta_ij, 12);

    double dependency = 0.0;
    double inverted_difference;

    if (delta_ij == 0.0) {
        double temp = delta_i;
        delta_i = delta_ij;
        delta_ij = temp;
    }

    if (delta_ij != 0.0) {
        inverted_difference = nround(fabs((double) delta_i / delta_ij), 6);

        if (inverted_difference > 1.0) {
            inverted_difference = nround(fabs((double) delta_ij / delta_i), 6);
        }
    } else {
        inverted_difference = 1.0;
    }

    int found_fitness_dependencies = 0;
    dependency = nround(1 - inverted_difference, 6);
    if (inverted_difference < 1 && inverted_difference > 0) {
        found_fitness_dependencies++;
    } else {
        dependency = 0.0;
    }

    fitness_dependency_matrix[i][j] = dependency;
    fitness_dependency_matrix[j][i] = dependency;

    return found_fitness_dependencies;
}

void population_t::printFitnessDependencyMatrix() {
    for (int i = 0; i < number_of_parameters; i++) {
        for (int j = 0; j < number_of_parameters; j++) {
            printf("%f", fitness_dependency_matrix[i][j]);

            if (j < number_of_parameters - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
    fflush(stdout);
}

void population_t::initializeFitnessDependencyMonitoringFile() {
    FILE *f = fopen("fitness_dependency_monitoring_per_generation.dat", "w");
    fprintf(f,
            "generation,number_of_checked_fitness_dependency_pairs,total_fitness_dependencies_found,fitness_dependency_check_iteration,current_fitness_dependency_waiting_position,number_of_fitness_dependency_waiting_cycles,matrix,fos\n");
    fclose(f);
}

void population_t::writeFitnessDependencyMonitoringToFile() {
    FILE *f = fopen("fitness_dependency_monitoring_per_generation.dat", "a");

    fprintf(f, "%d,", number_of_generations);
    fprintf(f, "%d,", number_of_checked_fitness_dependency_pairs);
    fprintf(f, "%d,", total_fitness_dependencies_found);
    fprintf(f, "%d,", fitness_dependency_check_iteration);
    fprintf(f, "%d,", current_fitness_dependency_waiting_position);
    fprintf(f, "%d,", number_of_fitness_dependency_waiting_cycles);

    for (int i = 0; i < number_of_parameters; i++) {
        for (int j = 0; j < number_of_parameters; j++) {
            fprintf(f, "%f", fitness_dependency_matrix[i][j]);

            if (!(j == number_of_parameters - 1 && i == number_of_parameters - 1)) {
                fprintf(f, "|");
            }
        }
    }
    fprintf(f, ",");

    for (size_t i = 0; i < linkage_model->sets.size(); i++) {
        fprintf(f, "[");
        int c = 0;
        for (int v: linkage_model->sets[i]) {
            if (c == linkage_model->sets[i].size() - 1)
                fprintf(f, "%d", v);
            else
                fprintf(f, "%d^", v);
            c++;
        }
        fprintf(f, "]");
        if (i < linkage_model->sets.size() - 1) {
            fprintf(f, "|");
        }
    }
    fprintf(f, "\n");
    fclose(f);
}
