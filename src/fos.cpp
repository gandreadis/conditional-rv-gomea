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
#include "fos.h"
#include <queue>
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
int max_clique_size;
bool include_cliques_as_fos_elements;
bool include_full_fos_element;
bool learn_conditional_linkage_tree;
int seed_cliques_per_variable;
bool use_conditional_sampling = false;
int FOS_element_ub,                       /* Cut-off value for bounded fixed linkage tree (BFLT). */
prune_linkage_tree,
        learn_linkage_tree,                   /* Whether the FOS is learned at the start of each generation. */
static_linkage_tree,                  /* Whether the FOS is fixed throughout optimization. */
random_linkage_tree,                  /* Whether the fixed linkage tree is learned based on a random distance measure. */
FOS_element_size = 0;                     /* If positive, the size of blocks of consecutive variables in the FOS. If negative, determines specific kind of linkage tree FOS. */
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

// Dummy FOS
fos_t::fos_t() {
}

// MPM
fos_t::fos_t(int element_size) {
    assert(number_of_parameters % element_size == 0);
    for (int i = 0; i < number_of_parameters / element_size; i++) {
        std::vector<int> group;
        for (int j = 0; j < element_size; j++)
            group.push_back(i * element_size + j);
        addGroup(group);
    }
    order = randomPermutation(getLength());
}

// Read a FOS from a file
fos_t::fos_t(FILE *file) {
    char c, string[1000];
    int i, j, k;

    /* Length */
    k = 0;
    int length = 0;
    c = fgetc(file);
    while ((c != EOF)) {
        while (c != '\n')
            c = fgetc(file);
        length++;
        c = fgetc(file);
    }

    fclose(file);
    fflush(stdout);
    file = fopen("FOS.in", "r");

    for (i = 0; i < length; i++) {
        std::vector<int> vec;
        c = fgetc(file);
        j = 0;
        while ((c != '\n') && (c != EOF)) {
            k = 0;
            while ((c == ' ') || (c == '\n') || (c == '\t'))
                c = fgetc(file);
            while ((c != ' ') && (c != '\n') && (c != '\t')) {
                string[k] = (char) c;
                c = fgetc(file);
                k++;
            }
            string[k] = '\0';
            printf("FOS[%d][%d] = %d\n", i, j, (int) atoi(string));
            vec.push_back((int) atoi(string));
            j++;
        }
        addGroup(vec);
    }
    fclose(file);
    order = randomPermutation(getLength());
}

// Copy a FOS
fos_t::fos_t(const fos_t &other) {
    for (int i = 0; i < other.sets.size(); i++) {
        std::vector<int> vec;
        for (int j = 0; j < sets[i].size(); j++)
            vec.push_back(other.sets[i][j]);
        addGroup(vec);
    }
    p_accept = other.p_accept;
}


fos_t::fos_t(vec_t <vec_t<double>> fitness_dependency_matrix, bool is_marginal) {
    assert(learn_linkage_tree);

    double **array_version = (double **) Malloc(number_of_parameters * sizeof(double *));
    for (int i = 0; i < number_of_parameters; i++) {
        array_version[i] = (double *) Malloc(number_of_parameters * sizeof(double));
        for (int j = 0; j < number_of_parameters; j++) {
            array_version[i][j] = fitness_dependency_matrix[i][j];
        }
    }

    if (is_marginal) {
        double *temp_fos_indices = (double *) Malloc(number_of_parameters * sizeof(double));
        int *grouped = (int *) Malloc(number_of_parameters * sizeof(int));
        for (int i = 0; i < number_of_parameters; i++) {
            grouped[i] = 0;
        }

        int i = 0;
        while (i < number_of_parameters) {
            if (grouped[i]) {
                i++;
                continue;
            } else {
                grouped[i] = 1;
            }
            int k = 1;
            temp_fos_indices[0] = i;
            for (int j = i + 1; j < number_of_parameters; j++) {
                if (grouped[j]) {
                    continue;
                }
                double dependency = fitness_dependency_matrix[i][j];
                if (dependency != 0.0) {
                    grouped[j] = 1;
                    temp_fos_indices[k] = j;
                    k++;
                }
            }

            std::vector<int> vec;
            for (int l = 0; l < k; l++) {
                vec.push_back(temp_fos_indices[l]);
            }
            addGroup(vec);

            i++;
        }
    } else {
        deriveTree(array_version);
    }
}

// Learn a linkage tree
fos_t::fos_t(double **covariance_matrix) {
    assert(learn_linkage_tree || (covariance_matrix == NULL && static_linkage_tree));

    double **MI_matrix = NULL;
    if (learn_linkage_tree) {
        MI_matrix = computeMIMatrix(covariance_matrix, number_of_parameters);
    }

    deriveTree(MI_matrix);
}

void fos_t::deriveTree(double **MI_matrix) {
    /* Initialize MPM to the univariate factorization */
    int **mpm = (int **) Malloc(number_of_parameters * sizeof(int *));
    int *mpm_num_ind = (int *) Malloc(number_of_parameters * sizeof(int));
    int **mpm_new = NULL;
    int mpm_length = 0;

    std::vector<bool> keep_FOS_element;
    int num_original_conditional_distributions;

    /* Initialize LT to the initial MPM */
    if (learn_conditional_linkage_tree) {
        // If conditional, load in all Uni or MP sets to seed the tree
        mpm_length = distributions.size();
        num_original_conditional_distributions = mpm_length;

        for (distribution_t *d: distributions) {
            conditional_distribution_t *c = (conditional_distribution_t *) d;

            int group_size = c->variable_groups[0].size();
            int *indices = (int *) Malloc(group_size * sizeof(int));
            std::vector<int> vec;

            for (int i = 0; i < c->variable_groups[0].size(); i++) {
                indices[i] = c->variable_groups[0][i];
                vec.push_back(c->variable_groups[0][i]);
            }

            mpm[keep_FOS_element.size()] = indices;
            mpm_num_ind[keep_FOS_element.size()] = group_size;
            keep_FOS_element.push_back(true);
        }
    } else {
        mpm_length = number_of_parameters;
        for (int i = 0; i < number_of_parameters; i++) {
            int *indices = (int *) Malloc(1 * sizeof(int));
            indices[0] = i;
            mpm[i] = indices;
            mpm_num_ind[i] = 1;

            std::vector<int> vec;
            vec.push_back(mpm[i][0]);
            addGroup(vec);
            keep_FOS_element.push_back(true);
        }
    }

    /* Initialize similarity matrix */
    S_matrix = NULL;
    if (!random_linkage_tree) {
        S_matrix = (double **) Malloc(number_of_parameters * sizeof(double *));
        for (int i = 0; i < number_of_parameters; i++)
            S_matrix[i] = (double *) Malloc(number_of_parameters * sizeof(double));
    }

    if (random_linkage_tree) {
        S_vector = (double *) Malloc(number_of_parameters * sizeof(double));
        for (int i = 0; i < number_of_parameters; i++)
            S_vector[i] = randu<double>();

    } else if (static_linkage_tree) {
        MI_matrix = (double **) Malloc(number_of_parameters * sizeof(double *));
        for (int j = 0; j < number_of_parameters; j++)
            MI_matrix[j] = (double *) Malloc(number_of_parameters * sizeof(double));

        if (problem_index == 0) {
            for (int i = 0; i < number_of_parameters; i++) {
                for (int j = 0; j < i; j++) {
                    MI_matrix[i][j] = randu<double>();
                    MI_matrix[j][i] = MI_matrix[j][i];
                }
                MI_matrix[i][i] = 0.0;
            }

        } else if (problem_index == 7) {
            MI_matrix[0][0] = 0.0;
            for (int i = 1; i < number_of_parameters; i++) {
                MI_matrix[i][i] = 0.0;
                MI_matrix[i - 1][i] = 1e8 + randu<double>();
                MI_matrix[i][i - 1] = MI_matrix[i - 1][i];
                for (int j = i + 1; j < number_of_parameters; j++) {
                    MI_matrix[j][i] = 0;
                    MI_matrix[i][j] = MI_matrix[j][i];
                }
            }

        } else if (problem_index == 8) {
            MI_matrix[0][0] = 0.0;
            for (int i = 1; i < number_of_parameters; i++) {
                MI_matrix[i][i] = 0.0;
                for (int j = 0; j < i; j++) {
                    MI_matrix[i][j] = 1e2 * i + randu<double>();
                    MI_matrix[j][i] = MI_matrix[i][j];
                }
            }

        } else if (problem_index == 14) {
            for (int i = 0; i < number_of_parameters; i++) {
                for (int j = 0; j < number_of_parameters; j++) {
                    MI_matrix[j][i] = 0;
                    MI_matrix[i][j] = MI_matrix[j][i];
                }
            }

            int single_block_size = 5;
            int dual_block_size = 2 * 5 - 1;
            for (int i = 0; i + dual_block_size <= number_of_parameters; i += dual_block_size) {
                for (int j = 0; j < single_block_size; j++) {
                    for (int k = 0; k < j; k++) {
                        MI_matrix[i + j][i + k] = 1e8 + randu<double>();
                        MI_matrix[i + k][i + j] = MI_matrix[i + j][i + k];
                    }
                    MI_matrix[i + j][i + j] = 0.0;
                }

                int offset = 4;

                for (int j = 0; j < single_block_size; j++) {
                    for (int k = 0; k < j; k++) {
                        MI_matrix[i + j + offset][i + k + offset] = 1e8 + randu<double>();
                        MI_matrix[i + k + offset][i + j + offset] = MI_matrix[i + j + offset][i + k + offset];
                    }
                    MI_matrix[i + j + offset][i + j + offset] = 0.0;
                }
            }

        } else if ((problem_index == 16) || (problem_index == 17) || (problem_index == 18)) {
            for (int i = 0; i < number_of_parameters; i++) {
                for (int j = 0; j < number_of_parameters; j++) {
                    MI_matrix[j][i] = 0;
                    MI_matrix[i][j] = MI_matrix[j][i];
                }
            }

            int single_block_size = 5;
            for (int i = 0; i < number_of_parameters; i += single_block_size) {
                for (int j = 0; j < single_block_size; j++) {
                    for (int k = 0; k < j; k++) {
                        if ((problem_index == 16) || (problem_index == 17)) {
                            MI_matrix[i + j][i + k] = 1e8 + randu<double>();
                        } else {
                            MI_matrix[i + j][i + k] = 1.0 + randu<double>();
                        }
                        MI_matrix[i + k][i + j] = MI_matrix[i + j][i + k];
                    }
                    MI_matrix[i + j][i + j] = 0.0;
                }

                if (i > 0) {
                    int offset = -1;
                    int small_block_size = 2;
                    for (int j = 0; j < small_block_size; j++) {
                        for (int k = 0; k < j; k++) {
                            if ((problem_index == 16) || (problem_index == 18)) {
                                MI_matrix[i + j - 1][i + k - 1] = 1e8 + randu<double>();
                            } else {
                                MI_matrix[i + j - 1][i + k - 1] = 1.0 + randu<double>();
                            }
                            MI_matrix[i + k - 1][i + j - 1] = MI_matrix[i + j - 1][i + k - 1];
                        }
                        MI_matrix[i + j + offset][i + j + offset] = 0.0;
                    }
                }
            }

        } else if (problem_index == 20) {
            for (int i = 0; i < number_of_parameters; i++) {
                for (int j = 0; j < number_of_parameters; j++) {
                    MI_matrix[j][i] = 0;
                    MI_matrix[i][j] = MI_matrix[j][i];
                }
            }

            int grid_width = round(sqrt(number_of_parameters));
            for (int i = 0; i < number_of_parameters; i++) {
                for (int j = 0; j < number_of_parameters; j++) {
                    if (i == j) {
                        MI_matrix[i][j] = 0.0;
                    } else {
                        int x1 = i % grid_width;
                        int y1 = i / grid_width;
                        int x2 = j % grid_width;
                        int y2 = j / grid_width;

                        if (abs(x1 - x2) == 1 || abs(y1 - y2) == 1 ||
                            (abs(x1 - x2) == 0 && abs(y1 - y2) == 2) || (abs(x1 - x2) == 2 && abs(y1 - y2) == 0)) {
                            MI_matrix[i][j] = 1e8 + randu<double>();
                            MI_matrix[j][i] = MI_matrix[i][j];
                        } else {
                            MI_matrix[i][j] = randu<double>();
                            MI_matrix[j][i] = MI_matrix[i][j];
                        }
                    }
                }
            }

        } else if (problem_index == 13 || problem_index > 10000) {
            for (int i = 0; i < number_of_parameters; i++) {
                for (int j = 0; j < number_of_parameters; j++) {
                    MI_matrix[j][i] = 0;
                    MI_matrix[i][j] = MI_matrix[j][i];
                }
            }

            int id = problem_index;
            // Two rotation angles
            id /= 10;
            id /= 10;
            // Two conditioning numbers
            id /= 10;
            id /= 10;
            int overlap_size = id % 10;
            id /= 10;
            int block_size = id;
            if (problem_index == 13) {
                block_size = 5;
                overlap_size = 0;
            }
            for (int i = 0; i + block_size <= number_of_parameters; i += (block_size - overlap_size)) {
                for (int j = 0; j < block_size; j++) {
                    for (int k = 0; k < j; k++) {
                        MI_matrix[i + j][i + k] = 1e8 + randu<double>();
                        MI_matrix[i + k][i + j] = MI_matrix[i + j][i + k];
                    }
                    MI_matrix[i + j][i + j] = 0.0;
                }
            }

        } else {
            printf("Implement this.\n");
            exit(0);
        }
    }

    for (int i = 0; i < mpm_length; i++) {
        for (int j = 0; j < i; j++) {
            double similarity_between_seeded_elements = 0;

            for (size_t k = 0; k < mpm_num_ind[i]; k++) {
                for (size_t l = 0; l < mpm_num_ind[j]; l++) {
                    if (MI_matrix[mpm[i][k]][mpm[j][l]] > 0.0) {
                        similarity_between_seeded_elements += 1.0;
                    }
                }
            }

            similarity_between_seeded_elements /= (double) mpm_num_ind[i] * (double) mpm_num_ind[j];
            S_matrix[i][j] = similarity_between_seeded_elements + randomRealUniform01() * 0.00001;
            S_matrix[j][i] = S_matrix[i][j];
        }
    }
    for (int i = 0; i < mpm_length; i++) {
        S_matrix[i][i] = 0;
    }

    int *NN_chain = (int *) Malloc((mpm_length + 2) * sizeof(int));
    int NN_chain_length = 0;
    short done = 0;
    while (!done) {
        if (NN_chain_length == 0) {
            NN_chain[NN_chain_length] = randomInt(mpm_length);
            NN_chain_length++;
        }

        if (NN_chain[NN_chain_length - 1] >= mpm_length) NN_chain[NN_chain_length - 1] = mpm_length - 1;

        while (NN_chain_length < 3) {
            NN_chain[NN_chain_length] = determineNearestNeighbour(NN_chain[NN_chain_length - 1], mpm_num_ind,
                                                                  mpm_length);
            NN_chain_length++;
        }

        while (NN_chain[NN_chain_length - 3] != NN_chain[NN_chain_length - 1]) {
            NN_chain[NN_chain_length] = determineNearestNeighbour(NN_chain[NN_chain_length - 1], mpm_num_ind,
                                                                  mpm_length);
            if (((getSimilarity(NN_chain[NN_chain_length - 1], NN_chain[NN_chain_length], mpm_num_ind) ==
                  getSimilarity(NN_chain[NN_chain_length - 1], NN_chain[NN_chain_length - 2], mpm_num_ind)))
                && (NN_chain[NN_chain_length] != NN_chain[NN_chain_length - 2]))
                NN_chain[NN_chain_length] = NN_chain[NN_chain_length - 2];
            NN_chain_length++;
            if (NN_chain_length > number_of_parameters)
                break;
        }
        int r0 = NN_chain[NN_chain_length - 2];
        int r1 = NN_chain[NN_chain_length - 1];

        if (r1 >= mpm_length || r0 >= mpm_length || mpm_num_ind[r0] + mpm_num_ind[r1] > FOS_element_ub) {
            NN_chain_length = 1;
            NN_chain[0] = 0;
            if (FOS_element_ub < number_of_parameters) {
                done = 1;
                for (int i = 1; i < mpm_length; i++) {
                    if (mpm_num_ind[i] + mpm_num_ind[NN_chain[0]] <= FOS_element_ub) done = 0;
                    if (mpm_num_ind[i] < mpm_num_ind[NN_chain[0]]) NN_chain[0] = i;
                }
                if (done) break;
            }
            continue;
        }

        if (r0 > r1) {
            int rswap = r0;
            r0 = r1;
            r1 = rswap;
        }
        NN_chain_length -= 3;

        /* This test is required for exceptional cases in which the nearest-neighbor ordering has changed within the chain while merging within that chain */
        if (r1 < mpm_length && r1 != r0) {
            int *indices = (int *) Malloc((mpm_num_ind[r0] + mpm_num_ind[r1]) * sizeof(int));

            int k = 0;
            for (int j = 0; j < mpm_num_ind[r0]; j++) {
                indices[k] = mpm[r0][j];
                k++;
            }
            for (int j = 0; j < mpm_num_ind[r1]; j++) {
                indices[k] = mpm[r1][j];
                k++;
            }

            // Determine based on similarity if this FOS element should be pruned
            // Two elements A and B, merged into C, should only be kept if C isn't also fully connected, similarity-wise
            if (prune_linkage_tree) {
                bool completely_dependent = true;
                for (size_t i = 0; i < mpm_num_ind[r0]; i++) {
                    for (size_t j = 0; j < mpm_num_ind[r1]; j++) {
                        if (MI_matrix[mpm[r0][i]][mpm[r1][j]] <= 0.0) {
                            completely_dependent = false;
                            break;
                        }
                    }
                }

                int current_fos_length = getLength();

                // Remove subsets that build this set
                if (completely_dependent) {
                    // Remove r0
                    int set_length_0 = mpm_num_ind[r0];
                    int set_length_1 = mpm_num_ind[r1];
                    for (size_t i = 0; i < current_fos_length; i++) {
                        if (!keep_FOS_element[i]) {
                            continue;
                        }

                        if (sets[i].size() == set_length_0) {
                            std::set<int> fos_element;
                            for (int set_ind = 0; set_ind < sets[i].size(); set_ind++) {
                                fos_element.insert(sets[i][set_ind]);
                            }

                            bool is_equal_set = true;
                            for (size_t x = 0; x < set_length_0; x++) {
                                if (fos_element.find(mpm[r0][x]) == fos_element.end()) {
                                    is_equal_set = false;
                                    break;
                                }
                            }
                            if (is_equal_set) {
                                keep_FOS_element[i] = false;
                            }
                        }

                        if (sets[i].size() == set_length_1) {
                            std::set<int> fos_element;
                            for (int set_ind = 0; set_ind < sets[i].size(); set_ind++) {
                                fos_element.insert(sets[i][set_ind]);
                            }

                            bool is_equal_set = true;
                            for (size_t x = 0; x < set_length_1; x++) {
                                if (fos_element.find(mpm[r1][x]) == fos_element.end()) {
                                    is_equal_set = false;
                                    break;
                                }
                            }
                            if (is_equal_set) {
                                keep_FOS_element[i] = false;
                            }
                        }
                    }
                }
            }

            std::vector<int> vec;
            int *sorted = mergeSortInt(indices, mpm_num_ind[r0] + mpm_num_ind[r1]);
            for (int j = 0; j < mpm_num_ind[r0] + mpm_num_ind[r1]; j++)
                vec.push_back(indices[sorted[j]]);

            if (learn_conditional_linkage_tree) {
                conditional_distribution_t *cond = new conditional_distribution_t();

                // Determine which sets this merges
                std::set<int> dists_to_merge;
                for (int var: vec) {
                    for (int j = 0; j < num_original_conditional_distributions; j++) {
                        for (int other_var: sets[j]) {
                            if (other_var == var) {
                                dists_to_merge.insert(j);
                                break;
                            }
                        }
                    }
                }

                std::vector<int> dist_list(dists_to_merge.size());
                std::copy(dists_to_merge.begin(), dists_to_merge.end(), dist_list.begin());

                for (int j = 0; j < dist_list.size(); j++) {
                    int dist = dist_list[j];
                    std::vector<int> variable_group = ((conditional_distribution_t *) distributions[dist])->variable_groups[0];
                    std::vector<int> cond_group = ((conditional_distribution_t *) distributions[dist])->variables_conditioned_on[0];

                    std::vector<int> copied_variable_group(variable_group.size());
                    std::copy(variable_group.begin(), variable_group.end(), copied_variable_group.begin());

                    std::vector<int> copied_cond(cond_group.size());
                    std::copy(cond_group.begin(), cond_group.end(), copied_cond.begin());

                    cond->addGroupOfVariables(copied_variable_group, copied_cond);
                }

                addGroup(cond);
            } else {
                addGroup(vec);
            }

            keep_FOS_element.push_back(true);

            free(sorted);
            free(indices);

            double mul0 = ((double) mpm_num_ind[r0]) / ((double) mpm_num_ind[r0] + mpm_num_ind[r1]);
            double mul1 = ((double) mpm_num_ind[r1]) / ((double) mpm_num_ind[r0] + mpm_num_ind[r1]);
            if (random_linkage_tree) {
                S_vector[r0] = mul0 * S_vector[r0] + mul1 * S_vector[r1];
            } else {
                for (int i = 0; i < mpm_length; i++) {
                    if ((i != r0) && (i != r1)) {
                        S_matrix[i][r0] = mul0 * S_matrix[i][r0] + mul1 * S_matrix[i][r1];
                        S_matrix[r0][i] = S_matrix[i][r0];
                    }
                }
            }

            mpm_new = (int **) Malloc((mpm_length - 1) * sizeof(int *));
            int *mpm_new_number_of_indices = (int *) Malloc((mpm_length - 1) * sizeof(int));
            int mpm_new_length = mpm_length - 1;
            for (int i = 0; i < mpm_new_length; i++) {
                mpm_new[i] = mpm[i];
                mpm_new_number_of_indices[i] = mpm_num_ind[i];
            }

            mpm_new[r0] = (int *) Malloc(vec.size() * sizeof(int));
            for (int i = 0; i < vec.size(); i++)
                mpm_new[r0][i] = vec[i];
            mpm_new_number_of_indices[r0] = mpm_num_ind[r0] + mpm_num_ind[r1];
            if (r1 < mpm_length - 1) {
                mpm_new[r1] = mpm[mpm_length - 1];
                mpm_new_number_of_indices[r1] = mpm_num_ind[mpm_length - 1];

                if (random_linkage_tree) {
                    S_vector[r1] = S_vector[mpm_length - 1];
                } else {
                    for (int i = 0; i < r1; i++) {
                        S_matrix[i][r1] = S_matrix[i][mpm_length - 1];
                        S_matrix[r1][i] = S_matrix[i][r1];
                    }

                    for (int j = r1 + 1; j < mpm_new_length; j++) {
                        S_matrix[r1][j] = S_matrix[j][mpm_length - 1];
                        S_matrix[j][r1] = S_matrix[r1][j];
                    }
                }
            }

            for (int i = 0; i < NN_chain_length; i++) {
                if (NN_chain[i] == mpm_length - 1) {
                    NN_chain[i] = r1;
                    break;
                }
            }

            free(mpm[r0]);
            free(mpm);
            free(mpm_num_ind);
            mpm = mpm_new;
            mpm_num_ind = mpm_new_number_of_indices;
            mpm_length = mpm_new_length;

            if (mpm_length == 1)
                done = 1;
        }
    }
    free(NN_chain);

    free(mpm_new);
    free(mpm_num_ind);

    if (random_linkage_tree)
        free(S_vector);
    else {
        for (int i = 0; i < number_of_parameters; i++)
            free(S_matrix[i]);
        free(S_matrix);
    }

    // Prune elements to be removed
    if (prune_linkage_tree) {
        for (int i = getLength() - 1; i >= 0; i--) {
            if (!keep_FOS_element[i]) {
                sets.erase(sets.begin() + i);
                distributions.erase(distributions.begin() + i);
            }
        }
    }

    order = randomPermutation(getLength());

    for (int i = 0; i < number_of_parameters; i++)
        free(MI_matrix[i]);
    free(MI_matrix);
}

fos_t::fos_t(const std::map<int, std::set<int>> &variable_interaction_graph,
             vec_t <vec_t<double>> *fitness_dependency_matrix) {
    assert(include_cliques_as_fos_elements || include_full_fos_element);

    uvec var_order = randomPermutation(number_of_parameters);

    conditional_distribution_t *full_cond;

    if (include_full_fos_element) {
        if (use_conditional_sampling) {
            full_cond = new conditional_distribution_t();
        } else {
            std::vector<int> all_variables;
            for (int i = 0; i < number_of_parameters; i++) {
                all_variables.push_back(i);
            }

            normal_distribution_t *full_non_cond;
            full_non_cond = new normal_distribution_t(all_variables);
            addGroup(full_non_cond);
        }
    }

    if (use_set_cover) {
        std::vector <std::pair<int, int>> candidate_sets;

        std::vector<bool> covered_variables;

        for (int i = 0; i < number_of_parameters; i++) {
            covered_variables.push_back(false);
            for (int x: variable_interaction_graph.at(i)) {
                if (i < x) {
                    candidate_sets.push_back(std::pair<int, int>(i, x));
                }
            }
        }

        while (true) {
            uvec candidate_order = randomPermutation(candidate_sets.size());

            // Determine which set to add next, with maximum added cover
            int max_cover_increase = 0;
            int max_cover_increase_index = -1;
            for (int candidate_index = 0; candidate_index < candidate_sets.size(); candidate_index++) {
                std::pair<int, int> candidate = candidate_sets[candidate_order[candidate_index]];

                int cover_increase = 0;
                if (!covered_variables[candidate.first]) {
                    cover_increase++;
                }
                if (!covered_variables[candidate.second]) {
                    cover_increase++;
                }

                if (cover_increase > max_cover_increase) {
                    max_cover_increase = cover_increase;
                    max_cover_increase_index = candidate_index;
                }

                if (cover_increase == 2) {
                    break;
                }
            }

            assert(max_cover_increase > 0);

            std::pair<int, int> candidate = candidate_sets[candidate_order[max_cover_increase_index]];
            covered_variables[candidate.first] = true;
            covered_variables[candidate.second] = true;

            std::vector<int> variables;
            variables.push_back(candidate.first);
            variables.push_back(candidate.second);

            std::set<int> conditioned_variables;

            for (int x: variable_interaction_graph.at(candidate.first)) {
                if (x == candidate.first || x == candidate.second) {
                    continue;
                }
                conditioned_variables.insert(x);
            }
            for (int x: variable_interaction_graph.at(candidate.second)) {
                if (x == candidate.first || x == candidate.second) {
                    continue;
                }

                conditioned_variables.insert(x);
            }

            if (use_conditional_sampling) {
                addConditionedGroup(variables, conditioned_variables);
            } else {
                addGroup(variables);
            }

            candidate_sets.erase(candidate_sets.begin() + candidate_order[max_cover_increase_index]);

            // Check if all variables are covered
            bool some_uncovered = false;
            for (int i = 0; i < number_of_parameters; i++) {
                if (!covered_variables[i]) {
                    some_uncovered = true;
                }
            }
            if (!some_uncovered) {
                break;
            }
        }
    }

    if (seed_cliques_per_variable) {
        // Approach that seeds clique searches at each variable

        std::vector <std::vector<int>> clique_candidate_list;
        std::vector <std::set<int>> cond_candidate_list;
        for (int i = 0; i < number_of_parameters; i++) {
            std::vector<int> clique;
            std::set<int> cond_candidates;

            int ind = var_order[i];
            clique.push_back(ind);

            for (int x: variable_interaction_graph.at(ind)) // neighbors of ind
            {
                cond_candidates.insert(x);

                bool add_to_clique = true;
                std::set<int> neighbors = variable_interaction_graph.at(x);
                if (clique.size() >= max_clique_size) {
                    add_to_clique = false;
                }

                if (add_to_clique) {
                    for (int y: clique) {
                        if (neighbors.find(y) == neighbors.end()) // edge (x,y) does not exist
                        {
                            add_to_clique = false;
                            break;
                        }
                    }
                }
                if (add_to_clique) {
                    clique.push_back(x);

                    cond_candidates.insert(x);
                    for (int y: neighbors) {
                        cond_candidates.insert(y);
                    }
                }
            }

            std::set<int> cond;
            for (int c: cond_candidates) {
                if (std::find(clique.begin(), clique.end(), c) == clique.end()) {
                    cond.insert(c);
                }
            }

            std::sort(clique.begin(), clique.end());
            clique_candidate_list.push_back(clique);
            cond_candidate_list.push_back(cond);
        }

        std::vector<bool> is_already_excluded;
        for (int clique_a = 0; clique_a < number_of_parameters; clique_a++) {
            bool clique_is_not_unique = false;
            for (int clique_b = 0; clique_b < number_of_parameters; clique_b++) {
                if (clique_a == clique_b || (is_already_excluded.size() > clique_b && is_already_excluded[clique_b])) {
                    continue;
                }

                clique_is_not_unique |= std::includes(
                        clique_candidate_list[clique_b].begin(), clique_candidate_list[clique_b].end(),
                        clique_candidate_list[clique_a].begin(), clique_candidate_list[clique_a].end());

                if (clique_is_not_unique) {
                    break;
                }
            }

            is_already_excluded.push_back(clique_is_not_unique);

            if (!clique_is_not_unique) {
                if (include_cliques_as_fos_elements) {
                    if (use_conditional_sampling) {
                        addConditionedGroup(clique_candidate_list[clique_a], cond_candidate_list[clique_a]);
                    } else {
                        addGroup(clique_candidate_list[clique_a]);
                    }
                }
            }
        }

#if CHECK_FOS_COMPLETENESS_AFTER_CLIQUE_BUILDING
        bool *variable_is_in_fos = new bool[number_of_parameters];

        for (int i = 0; i < number_of_parameters; i++) {
            variable_is_in_fos[i] = false;
        }

        for (int i = 0; i < sets.size(); i++) {
            for (int j = 0; j < sets[i].size(); j++) {
                variable_is_in_fos[sets[i][j]] = true;
            }
        }

        for (int i = 0; i < number_of_parameters; i++) {
            assert(variable_is_in_fos[i]);
        }

        delete[] variable_is_in_fos;
#endif

    }

    // Conventional VIG search
    const int UNVISITED = 0;
    const int IS_VISITED = 1;
    const int IN_CLIQUE = 2;
    const int IN_QUEUE = 3;
    int visited[number_of_parameters];
    for (int i = 0; i < number_of_parameters; i++) {
        visited[i] = UNVISITED;
    }

    int conventional_max_clique_size = max_clique_size;
    if (seed_cliques_per_variable == 2) {
        conventional_max_clique_size = 1;
    }

    std::vector<int> fitness_based_permutation(number_of_parameters);

    if (similarity_measure == 'F' && fitness_based_ordering) {
        std::vector<double> aggregate_fitness_strength(number_of_parameters);
        for (int i = 0; i < number_of_parameters; i++) {
            aggregate_fitness_strength[i] = 0.0;
            for (int j = 0; j < i; j++) {
                aggregate_fitness_strength[i] -= (*fitness_dependency_matrix)[i][j];
            }
            if (i > 0) {
                aggregate_fitness_strength[i] /= (double) i;
            }
        }

        iota(fitness_based_permutation.begin(), fitness_based_permutation.end(), 0);
        std::stable_sort(fitness_based_permutation.begin(), fitness_based_permutation.end(),
                         [&aggregate_fitness_strength](size_t i1, size_t i2) {
                             return aggregate_fitness_strength[i1] < aggregate_fitness_strength[i2];
                         });
    }

    for (int i = 0; i < number_of_parameters; i++) {
        int ind;

        if (fitness_based_ordering) {
            ind = fitness_based_permutation[i];
        } else {
            ind = var_order[i];
        }

        if (visited[ind] == IS_VISITED)
            continue;
        visited[ind] = IN_CLIQUE;

        std::queue<int> q;
        q.push(ind);

        while (!q.empty()) {
            ind = q.front();
            q.pop();

            if (visited[ind] == IS_VISITED)
                continue;
            visited[ind] = IS_VISITED;

            std::vector<int> clique;
            std::set<int> cond;
            clique.push_back(ind);
            for (int x: variable_interaction_graph.at(ind)) // neighbors of ind
            {
                if (visited[x] == IS_VISITED)
                    cond.insert(x);
            }

            for (int x: variable_interaction_graph.at(ind)) // neighbors of ind
            {
                if (visited[x] != IS_VISITED) {
                    bool add_to_clique = true;
                    std::set<int> neighbors = variable_interaction_graph.at(x);
                    if (clique.size() >= conventional_max_clique_size) {
                        add_to_clique = false;
                    }
                    if (add_to_clique) {
                        for (int y: clique) {
                            if (neighbors.find(y) == neighbors.end()) // edge (x,y) does not exist
                            {
                                add_to_clique = false;
                                break;
                            }
                        }
                    }
                    if (add_to_clique) {
                        for (int y: cond) {
                            if (neighbors.find(y) == neighbors.end()) // edge (x,y) does not exist
                            {
                                add_to_clique = false;
                                break;
                            }
                        }
                    }
                    if (add_to_clique) {
                        clique.push_back(x);
                    }
                }
            }

            for (int x: clique) {
                visited[x] = IS_VISITED;
                for (int y: variable_interaction_graph.at(x)) // neighbors of x
                {
                    if (visited[y] == UNVISITED) {
                        q.push(y);
                        visited[y] = IN_QUEUE;
                    }
                }
            }

            // Hybridize CS with UCond FOS elements
            if (seed_cliques_per_variable == 2) {
                assert(use_conditional_sampling);
                bool insert = true;
                for (distribution_t *d: distributions) {
                    conditional_distribution_t *c = (conditional_distribution_t *) d;

                    if (c->variable_groups[0].size() > 1) {
                        continue;
                    }
                    if (c->variable_groups[0][0] == clique[0]) {
                        insert = false;
                    }
                }
                if (insert) {
                    addConditionedGroup(clique, cond);
                }
            }

            if (!seed_cliques_per_variable && !use_set_cover && include_cliques_as_fos_elements) {
                if (use_conditional_sampling) {
                    addConditionedGroup(clique, cond);
                } else {
                    addGroup(clique);
                }
            }

            if (include_full_fos_element && use_conditional_sampling) {
                full_cond->addGroupOfVariables(clique, cond);
            }
        }
    }

    if (learn_conditional_linkage_tree) {
        delete full_cond;

        if (similarity_measure == 'F') {
            double **array_version = (double **) Malloc(number_of_parameters * sizeof(double *));
            for (int i = 0; i < number_of_parameters; i++) {
                array_version[i] = (double *) Malloc(number_of_parameters * sizeof(double));
                for (int j = 0; j < i; j++) {
                    array_version[i][j] = (*fitness_dependency_matrix)[i][j] > 0 ? 1e8 + randomRealUniform01() : 0;
                    array_version[j][i] = array_version[i][j];
                }
            }

            deriveTree(array_version);
        } else {
            static_linkage_tree = true;
            deriveTree(NULL);
        }
    }

    if (include_full_fos_element && use_conditional_sampling && !learn_conditional_linkage_tree) {
        if (getLength() != 1) {
            // if length == 1, only 1 clique was found, which must have been the full model;
            // in that case, do not add it again
            addGroup(full_cond);
        } else {
            delete full_cond;
        }
    }

    // Uncomment to print conditional distributions
//    if (write_fitness_dependencies) {
//        for (distribution_t *d : distributions) {
//            conditional_distribution_t *c = (conditional_distribution_t *) d;
//            for (int g = 0; g < c->variable_groups.size(); g++) {
//                printf("{");
//                for (int var : c->variable_groups[g]) {
//                    printf("%d,", var+1);
//                }
//                printf("}|(");
//                for (int var : c->variables_conditioned_on[g]) {
//                    printf("%d,", var+1);
//                }
//                printf(") ");
//            }
//            printf("|| \n");
//        }
//        printf("\n\n");
//    }

    // Uncomment to print VIG
//    for (int i = 0; i < number_of_parameters; i++) {
//        printf("[%d] ", i);
//
//        for (int j: variable_interaction_graph.at(i)) {
//            printf("%d, ", j);
//        }
//        printf("\n");
//    }
//    printf("\n");

    // Write set cover results
    if (use_set_cover && write_fitness_dependencies) {
        FILE *f = fopen("set_cover_fos.dat", "w");

        for (size_t i = 0; i < sets.size(); i++) {
            fprintf(f, "[");
            int c = 0;
            for (int v: sets[i]) {
                if (c == sets[i].size() - 1)
                    fprintf(f, "%d", v);
                else
                    fprintf(f, "%d^", v);
                c++;
            }
            fprintf(f, "]");
            if (i < sets.size() - 1) {
                fprintf(f, "|");
            }
        }
        fprintf(f, "\n");

        fclose(f);
    }
}

fos_t::~fos_t() {
    for (auto d: distributions)
        delete (d);
}

int fos_t::getLength() {
    return (sets.size());
}

std::vector<int> fos_t::getSet(int element_index) {
    return (sets[element_index]);
}

int fos_t::getSetLength(int element_index) {
    return (sets[element_index].size());
}

double fos_t::getDistributionMultiplier(int element_index) {
    return (distributions[element_index]->distribution_multiplier);
}

double fos_t::getAcceptanceRate() {
    return (p_accept);
}

void fos_t::addGroup(int var_index) {
    std::vector<int> vec;
    vec.push_back(var_index);
    addGroup(vec);
}

void fos_t::addGroup(const std::set<int> &group) {
    std::vector<int> vec;
    for (int x: group)
        vec.push_back(x);
    addGroup(vec);
}

void fos_t::addGroup(std::vector<int> group) {
    std::sort(group.begin(), group.end());
    sets.push_back(group);
    distributions.push_back(new normal_distribution_t(group));
}

void fos_t::addGroup(distribution_t *dist) {
    //std::sort(dist->variables.begin(),dist->variables.end());
    sets.push_back(dist->variables);
    distributions.push_back(dist);
}

void fos_t::addConditionedGroup(std::vector<int> variables) {
    std::set<int> cond;
    addConditionedGroup(variables, cond);
}

void fos_t::addConditionedGroup(std::vector<int> variables, std::set<int> conditioned_variables) {
    std::sort(variables.begin(), variables.end());
    sets.push_back(variables);
    conditional_distribution_t *dist = new conditional_distribution_t(variables, conditioned_variables);
    distributions.push_back(dist);
}

double fos_t::getSimilarity(int a, int b, int *mpm_num_ind) {
    if (FOS_element_ub < number_of_parameters && mpm_num_ind[a] + mpm_num_ind[b] > FOS_element_ub) return (0);
    if (random_linkage_tree) return (1.0 - fabs(S_vector[a] - S_vector[b]));
    return (S_matrix[a][b]);
}

int fos_t::determineNearestNeighbour(int index, int *mpm_num_ind, int mpm_length) {
    int result = 0;
    if (result == index)
        result++;
    for (int i = 1; i < mpm_length; i++) {
        if (((getSimilarity(index, i, mpm_num_ind) > getSimilarity(index, result, mpm_num_ind)) ||
             ((getSimilarity(index, i, mpm_num_ind) == getSimilarity(index, result, mpm_num_ind)) &&
              (mpm_num_ind[i] < mpm_num_ind[result]))) && (i != index))
            result = i;
    }

    return (result);
}

void fos_t::randomizeOrder() {
    int index_of_full_element = -1;
    for (int i = 0; i < getLength(); i++) {
        if (sets[i].size() == number_of_parameters) {
            index_of_full_element = i;
            break;
        }
    }

    order = randomPermutation(getLength());

    if (index_of_full_element != -1) {
        int loc = -1;
        for (int i = 0; i < getLength(); i++) {
            if (order[i] == index_of_full_element) {
                loc = i;
                break;
            }
        }

        if (loc != getLength() - 1) {
            order[loc] = order[0];
            order[0] = index_of_full_element;
        }
    }
}

void fos_t::randomizeOrder(const std::map<int, std::set<int>> &variable_interaction_graph) {
    int visited[number_of_parameters];
    for (int i = 0; i < number_of_parameters; i++) {
        visited[i] = 0;
    }

    std::vector<int> VIG_order = getVIGOrderBreadthFirst(variable_interaction_graph);

    /*printf("VIG_ORDER: ");
    for(int i = 0; i < VIG_order.size(); i++ )
        printf("%d ",VIG_order[i]);
    printf("\n");*/

    int FOS_length = 0;
    order = uvec(getLength(), fill::none);
    if (include_cliques_as_fos_elements) {
        for (int i = 0; i < number_of_parameters; i++) {
            assert(sets[i][0] == i);
            assert(getSetLength(i) == 1);
            order[i] = VIG_order[i];
            distributions[order[i]]->updateConditionals(variable_interaction_graph, visited);
        }
        FOS_length = number_of_parameters;
    }
    if (include_full_fos_element) {
        order[FOS_length] = number_of_parameters;
        distributions[FOS_length]->setOrder(VIG_order);
        distributions[FOS_length]->updateConditionals(variable_interaction_graph, visited);
    }
}

std::vector<int> fos_t::getVIGOrderBreadthFirst(const std::map<int, std::set<int>> &variable_interaction_graph) {
    const int UNVISITED = 0;
    const int IS_VISITED = 1;
    const int IN_CLIQUE = 2;
    const int IN_QUEUE = 3;
    int visited[number_of_parameters];
    for (int i = 0; i < number_of_parameters; i++) {
        visited[i] = UNVISITED;
    }

    uvec var_order = randomPermutation(number_of_parameters);

    std::vector<int> VIG_order;
    for (int i = 0; i < number_of_parameters; i++) {
        int ind = var_order[i];
        if (visited[ind] == IS_VISITED)
            continue;
        visited[ind] = IN_CLIQUE;

        std::queue<int> q;
        q.push(ind);

        while (!q.empty()) {
            ind = q.front();
            q.pop();

            if (visited[ind] == IS_VISITED)
                continue;
            visited[ind] = IS_VISITED;

            VIG_order.push_back(ind);

            for (int x: variable_interaction_graph.at(ind)) {
                if (visited[x] == UNVISITED) {
                    q.push(x);
                    visited[x] = IN_QUEUE;
                }
            }
        }
    }
    return (VIG_order);
}

double **fos_t::computeMIMatrix(double **covariance_matrix, int n) {
    double **MI_matrix = (double **) Malloc(n * sizeof(double *));
    for (int j = 0; j < n; j++)
        MI_matrix[j] = (double *) Malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        MI_matrix[i][i] = 1e20;
        for (int j = 0; j < i; j++) {
            double si = sqrt(covariance_matrix[i][i]);
            double sj = sqrt(covariance_matrix[j][j]);
            double r = covariance_matrix[i][j] / (si * sj);
            MI_matrix[i][j] = log(sqrt(1 / (1 - r * r)));
            MI_matrix[j][i] = MI_matrix[i][j];
        }
    }
    return (MI_matrix);
}

void fos_t::print() {
    printf("{");
    for (int i = 0; i < getLength(); i++) {
        printf("[");
        for (int j = 0; j < sets[i].size(); j++) {
            printf("%d", sets[i][j]);
            if (j != sets[i].size() - 1)
                printf(",");
        }
        printf("]");
        printf(",");
    }
    printf("}\n");
}

partial_solution_t *fos_t::generatePartialSolution(int FOS_index, solution_t *solution_conditioned_on) {
    return (distributions[FOS_index]->generatePartialSolution(solution_conditioned_on));
}

void fos_t::estimateDistributions(solution_t **selection, int selection_size,
                                  vec_t <vec_t<double>> fitness_dependency_matrix) {
    for (int i = 0; i < getLength(); i++)
        estimateDistribution(i, selection, selection_size, fitness_dependency_matrix);
    order = randomPermutation(getLength());
}

void fos_t::estimateDistribution(int FOS_index, solution_t **selection, int selection_size,
                                 vec_t <vec_t<double>> fitness_dependency_matrix) {
    distributions[FOS_index]->estimateDistribution(selection, selection_size, fitness_dependency_matrix);
}

void fos_t::adaptDistributionMultiplier(int FOS_index, partial_solution_t **solutions, int num_solutions) {
    if (no_improvement_stretch >= maximum_no_improvement_stretch)
        distributions[FOS_index]->adaptDistributionMultiplierMaximumStretch(solutions, num_solutions);
    else
        distributions[FOS_index]->adaptDistributionMultiplier(solutions, num_solutions);
}

