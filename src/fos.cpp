/**
 *
 * RV-GOMEA
 *
 * If you use this software for any purpose, please cite the most recent publication:
 * A. Bouter, C. Witteveen, T. Alderliesten, P.A.N. Bosman. 2017.
 * Exploiting Linkage Information in Real-Valued Optimization with the Real-Valued
 * Gene-pool Optimal Mixing Evolutionary Algorithm. In Proceedings of the Genetic 
 * and Evolutionary Computation Conference (GECCO 2017).
 * DOI: 10.1145/3071178.3071272
 *
 * Copyright (c) 1998-2017 Peter A.N. Bosman
 *
 * The software in this file is the proprietary information of
 * Peter A.N. Bosman.
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
 * The software in this file is the result of (ongoing) scientific research.
 * The following people have been actively involved in this research over
 * the years:
 * - Peter A.N. Bosman
 * - Dirk Thierens
 * - Jörn Grahl
 * - Anton Bouter
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


fos_t::fos_t(vec_t<vec_t<double>> fitness_dependency_matrix) {
    assert(learn_linkage_tree);

    double **array_version = (double **) Malloc(number_of_parameters * sizeof(double *));
    for (int i = 0; i < number_of_parameters; i++) {
        array_version[i] = (double *) Malloc(number_of_parameters * sizeof(double));
        for (int j = 0; j < number_of_parameters; j++) {
            array_version[i][j] = fitness_dependency_matrix[i][j];
        }
    }

    deriveTree(array_version);
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

void fos_t::deriveTree(double** MI_matrix) {
    /* Initialize MPM to the univariate factorization */
    int **mpm = (int **) Malloc(number_of_parameters * sizeof(int *));
    int *mpm_num_ind = (int *) Malloc(number_of_parameters * sizeof(int));
    int mpm_length = number_of_parameters;
    int **mpm_new = NULL;
    for (int i = 0; i < number_of_parameters; i++) {
        int *indices = (int *) Malloc(1 * sizeof(int));
        indices[0] = i;
        mpm[i] = indices;
        mpm_num_ind[i] = 1;
    }

    std::vector<bool> keep_FOS_element;

    /* Initialize LT to the initial MPM */
    if (problem_index != 14) {
        for (int i = 0; i < mpm_length; i++) {
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

    if (learn_linkage_tree) {
        for (int i = 0; i < mpm_length; i++)
            for (int j = 0; j < mpm_length; j++)
                S_matrix[i][j] = MI_matrix[mpm[i][0]][mpm[j][0]];
        for (int i = 0; i < mpm_length; i++)
            S_matrix[i][i] = 0;

        for (int i = 0; i < number_of_parameters; i++)
            free(MI_matrix[i]);
        free(MI_matrix);

    } else if (random_linkage_tree) {
        S_vector = (double *) Malloc(number_of_parameters * sizeof(double));
        for (int i = 0; i < number_of_parameters; i++)
            S_vector[i] = randu<double>();

    } else if (static_linkage_tree) {
        if (problem_index == 0) {
            random_linkage_tree = 1;
            S_vector = (double *) Malloc(number_of_parameters * sizeof(double));
            for (int i = 0; i < number_of_parameters; i++)
                S_vector[i] = randu<double>();
        } else if (problem_index == 7) {
            S_matrix[0][0] = 0.0;
            for (int i = 1; i < number_of_parameters; i++) {
                S_matrix[i][i] = 0.0;
                S_matrix[i - 1][i] = 1e8 + randu<double>();
                S_matrix[i][i - 1] = S_matrix[i - 1][i];
                for (int j = i + 1; j < number_of_parameters; j++) {
                    S_matrix[j][i] = randu<double>();
                    S_matrix[i][j] = S_matrix[j][i];
                }
            }

        } else if (problem_index == 14) {
            for (int i = 0; i < number_of_parameters - 2; i += 2) {
                S_matrix[i][i] = 0.0;
                S_matrix[i + 1][i + 1] = 0.0;
                S_matrix[i][i + 1] = 100 * number_of_parameters;
                S_matrix[i + 1][i] = 100 * number_of_parameters;
                for (int j = i + 2; j < number_of_parameters; j++) {
                    S_matrix[i][j] = 0.0;
                    S_matrix[i + 1][j] = 0.0;
                    S_matrix[j][i] = S_matrix[i][j];
                    S_matrix[j][i + 1] = S_matrix[i + 1][j];
                }
            }

        } else if (problem_index == 13 || problem_index > 1000) {
            int id = problem_index;
            double rotation_angle = (id % 10) * 5;
            id /= 10;
            double conditioning_number = id % 10;
            id /= 10;
            int overlap_size = id % 10;
            id /= 10;
            int block_size = id;
            if (problem_index == 13) {
                block_size = 5;
                overlap_size = 0;
                conditioning_number = 6;
                rotation_angle = 45;
            }
            for (int i = 0; i + block_size <= number_of_parameters; i += (block_size - overlap_size)) {
                for (int j = 0; j < block_size; j++) {
                    for (int k = 0; k < j; k++) {
                        S_matrix[i + j][i + k] = 1e8 + randu<double>();
                        S_matrix[i + k][i + j] = S_matrix[i + j][i + k];
                    }
                    for (int k = j; i + k < number_of_parameters; k++) {
                        S_matrix[i + j][i + k] = randu<double>();
                        S_matrix[i + k][i + j] = S_matrix[i + j][i + k];
                    }
                    S_matrix[i + j][i + j] = 0.0;
                }
            }

        } else {
            printf("Implement this.\n");
            exit(0);
        }
    }

    int *NN_chain = (int *) Malloc((number_of_parameters + 2) * sizeof(int));
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
        if (r1 < mpm_length && r1 != r0)
        {
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
            if( prune_linkage_tree )
            {
                bool completely_dependent = true;
                for (size_t i = 0; i < mpm_num_ind[r0]; i++) {
                    for (size_t j = 0; j < mpm_num_ind[r1]; j++) {
                        if (S_matrix[mpm[r0][i]][mpm[r1][j]] <= 0.0) {
                            completely_dependent = false;
                            break;
                        }
                    }
                }

                int current_fos_length = getLength();

                // Remove subsets that build this set
                if (completely_dependent) {
                    // Remove r0
                    int set_length = mpm_num_ind[r0];
                    for (size_t i = 0; i < current_fos_length; i++) {
                        if (sets[i].size() == set_length) {
                            bool is_equal_set = true;
                            for (size_t x = 0; x < set_length; x++) {
                                if (mpm[r0][x] != sets[i][x]) {
                                    is_equal_set = false;
                                    break;
                                }
                            }
                            if (is_equal_set) {
                                keep_FOS_element[i] = false;
                            }
                        }
                    }

                    // Remove r1
                    set_length = mpm_num_ind[r1];
                    for (size_t i = 0; i < current_fos_length; i++) {
                        if (sets[i].size() == set_length) {
                            bool is_equal_set = true;
                            for (size_t x = 0; x < set_length; x++) {
                                if (mpm[r1][x] != sets[i][x]) {
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
            addGroup(vec);
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
}

fos_t::fos_t(const std::map<int, std::set<int>> &variable_interaction_graph) {
    this->is_conditional = true;

    assert(include_cliques_as_fos_elements || include_full_fos_element);

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
    conditional_distribution_t *full_cond;
    if (include_full_fos_element)
        full_cond = new conditional_distribution_t();
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

            std::vector<int> clique;
            std::set<int> cond;
            clique.push_back(ind);
            for (int x: variable_interaction_graph.at(ind)) // neighbors of ind
            {
                if (visited[x] == IS_VISITED)
                    cond.insert(x);
            }

            /*printf("VIG:\n");
            for( int x : variable_interaction_graph.at(ind) )
                printf("%d,",x);
            printf("\n");*/
            for (int x: variable_interaction_graph.at(ind)) // neighbors of ind
            {
                if (visited[x] != IS_VISITED) {
                    bool add_to_clique = true;
                    std::set<int> neighbors = variable_interaction_graph.at(x);
                    if (clique.size() >= max_clique_size)
                        add_to_clique = false;
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
                    if (add_to_clique)
                        clique.push_back(x);
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
            if (include_cliques_as_fos_elements)
                addConditionedGroup(clique, cond);
            if (include_full_fos_element)
                full_cond->addGroupOfVariables(clique, cond);
        }
    }
    if (include_full_fos_element) {
        if (getLength() !=
            1) // if length == 1, only 1 clique was found, which must have been the full model; in that case, do not add it again
            addGroup(full_cond);
        else
            delete full_cond;
    }
    /*print();
    for( auto d : distributions )
        d->print();*/
    //exit(0);
}


/*fos_t::fos_t( const std::map<int,std::set<int>> &variable_interaction_graph, int max_clique_size, bool include_cliques_as_fos_elements, bool include_full_fos_element, int VIG_order )
{
	this->include_cliques_as_fos_elements = include_cliques_as_fos_elements;
	this->include_full_fos_element = include_full_fos_element;
	this->is_conditional = true;
	assert( include_cliques_as_fos_elements || include_full_fos_element );
	if( include_cliques_as_fos_elements )
	{
		for(int i = 0; i < number_of_parameters; i++ )
		{
			std::vector<int> vars;
			vars.push_back(i);
			addConditionedGroup( vars );
		}
	}
	if( include_full_fos_element )
	{
		std::vector<int> vars;
		for(int i = 0; i < number_of_parameters; i++ )
			vars.push_back(i);
		addConditionedGroup( vars );
	}

	randomizeOrder( variable_interaction_graph );
}*/

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
    order = randomPermutation(getLength());
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
        //distributions[FOS_length]->print();
    }
    /*if( getLength() > 1 )
    {
        printf("ORDER: ");
        for(int i = 0; i < getLength(); i++ )
            printf("%d ",order[i]);
        printf("\n");
    }*/
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
                    //printf("Q[ %d ]\n",x);
                }
            }
        }
    }
    return (VIG_order);
}

/*{
	order = randomPermutation( getLength() );
	
	int visited[number_of_parameters]{};

	for( int i = 0; i < getLength(); i++ )
	{
		int group_index = order[i];
		std::vector<int> clique = getSet(group_index);
		if( getSetLength(group_index) == variable_interaction_graph.size() )
			continue;
	
		distributions[group_index]->updateConditionals( variable_interaction_graph, visited );
	}
}*/

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

void fos_t::estimateDistributions(solution_t **selection, int selection_size, vec_t<vec_t<double>> fitness_dependency_matrix) {
    for (int i = 0; i < getLength(); i++)
        estimateDistribution(i, selection, selection_size, fitness_dependency_matrix);
    order = randomPermutation(getLength());
}

void fos_t::estimateDistribution(int FOS_index, solution_t **selection, int selection_size, vec_t<vec_t<double>> fitness_dependency_matrix) {
    distributions[FOS_index]->estimateDistribution(selection, selection_size, fitness_dependency_matrix);
}

void fos_t::adaptDistributionMultiplier(int FOS_index, partial_solution_t **solutions, int num_solutions) {
    if (no_improvement_stretch >= maximum_no_improvement_stretch)
        distributions[FOS_index]->adaptDistributionMultiplierMaximumStretch(solutions, num_solutions);
    else
        distributions[FOS_index]->adaptDistributionMultiplier(solutions, num_solutions);
}

