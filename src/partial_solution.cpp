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

#include "partial_solution.h"

partial_solution_t::partial_solution_t(int num_touched_variables) {
    this->num_touched_variables = num_touched_variables;
    this->touched_variables = vec(num_touched_variables, fill::none);
    this->sample_zs = zeros<vec>(num_touched_variables);
    objective_value = 1e308;
    constraint_value = 1e308;
}

partial_solution_t::partial_solution_t(vec touched_variables, std::vector<int> &touched_indices) {
    this->num_touched_variables = touched_variables.n_elem;
    this->touched_indices = touched_indices;
    this->touched_variables = touched_variables;
    this->sample_zs = zeros<vec>(num_touched_variables);
    objective_value = 1e308;
    constraint_value = 1e308;
}

partial_solution_t::partial_solution_t(vec touched_variables, vec sample_zs, std::vector<int> &touched_indices) {
    this->num_touched_variables = touched_variables.n_elem;
    this->touched_indices = touched_indices;
    this->touched_variables = touched_variables;
    this->sample_zs = sample_zs;
    objective_value = 1e308;
    constraint_value = 1e308;
}

partial_solution_t::partial_solution_t(partial_solution_t &other) {
    this->num_touched_variables = other.num_touched_variables;
    for (int i = 0; i < num_touched_variables; i++)
        this->touched_indices.push_back(other.touched_indices[i]);
    this->touched_variables = other.touched_variables;
    this->sample_zs = other.sample_zs;
    objective_value = other.objective_value;
    constraint_value = other.constraint_value;
}

int partial_solution_t::getTouchedIndex(int ind) {
    if (touched_index_map.empty()) {
        for (int i = 0; i < num_touched_variables; i++)
            touched_index_map[touched_indices[i]] = i;
    }

    auto map_ind = touched_index_map.find(ind);
    if (map_ind == touched_index_map.end())
        return (-1);
    else
        return map_ind->second;
}

void partial_solution_t::setSampleMean(vec means) {
    this->sample_means = means;
}

void partial_solution_t::print() {
    for (int i = 0; i < num_touched_variables; i++)
        printf("[%d][%6.3e]", touched_indices[i], touched_variables[i]);
}
