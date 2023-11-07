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

#pragma once

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include "optimization.h"
#include "tools.h"
#include "fitness_buffer.h"
#include "solution.h"
#include "partial_solution.h"
#include <deque>
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


class surrogate_t 
{
	public:
		int number_of_variables;
		int num_model_parameters;
		double *weights;
		double error = 0.0;
		VEC model_parameters;

		short replace_fitness = 0;

		double surrogate_time = 0.0;

		~surrogate_t();

		static VEC solveLinearEquations( MAT A, VEC b );
		
		virtual short checkForImprovement( solution_t *solution, partial_solution_t *part ) = 0;
		virtual double evaluate( solution_t *solution );
		
		void printModelParameters();
		void printModelError();
		void writeData(std::string filename);

	//protected:
		MAT A;
		VEC b;
};

class surrogate_lin_t : public surrogate_t
{
	public:
		surrogate_lin_t( solution_t **population, int population_size, int number_of_variables );

		short checkForImprovement( solution_t *solution, partial_solution_t *part );

};
class surrogate_quad_t : public surrogate_t
{
	public:
		surrogate_quad_t( solution_t **population, int population_size, int number_of_variables );

		short checkForImprovement( solution_t *solution, partial_solution_t *part );

};

class surrogate_linquad_t : public surrogate_t
{
	public:
		surrogate_linquad_t( solution_t **population, int population_size, int number_of_variables );
		surrogate_linquad_t( std::deque<solution_t> data, int number_of_variables );
		surrogate_linquad_t( solution_t **parents, partial_solution_t **samples, int sample_size, int number_of_variables );

		short checkForImprovement( solution_t *solution, partial_solution_t *part );

};

class surrogate_chain_t : public surrogate_t
{
	public:
		surrogate_chain_t( solution_t **population, int population_size, int number_of_variables );

		short checkForImprovement( solution_t *solution, partial_solution_t *part );

};

class surrogate_blocks_t : public surrogate_t
{
	public:
		surrogate_blocks_t( std::deque<solution_t> data, int number_of_variables, int block_size );

		int block_size, num_blocks;

		short checkForImprovement( solution_t *solution, partial_solution_t *part );
		double evaluate( solution_t *solution );
};

class surrogate_full_t : public surrogate_t
{
	public:
		surrogate_full_t( solution_t **population, int population_size, int number_of_variables );
		surrogate_full_t( std::deque<solution_t> data, int number_of_variables );

		short checkForImprovement( solution_t *solution, partial_solution_t *part );

};

class surrogate_rbf_t : public surrogate_t
{
	public:
		double eps = 0.1;
		double **centroids;

		surrogate_rbf_t( solution_t **population, int population_size, int number_of_variables );
		~surrogate_rbf_t();
		
		short checkForImprovement( solution_t *solution, partial_solution_t *part );
		double phi( double *c, double *x );
};

