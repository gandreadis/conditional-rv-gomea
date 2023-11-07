#include "surrogate.h"

surrogate_t::~surrogate_t()
{
	delete weights;
}

VEC surrogate_t::solveLinearEquations( MAT A, VEC b )
{
#ifdef EIGEN
	return( A.bdcSvd(ComputeThinU | ComputeThinV).solve(b) );
	//return( A.fullPivHouseholderQr().solve(b) );
	//return( A.colPivHouseholderQr().solve(b) );
	//return( A.householderQr().solve(b) );
#elif defined ARMADILLO
	return( solve( A, b ) );
#endif
}
		
double surrogate_t::evaluate( solution_t *solution )
{
	return( 1e308 );
}

surrogate_lin_t::surrogate_lin_t( solution_t **population, int population_size, int number_of_variables )
{
	//double cur_time = getTimer();
	this->number_of_variables = number_of_variables;
	this->num_model_parameters = number_of_variables + 1;

	weights = new double[num_model_parameters];
	for( int i = 0; i < num_model_parameters; i++ )
		weights[i] = 1.0;

	A = MAT(population_size,num_model_parameters);
	for( int i = 0; i < population_size; i++ )
	{
		A(i,0) = weights[0];
		for( int j = 0; j < number_of_variables; j++ )
			A(i,j+1) = weights[j+1]*population[i]->variables[j];
	}

	b = VEC(population_size);
	for( int i = 0; i < population_size; i++ )
    	b(i) = population[i]->objective_value;
	
	model_parameters = surrogate_t::solveLinearEquations( A, b );
	//printf("surrogate time: %.3f\n",getTimer()-cur_time);
}

short surrogate_lin_t::checkForImprovement( solution_t *solution, partial_solution_t *part )
{
	double f_delta = 0;
	for( int i = 0; i < part->num_touched_variables; i++ )
	{
		int ind = part->touched_indices[i];
		f_delta -= model_parameters(ind+1) * solution->variables[ind]; 
		f_delta += model_parameters(ind+1) * part->touched_variables[i];
	}
	return( f_delta < 0 );
}


surrogate_quad_t::surrogate_quad_t( solution_t **population, int population_size, int number_of_variables )
{
	//double cur_time = getTimer();
	this->number_of_variables = number_of_variables;
	this->num_model_parameters = number_of_variables + 1;

	weights = new double[num_model_parameters];
	for( int i = 0; i < num_model_parameters; i++ )
		weights[i] = 1.0;

	A = MAT(population_size,num_model_parameters);
	for( int i = 0; i < population_size; i++ )
	{
		A(i,0) = weights[0];
		for( int j = 0; j < number_of_variables; j++ )
			A(i,j+1) = weights[j+1]*population[i]->variables[j]*population[i]->variables[j];
	}

	b = VEC(population_size);
	for( int i = 0; i < population_size; i++ )
    	b(i) = population[i]->objective_value;
	
	model_parameters = surrogate_t::solveLinearEquations( A, b );
	printModelParameters();
	//printf("surrogate time: %.3f\n",getTimer()-cur_time);
}

short surrogate_quad_t::checkForImprovement( solution_t *solution, partial_solution_t *part )
{
	double f_delta = 0;
	for( int i = 0; i < part->num_touched_variables; i++ )
	{
		int ind = part->touched_indices[i];
		f_delta -= model_parameters(ind+1) * solution->variables[ind] * solution->variables[ind];
		f_delta += model_parameters(ind+1) * part->touched_variables[i] * part->touched_variables[i];
		//f_delta -= solution->variables[ind] * solution->variables[ind];
		//f_delta += part->touched_variables[i] * part->touched_variables[i];
	}
	return( f_delta < 0 );
}


surrogate_linquad_t::surrogate_linquad_t( std::deque<solution_t> data, int number_of_variables )
{
	double cur_time = getTimer();
	this->number_of_variables = number_of_variables;
	this->num_model_parameters = 2*number_of_variables + 1;

	weights = new double[num_model_parameters];
	for( int i = 0; i < num_model_parameters; i++ )
		weights[i] = 1.0;
	/*for( int i = 0; i <= number_of_variables; i++ )
		weights[i] = 0.0;*/

	A = MAT(data.size(),num_model_parameters);
	int i = 0;
	auto it = data.begin();
	while( it != data.end() )
	{
		A(i,0) = weights[0];
		for( int j = 0; j < number_of_variables; j++ )
			A(i,j+1) = weights[j+1] * it->variables[j];
		for( int j = 0; j < number_of_variables; j++ )
			A(i,number_of_variables+j+1) = weights[number_of_variables+j+1] * it->variables[j]*it->variables[j];
		it++;
		i++;
	}

	i = 0;
	it = data.begin();
	b = VEC(data.size());
	while( it != data.end() )
	{
		b(i) = it->objective_value;
		it++;
		i++;
	}
	
	model_parameters = surrogate_t::solveLinearEquations( A, b );
	printModelError();
	surrogate_time += getTimer() - cur_time;
}

surrogate_linquad_t::surrogate_linquad_t( solution_t **parents, partial_solution_t **samples, int sample_size, int number_of_variables )
{
	this->number_of_variables = number_of_variables;
	this->num_model_parameters = 2*number_of_variables + 1;

	weights = new double[num_model_parameters];
	for( int i = 0; i < num_model_parameters; i++ )
		weights[i] = 1.0;

	A = MAT(sample_size,num_model_parameters);
	for( int i = 0; i < sample_size; i++ )
	{
		A(i,0) = weights[0];
		for( int j = 0; j < number_of_variables; j++ )
		{
			int ind = samples[i]->touched_indices[j];
			double vsamp = samples[i]->touched_variables[j];
			double vpar = parents[i]->variables[ind];
			A(i,j+1) = weights[j+1]*(vpar - vsamp);
			A(i,number_of_variables+j+1) = weights[number_of_variables+j+1]*(vpar*vpar - vsamp*vsamp);
		}
	}

	b = VEC(sample_size);
	for( int i = 0; i < sample_size; i++ )
    	b(i) = parents[i]->objective_value - samples[i]->objective_value;

	model_parameters = surrogate_t::solveLinearEquations( A, b );
	//printModelParameters();
}

surrogate_linquad_t::surrogate_linquad_t( solution_t **population, int population_size, int number_of_variables )
{
	double cur_time = getTimer();
	this->number_of_variables = number_of_variables;
	this->num_model_parameters = 2*number_of_variables + 1;

	weights = new double[num_model_parameters];
	for( int i = 0; i < num_model_parameters; i++ )
		weights[i] = 1.0;

	A = MAT(population_size,num_model_parameters);
	for( int i = 0; i < population_size; i++ )
	{
		A(i,0) = weights[0];
		for( int j = 0; j < number_of_variables; j++ )
			A(i,j+1) = weights[j+1]*population[i]->variables[j];
		for( int j = 0; j < number_of_variables; j++ )
			A(i,number_of_variables+j+1) = weights[number_of_variables+j+1] * population[i]->variables[j]*population[i]->variables[j];
	}

	b = VEC(population_size);
	for( int i = 0; i < population_size; i++ )
    	b(i) = population[i]->objective_value;
	
	model_parameters = surrogate_t::solveLinearEquations( A, b );
	surrogate_time += getTimer()-cur_time;
	//printModelParameters();
}

short surrogate_linquad_t::checkForImprovement( solution_t *solution, partial_solution_t *part )
{
	double cur_time = getTimer();
	double *var_backup = new double[part->num_touched_variables];
	for( int i = 0; i < part->num_touched_variables; i++ )
	{
		int ind = part->touched_indices[i];
		var_backup[i] = solution->variables[ind];
	}

	double fprev = weights[0] * model_parameters(0);
	for( int i = 0; i < number_of_variables; i++ )
	{
		fprev += weights[i+1] * model_parameters(i+1) * solution->variables[i];
		fprev += weights[number_of_variables+i+1] * model_parameters(number_of_variables+i+1) * solution->variables[i]*solution->variables[i];
	}

	for( int i = 0; i < part->num_touched_variables; i++ )
		solution->variables[part->touched_indices[i]] = part->touched_variables[i];
	
	double fnew = weights[0] * model_parameters(0);
	for( int i = 0; i < number_of_variables; i++ )
	{
		fnew += weights[i+1] * model_parameters(i+1) * solution->variables[i];
		fnew += weights[number_of_variables+i+1] * model_parameters(number_of_variables+i+1) * solution->variables[i]*solution->variables[i];
	}
	//fprintf(stderr,"%10.3e ",fnew - part->objective_value);
	//fprintf(stderr,"[%10.3e][%10.3e]->[%10.3e][%10.3e] (%d %d)\n",solution->objective_value,fprev,part->objective_value,fnew,part->objective_value<solution->objective_value?1:0,fnew<fprev?1:0);
	
	for( int i = 0; i < part->num_touched_variables; i++ )
		solution->variables[part->touched_indices[i]] = var_backup[i];
	delete var_backup;
	
	surrogate_time += getTimer()-cur_time;

	if( fnew < fprev )
		return 1;
	else
		return 0;
}

/*short surrogate_linquad_t::checkForImprovement( solution_t *solution, partial_solution_t *part )
{
	double f_delta = 0;
	for( int i = 0; i < part->num_touched_variables; i++ )
	{
		int ind = part->touched_indices[i];
		f_delta += weights[ind+1] * model_parameters(ind+1) * (part->touched_variables[i] - solution->variables[ind]);
		f_delta += weights[number_of_variables+ind+1] * model_parameters(number_of_variables+ind+1) * (part->touched_variables[i]*part->touched_variables[i] - solution->variables[ind]*solution->variables[ind]);
	}
	return( f_delta < 0 );
}*/

surrogate_chain_t::surrogate_chain_t( solution_t **population, int population_size, int number_of_variables )
{
	//double cur_time = getTimer();
	this->number_of_variables = number_of_variables;
	this->num_model_parameters = 3*number_of_variables;

	weights = new double[num_model_parameters];
	for( int i = 0; i < num_model_parameters; i++ )
		weights[i] = 1.0;

	A = MAT(population_size,num_model_parameters);
	for( int i = 0; i < population_size; i++ )
	{
		A(i,0) = weights[0];
		for( int j = 0; j < number_of_variables; j++ )
		{
			A(i,j+1) = weights[j+1]*population[i]->variables[j];
			A(i,number_of_variables+j+1) = weights[number_of_variables+j+1] * population[i]->variables[j]*population[i]->variables[j];
		}
		for( int j = 0; j < number_of_variables-1; j++ )
			A(i,2*number_of_variables+j+1) = weights[2*number_of_variables+j+1] * population[i]->variables[j]*population[i]->variables[j+1];
	}

	b = VEC(population_size);
	for( int i = 0; i < population_size; i++ )
    	b(i) = population[i]->objective_value;
	
	model_parameters = surrogate_t::solveLinearEquations( A, b );
	//printModelParameters();
	//printf("surrogate time: %.3f\n",getTimer()-cur_time);
}

short surrogate_chain_t::checkForImprovement( solution_t *solution, partial_solution_t *part )
{
	double cur_time = getTimer();
	double *var_backup = new double[part->num_touched_variables];
	for( int i = 0; i < part->num_touched_variables; i++ )
	{
		int ind = part->touched_indices[i];
		var_backup[i] = solution->variables[ind];
	}

	double fprev = weights[0] * model_parameters(0);
	int c = 0;
	for( int i = 0; i < number_of_variables; i++ )
	{
		fprev += weights[i+1] * model_parameters(i+1) * solution->variables[i];
		fprev += weights[number_of_variables+i+1] * model_parameters(number_of_variables+i+1) * solution->variables[i]*solution->variables[i];
		for( int j = 0; j < number_of_variables-1; j++ )
			fprev += weights[2*number_of_variables+j+1] * model_parameters[2*number_of_variables+j+1] * solution->variables[j] * solution->variables[j+1];
	}

	for( int i = 0; i < part->num_touched_variables; i++ )
		solution->variables[part->touched_indices[i]] = part->touched_variables[i];
	
	double fnew = weights[0] * model_parameters(0);
	c = 0;
	for( int i = 0; i < number_of_variables; i++ )
	{
		fnew += weights[i+1] * model_parameters(i+1) * solution->variables[i];
		fnew += weights[number_of_variables+i+1] * model_parameters(number_of_variables+i+1) * solution->variables[i]*solution->variables[i];
		for( int j = 0; j < number_of_variables-1; j++ )
			fnew += weights[2*number_of_variables+j+1] * model_parameters[2*number_of_variables+j+1] * solution->variables[j] * solution->variables[j+1];
	}
	//fprintf(stderr,"%10.3e ",fnew - part->objective_value);
	//fprintf(stderr,"[%10.3e][%10.3e]->[%10.3e][%10.3e] (%d %d)\n",solution->objective_value,fprev,part->objective_value,fnew,part->objective_value<solution->objective_value?1:0,fnew<fprev?1:0);
	
	for( int i = 0; i < part->num_touched_variables; i++ )
		solution->variables[part->touched_indices[i]] = var_backup[i];
	delete var_backup;
	
	surrogate_time += getTimer()-cur_time;

	if( fnew < fprev )
		return 1;
	else
		return 0;
}

/*short surrogate_chain_t::checkForImprovement( solution_t *solution, partial_solution_t *part )
{
	assert( part->num_touched_variables == 1 );

	double f_delta = 0;
	for( int i = 0; i < part->num_touched_variables; i++ )
	{
		int ind = part->touched_indices[i];
		f_delta += weights[ind+1] * model_parameters(ind+1) * (part->touched_variables[i] - solution->variables[ind]);
		f_delta += weights[number_of_variables+ind+1] * model_parameters(number_of_variables+ind+1) * (part->touched_variables[i]*part->touched_variables[i] - solution->variables[ind]*solution->variables[ind]);
		//f_delta += weights[3*number_of_variables+ind] * model_parameters(3*number_of_variables+ind) * (pow(part->touched_variables[i],4) - pow(solution->variables[ind]*solution->variables[ind],4));
		
		if( ind > 0 )
		{
			f_delta += weights[number_of_variables+ind+1] * model_parameters(2*number_of_variables+ind) * solution->variables[ind]*solution->variables[ind-1];
			f_delta -= weights[number_of_variables+ind+1] * model_parameters(2*number_of_variables+ind) * part->touched_variables[i]*solution->variables[ind-1];
		}
		if( ind < solution->number_of_variables-1 )
		{
			f_delta += weights[number_of_variables+ind+1] * model_parameters(2*number_of_variables+ind+1) * solution->variables[ind]*solution->variables[ind+1];
			f_delta -= weights[number_of_variables+ind+1] * model_parameters(2*number_of_variables+ind+1) * part->touched_variables[i]*solution->variables[ind+1];
		}
	}
	return( f_delta < 0 );
}*/
		
surrogate_blocks_t::surrogate_blocks_t( std::deque<solution_t> data, int number_of_variables, int block_size )
{
	double cur_time = getTimer();
	this->number_of_variables = number_of_variables;
	this->block_size = block_size;
	this->num_blocks = number_of_variables / block_size;
	assert( number_of_variables % block_size == 0 );
	this->num_model_parameters = 1+number_of_parameters + num_blocks * (block_size*(block_size+1))/2;

	weights = new double[num_model_parameters];
	for( int i = 0; i < num_model_parameters; i++ )
		weights[i] = 1.0;

	A = MAT(data.size(),num_model_parameters);
	int c, s;
	int i = 0;
	auto it = data.begin();
	while( it != data.end() )
	{
		A(i,0) = weights[0];
		for( int j = 0; j < number_of_variables; j++ )
			A(i,j+1) = weights[j+1] * it->variables[j];
		s = number_of_variables+1;
		c = 0;
		for( int j = 0; j < num_blocks; j++ )
		{
			int bstart = block_size * j;
			for( int k = 0; k < block_size; k++ )
			{
				for( int m = 0; m <= k; m++ )
				{
					A(i,s+c) = weights[s+c] * it->variables[bstart+k]*it->variables[bstart+m];
					c++;
				}
			}
		}
		it++;
		i++;
	}

	i = 0;
	it = data.begin();
	b = VEC(data.size());
	while( it != data.end() )
	{
		b(i) = it->objective_value;
		it++;
		i++;
	}

	//printf("[ %d %d ] model params\n",num_model_parameters,s+c);	
	model_parameters = surrogate_t::solveLinearEquations( A, b );
	surrogate_time += getTimer() - cur_time;
}

short surrogate_blocks_t::checkForImprovement( solution_t *solution, partial_solution_t *part )
{
	double cur_time = getTimer();
	double *var_backup = new double[part->num_touched_variables];
	for( int i = 0; i < part->num_touched_variables; i++ )
	{
		int ind = part->touched_indices[i];
		var_backup[i] = solution->variables[ind];
	}

	double fprev = evaluate( solution );
	if( replace_fitness )
	{
		solution->objective_value = fprev;
		solution->constraint_value = 0.0;
	}
	//fprintf(stderr,"VAR[ %10.3e %10.3e ] FREAL %10.3e FSURR %10.3e ",solution->variables[0],solution->variables[1],solution->objective_value,fprev);

	for( int i = 0; i < part->num_touched_variables; i++ )
		solution->variables[part->touched_indices[i]] = part->touched_variables[i];
	
	double fnew = evaluate( solution );
	if( replace_fitness )
	{
		part->objective_value = fnew;
		part->constraint_value = 0.0;
	}
	//fprintf(stderr,"VARNEW[ %10.3e %10.3e ] FREAL %10.3e FSURR %10.3e\n",solution->variables[0],solution->variables[1],part->objective_value,fnew);
	
	for( int i = 0; i < part->num_touched_variables; i++ )
		solution->variables[part->touched_indices[i]] = var_backup[i];
	delete var_backup;
	
	surrogate_time += getTimer()-cur_time;

	if( fnew < fprev )
		return 1;
	else
		return 0;
}

double surrogate_blocks_t::evaluate( solution_t *solution )
{
	double objv = weights[0] * model_parameters(0);
	for( int i = 0; i < number_of_variables; i++ )
		objv += weights[i+1] * model_parameters(i+1) * solution->variables[i];
	int s = number_of_variables+1;
	int c = 0;
	for( int i = 0; i < num_blocks; i++ )
	{
		int bstart = block_size * i;
		for( int k = 0; k < block_size; k++ )
		{
			for( int m = 0; m <= k; m++ )
			{
				objv += weights[s+c] * model_parameters[s+c] * solution->variables[bstart+k] * solution->variables[bstart+m];
				c++;
			}
		}
	}
	return( objv );
}


surrogate_full_t::surrogate_full_t( solution_t **population, int population_size, int number_of_variables )
{
	double cur_time = getTimer();
	this->number_of_variables = number_of_variables;
	this->num_model_parameters = 2*number_of_variables+1 + (number_of_variables*(number_of_variables-1))/2;

	weights = new double[num_model_parameters];
	for( int i = 0; i < num_model_parameters; i++ )
		weights[i] = 1.0;

	A = MAT(population_size,num_model_parameters);
	for( int i = 0; i < population_size; i++ )
	{
		A(i,0) = weights[0];
		for( int j = 0; j < number_of_variables; j++ )
		{
			A(i,j+1) = weights[j+1]*population[i]->variables[j];
			A(i,number_of_variables+j+1) = weights[number_of_variables+j+1] * population[i]->variables[j]*population[i]->variables[j];
		}
		int s = 2*number_of_variables+1;
		int c = 0;
		for( int j = 0; j < number_of_variables; j++ )
			for( int k = 0; k < j; k++ )
			{
				A(i,s+c) = weights[s+c] * population[i]->variables[j]*population[i]->variables[k];
				c++;
			}
	}

	b = VEC(population_size);
	for( int i = 0; i < population_size; i++ )
    	b(i) = population[i]->objective_value;
	
	model_parameters = surrogate_t::solveLinearEquations( A, b );
	surrogate_time += getTimer()-cur_time;
	//printf("surrogate time: %.3f\n",getTimer()-cur_time);
}

surrogate_full_t::surrogate_full_t( std::deque<solution_t> data, int number_of_variables )
{
	double cur_time = getTimer();
	this->number_of_variables = number_of_variables;
	this->num_model_parameters = 2*number_of_variables+1 + (number_of_variables*(number_of_variables-1))/2;

	weights = new double[num_model_parameters];
	for( int i = 0; i < num_model_parameters; i++ )
		weights[i] = 1.0;

	A = MAT(data.size(),num_model_parameters);
	int i = 0;
	auto it = data.begin();
	while( it != data.end() )
	{
		A(i,0) = weights[0];
		for( int j = 0; j < number_of_variables; j++ )
			A(i,j+1) = weights[j+1] * it->variables[j];
		for( int j = 0; j < number_of_variables; j++ )
			A(i,number_of_variables+j+1) = weights[number_of_variables+j+1] * it->variables[j]*it->variables[j];
		int s = 2*number_of_variables+1;
		int c = 0;
		for( int j = 0; j < number_of_variables; j++ )
			for( int k = 0; k < j; k++ )
			{
				A(i,s+c) = weights[s+c] * it->variables[j]*it->variables[k];
				c++;
			}
		it++;
		i++;
	}

	i = 0;
	it = data.begin();
	b = VEC(data.size());
	while( it != data.end() )
	{
		b(i) = it->objective_value;
		it++;
		i++;
	}
	
	model_parameters = surrogate_t::solveLinearEquations( A, b );
	surrogate_time += getTimer() - cur_time;
}

short surrogate_full_t::checkForImprovement( solution_t *solution, partial_solution_t *part )
{
	double cur_time = getTimer();
	double *var_backup = new double[part->num_touched_variables];
	for( int i = 0; i < part->num_touched_variables; i++ )
	{
		int ind = part->touched_indices[i];
		var_backup[i] = solution->variables[ind];
	}

	double fprev = weights[0] * model_parameters(0);
	int c = 0;
	for( int i = 0; i < number_of_variables; i++ )
	{
		fprev += weights[i+1] * model_parameters(i+1) * solution->variables[i];
		fprev += weights[number_of_variables+i+1] * model_parameters(number_of_variables+i+1) * solution->variables[i]*solution->variables[i];
		int s = 2*number_of_variables+1;
		for( int j = 0; j < i; j++ )
		{
			fprev += weights[s+c] * model_parameters[s+c] * solution->variables[i] * solution->variables[j];
			c++;
		}
	}
	//fprintf(stderr,"VAR[ %10.3e %10.3e ] FREAL %10.3e FSURR %10.3e ",solution->variables[0],solution->variables[1],solution->objective_value,fprev);

	for( int i = 0; i < part->num_touched_variables; i++ )
		solution->variables[part->touched_indices[i]] = part->touched_variables[i];
	
	double fnew = weights[0] * model_parameters(0);
	c = 0;
	for( int i = 0; i < number_of_variables; i++ )
	{
		fnew += weights[i+1] * model_parameters(i+1) * solution->variables[i];
		fnew += weights[number_of_variables+i+1] * model_parameters(number_of_variables+i+1) * solution->variables[i]*solution->variables[i];
		int s = 2*number_of_variables+1;
		for( int j = 0; j < i; j++ )
		{
			fnew += weights[s+c] * model_parameters[s+c] * solution->variables[i] * solution->variables[j];
			c++;
		}
	}
	//fprintf(stderr,"VARNEW[ %10.3e %10.3e ] FREAL %10.3e FSURR %10.3e\n",solution->variables[0],solution->variables[1],part->objective_value,fnew);
	
	for( int i = 0; i < part->num_touched_variables; i++ )
		solution->variables[part->touched_indices[i]] = var_backup[i];
	delete var_backup;
	
	surrogate_time += getTimer()-cur_time;

	if( fnew < fprev )
		return 1;
	else
		return 0;
}

surrogate_rbf_t::surrogate_rbf_t( solution_t **population, int population_size, int number_of_variables )
{
	double cur_time = getTimer();
	this->number_of_variables = number_of_variables;
	this->num_model_parameters = 10; //population_size/10;
	centroids = (double**) Malloc( num_model_parameters * sizeof(double) );
	for( int i = 0; i < num_model_parameters; i++ )
		centroids[i] = (double*) Malloc( number_of_variables );
	weights = new double[num_model_parameters];

	A = MAT(population_size,num_model_parameters);
	for( int i = 0; i < num_model_parameters; i++ )
		for( int j = 0; j < number_of_variables; j++ )
			centroids[i][j] = population[i]->variables[j];
	for( int i = 0; i < population_size; i++ )
		for( int j = 0; j < num_model_parameters; j++ )
			A(i,j) = phi(centroids[j],(double*)population[i]->variables.mem);

	b = VEC(population_size);
	for( int i = 0; i < population_size; i++ )
    	b(i) = population[i]->objective_value;
	
	model_parameters = surrogate_t::solveLinearEquations( A, b );
	surrogate_time += getTimer()-cur_time;
	//printf("surrogate time: %.3f\n",getTimer()-cur_time);
}

surrogate_rbf_t::~surrogate_rbf_t()
{
	for( int i = 0; i < num_model_parameters; i++ )
		free( centroids[i] );
	free( centroids );
}

short surrogate_rbf_t::checkForImprovement( solution_t *solution, partial_solution_t *part )
{
	double cur_time = getTimer();
	double *var_backup = new double[part->num_touched_variables];
	for( int i = 0; i < part->num_touched_variables; i++ )
	{
		int ind = part->touched_indices[i];
		var_backup[i] = solution->variables[ind];
	}

	double fprev = 0.0;
	int c = 0;
	for( int i = 0; i < num_model_parameters; i++ )
		fprev += model_parameters(i) * phi(centroids[i],(double*)solution->variables.mem);
	fprintf(stderr,"VAR[ %10.3e %10.3e ] FREAL %10.3e FSURR %10.3e ",solution->variables[0],solution->variables[1],solution->objective_value,fprev);

	for( int i = 0; i < part->num_touched_variables; i++ )
		solution->variables[part->touched_indices[i]] = part->touched_variables[i];
	
	double fnew = 0.0;
	for( int i = 0; i < num_model_parameters; i++ )
		fnew += model_parameters(i) * phi(centroids[i],(double*)solution->variables.mem);
	fprintf(stderr,"VARNEW[ %10.3e %10.3e ] FREAL %10.3e FSURR %10.3e\n",solution->variables[0],solution->variables[1],part->objective_value,fnew);
	
	for( int i = 0; i < part->num_touched_variables; i++ )
		solution->variables[part->touched_indices[i]] = var_backup[i];
	delete var_backup;
	
	surrogate_time += getTimer()-cur_time;

	if( fnew < fprev )
		return 1;
	else
		return 0;
}

double surrogate_rbf_t::phi( double *c, double *x )
{
	double r = distanceEuclidean(c,x,number_of_parameters);
	return( exp(-1*(eps*r)*(eps*r)) );
}

void surrogate_t::printModelError()
{
#ifdef EIGEN
	error = (A*model_parameters - b).norm();
	fprintf(stderr,"[ %10.3e\t%10.3e ]",error,error / b.norm());
#elif defined ARMADILLO
	error = norm(A*model_parameters - b);
	fprintf(stderr,"[ %10.3e\t%10.3e ]",error,error / norm(b));
#endif
	fprintf(stderr,"\n");
}

void surrogate_t::printModelParameters()
{
	for(int i = 0; i < num_model_parameters; i++ )
		fprintf(stderr,"%10.3e ",model_parameters(i));
	printModelError();
}

void surrogate_t::writeData(std::string filename)
{
	FILE *f = fopen(filename.c_str(),"w");
#ifdef ARMADILLO
	MAT C = A*model_parameters;
	for( int i = 0; i < A.n_rows; i++ )
	{
		fprintf(f,"%10.3e %10.3e\t",C(i),b(i));
		for( int j = 0; j < A.n_cols; j++ )
			fprintf(f,"%10.3e ",A(i,j));
		fprintf(f,"\n");
	}
#endif
	fclose(f);
}
