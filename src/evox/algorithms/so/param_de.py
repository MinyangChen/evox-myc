import jax
import jax.numpy as jnp
from jax import vmap, lax
from jax.experimental.host_callback import id_print

from evox import (
    Algorithm,
    State,
    jit_class,
)
from evox.operators.selection import select_rand_pbest
from evox.operators.crossover import (
    de_diff_sum,
    de_arith_recom,
    de_bin_cross,
    de_exp_cross,
)
from evox.utils import *

@jit_class
class ParamDE(Algorithm):
    """Parametric DE"""
    def __init__(
        self,
        lb,
        ub,
        pop_size=100,
        diff_padding_num = 9,

        differential_weight = 0.3471559,
        cross_probability = 0.78762645 ,
        basevect_prim_type = 0,
        basevect_sec_type = 2,
        num_diff_vects = 3,
        cross_strategy = 2,
    ):
        self.num_diff_vects = num_diff_vects
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        # batch_size：1-popsize, and pop_size % batch_size == 0
        self.batch_size = pop_size      
        self.cross_probability = cross_probability
        self.differential_weight = differential_weight
        self.cross_strategy = cross_strategy
        self.diff_padding_num = diff_padding_num
        self.basevect_prim_type = basevect_prim_type
        self.basevect_sec_type = basevect_sec_type

    def setup(self, key):  
        state_key, init_key = jax.random.split(key, 2)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        trial_vectors = jnp.zeros(shape=(self.batch_size, self.dim))
        best_index = 0
        start_index = 0     # The index of the individual currently operating
        
        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            trial_vectors=trial_vectors,
        )

    def ask(self, state):
        # Unlike traditional DE, this DE can mutate/crossover batch_size solutions at one time, 
        # and the batch_size is between 1 and popsize.
        key, ask_one_key = jax.random.split(state.key, 2) 
        ask_one_keys = jax.random.split(ask_one_key, self.batch_size)
        indices = jnp.arange(self.batch_size) + state.start_index    

        trial_vectors = vmap(partial(self._ask_one, state_inner = state,) )(ask_one_key=ask_one_keys, index=indices)
        
        return trial_vectors, state.update(trial_vectors=trial_vectors, key=key) 
    
    def _ask_one(self, state_inner, ask_one_key, index):
        select_key, pbest_key, crossover_key = jax.random.split(ask_one_key, 3)

        population = state_inner.population
        best_index = state_inner.best_index
        fitness = state_inner.fitness
        
        """Mutation: base_vect + F * difference_sum"""
        # difference_sum, rand_vect_idx = diff_sum(self.num_diff_vects, select_key, self.pop_size, self.diff_padding_num, index, population)
        difference_sum, rand_vect_idx = de_diff_sum(
            select_key,
            self.diff_padding_num,
            self.num_diff_vects,
            index,
            population,
        )
  
        rand_vect = population[rand_vect_idx]
        best_vect = population[best_index]     
        pbest_vect = select_rand_pbest(pbest_key, 0.05, population, fitness)
        current_vect = population[index]
        vector_merge = jnp.stack((rand_vect, best_vect, pbest_vect, current_vect))

        base_vector_prim = vector_merge[self.basevect_prim_type]
        base_vector_sec = vector_merge[self.basevect_sec_type]

        base_vector = base_vector_prim + self.differential_weight * (base_vector_sec - base_vector_prim)

        mutation_vector = (base_vector + difference_sum * self.differential_weight)

        """Crossover: 0 = bin, 1 = exp, 2 = arith"""
        cross_funcs = (
            de_bin_cross,
            de_exp_cross,
            lambda _key, x, y, z: de_arith_recom(x, y, z),
        )
        trial_vector = lax.switch(
            self.cross_strategy,
            cross_funcs,
            crossover_key,
            mutation_vector,
            current_vect,
            self.cross_probability,
        )

        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)
        
        return trial_vector

    def tell(self, state, trial_fitness):   
        start_index = state.start_index
        batch_pop = jax.lax.dynamic_slice_in_dim( 
            state.population, start_index, self.batch_size, axis=0
        )
        batch_fitness = jax.lax.dynamic_slice_in_dim(
            state.fitness, start_index, self.batch_size, axis=0
        )

        compare = trial_fitness <= batch_fitness 

        population_update = jnp.where(
            compare[:, jnp.newaxis], state.trial_vectors, batch_pop
        )
        fitness_update = jnp.where(compare, trial_fitness, batch_fitness)

        population = jax.lax.dynamic_update_slice_in_dim(
            state.population, population_update, start_index, axis=0
        )
        fitness = jax.lax.dynamic_update_slice_in_dim(
            state.fitness, fitness_update, start_index, axis=0
        )
        best_index = jnp.argmin(fitness)
        start_index = (state.start_index + self.batch_size) % self.pop_size 
        return state.update(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
        )