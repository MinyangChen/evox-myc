import jax
from jax import lax, vmap
import jax.numpy as jnp
from evox import Algorithm, jit_class, State
from evox.operators.selection import select_rand_pbest
from evox.operators.crossover import (
    de_diff_sum,
    de_diff_sum_archive,
    de_arith_recom,
    de_bin_cross,
    de_exp_cross,
)

from functools import partial

from jax.experimental.host_callback import id_print

@jit_class
class JaDE(Algorithm):
    """JaDE
    Zhang J, Sanderson A C.
    JADE: adaptive differential evolution with optional external archive[J].
    IEEE Transactions on evolutionary computation, 2009, 13(5): 945-958.
    """

    def __init__(
        self,
        lb,
        ub,
        pop_size=100,
        diff_padding_num=9,
        differential_weight=None,
        cross_probability=None,
        c=0.1,
        p=0.05,
        with_archive=1,
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.diff_padding_num = diff_padding_num
        self.batch_size = pop_size
        self.c = c

        self.cross_probability = cross_probability
        self.differential_weight = differential_weight
        self.num_diff_vects = 1

        self.p = p
        self.with_archive = with_archive

    def setup(self, key):
        state_key, init_key = jax.random.split(key, 2)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        trial_vectors = jnp.zeros(shape=(self.batch_size, self.dim))
        best_index = 0
        start_index = 0

        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            key=state_key,
            trial_vectors=trial_vectors,
            F_u=0.5,
            CR_u=0.5,
            F_vect=jnp.empty(self.pop_size),
            CR_vect=jnp.empty(self.pop_size),
            archive=population,
        )

    def ask(self, state):
        key, ask_one_key, F_key, CR_key = jax.random.split(state.key, 4)
        ask_one_keys = jax.random.split(ask_one_key, self.batch_size)
        indices = jnp.arange(self.batch_size) + state.start_index

        # Generare F and CR
        F_vect = jax.random.cauchy(F_key, shape=(self.pop_size,)) * 0.1 + state.F_u
        F_vect = jnp.clip(F_vect, jnp.zeros(self.pop_size), jnp.ones(self.pop_size))
        CR_vect = jax.random.normal(CR_key, shape=(self.pop_size,)) * 0.1 + state.CR_u
        CR_vect = jnp.clip(CR_vect, jnp.zeros(self.pop_size), jnp.ones(self.pop_size))

        trial_vectors = vmap(
            partial(
                self._ask_one,
                state_inner=state,
            )
        )(ask_one_key=ask_one_keys, index=indices, F=F_vect, CR=CR_vect)

        return trial_vectors, state.update(
            trial_vectors=trial_vectors, key=key, F_vect=F_vect, CR_vect=CR_vect
        )

    def _ask_one(self, state_inner, ask_one_key, index, F, CR):
        select_key, pbest_key, crossover_key = jax.random.split(ask_one_key, 3)

        population = state_inner.population
        fitness = state_inner.fitness

        self.differential_weight = F
        self.cross_probability = CR

        if self.with_archive:
            difference_sum, _rand_vect_idx = de_diff_sum_archive(
                select_key,
                self.diff_padding_num,
                self.num_diff_vects,
                index,
                population,
                state_inner.archive,
            )
        else:
            difference_sum, _rand_vect_idx = de_diff_sum(
                select_key,
                self.diff_padding_num,
                self.num_diff_vects,
                index,
                population,
            )

        pbest_vect = select_rand_pbest(pbest_key, self.p, population, fitness)
        current_vect = population[index]

        base_vector_prim = current_vect
        base_vector_sec = pbest_vect

        base_vector = base_vector_prim + self.differential_weight * (
            base_vector_sec - base_vector_prim
        )

        mutation_vector = base_vector + difference_sum * self.differential_weight

        trial_vector = de_bin_cross(
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

        compare = trial_fitness < batch_fitness

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

        """Update F_u and CR_u:
        Each generation's successful F and CR values are recorded, while the unsuccessful ones are not. 
        Update the mean values of F (F_u) and CR (CR_u) for the next generation using formulas (F uses the Lehmer mean)."""
        S_F_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)
        S_CR_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)

        S_F = jnp.where(compare, state.F_vect, S_F_init)
        S_CR = jnp.where(compare, state.CR_vect, S_CR_init)


        no_success = jnp.all(~compare)

        F_u_temp = (1 - self.c) * state.F_u + self.c * (
            jnp.nansum(S_F**2) / jnp.nansum(S_F)
        )
        F_u = lax.select(no_success, state.F_u, F_u_temp)

        CR_u_temp = (1 - self.c) * state.CR_u + self.c * jnp.nanmean(S_CR)
        CR_u = lax.select(no_success, state.CR_u, CR_u_temp)

        """Update archive"""
        archive = jnp.where(compare[:, jnp.newaxis], batch_pop, state.archive)

        return state.update(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            F_u=F_u,
            CR_u=CR_u,
            archive=archive,
        )
