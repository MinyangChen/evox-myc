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

@jit_class
class SHADE(Algorithm):
    """SHADE
    Tanabe R, Fukunaga A.
    Success-history based parameter adaptation for differential evolution[C]//2013
    IEEE congress on evolutionary computation. IEEE, 2013: 71-78.
    """

    def __init__(
        self,
        lb,
        ub,
        pop_size=100,
        diff_padding_num=9,
        with_archive=1,
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.diff_padding_num = diff_padding_num
        self.batch_size = pop_size
        self.H = pop_size

        self.num_diff_vects = 1
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
            Memory_F=jnp.full(shape=(100,), fill_value=0.5),
            Memory_CR=jnp.full(shape=(100,), fill_value=0.5),
            F_vect=jnp.empty(self.pop_size),
            CR_vect=jnp.empty(self.pop_size),
            archive=population,
        )

    def ask(self, state):
        key, ask_one_key, choice_key, F_key, CR_key = jax.random.split(state.key, 5)
        ask_one_keys = jax.random.split(ask_one_key, self.batch_size)
        indices = jnp.arange(self.batch_size) + state.start_index

        FCR_ids = jax.random.choice(
            choice_key, a=self.H, shape=(self.batch_size,), replace=True
        )
        M_F_vect = state.Memory_F[FCR_ids]
        M_CR_vect = state.Memory_CR[FCR_ids]

        # Generare F and CR
        F_vect = jax.random.cauchy(F_key, shape=(self.pop_size,)) * 0.1 + M_F_vect
        F_vect = jnp.clip(F_vect, jnp.zeros(self.pop_size), jnp.ones(self.pop_size))

        CR_vect = jax.random.normal(CR_key, shape=(self.pop_size,)) * 0.1 + M_CR_vect
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

        differential_weight = F
        cross_probability = CR

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

        pbest_vect = select_rand_pbest(pbest_key, 0.05, population, fitness)
        current_vect = population[index]

        base_vector_prim = current_vect
        base_vector_sec = pbest_vect

        base_vector = base_vector_prim + differential_weight * (
            base_vector_sec - base_vector_prim
        )

        mutation_vector = base_vector + difference_sum * differential_weight

        trial_vector = de_bin_cross(
            crossover_key,
            mutation_vector,
            current_vect,
            cross_probability,
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

        """Update Memory_F and Memory_CR:
        Each generation's successful F and CR values are recorded, and the unsuccessful ones are not. 
        Calculate the mean of the recorded F and CR values (F uses a weighted Lehmer mean, and CR uses a weighted arithmetic mean), 
        and store them in two archive tables (Memory_F and Memory_CR)."""
        S_F_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)
        S_CR_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)
        S_delta_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)
        deltas = batch_fitness - trial_fitness

        compare2 = trial_fitness < batch_fitness
        S_F = jnp.where(compare2, state.F_vect, S_F_init)
        S_CR = jnp.where(compare2, state.CR_vect, S_CR_init)
        S_delta = jnp.where(compare2, deltas, S_delta_init)

        norm_delta = S_delta / jnp.nansum(S_delta)
        M_CR = jnp.nansum(norm_delta * S_CR)
        M_F = jnp.nansum(norm_delta * (S_F**2)) / jnp.nansum(norm_delta * S_F)


        Memory_F_update = jnp.roll(state.Memory_F, shift=1)
        Memory_F_update = Memory_F_update.at[0].set(M_F)
        is_F_nan = jnp.isnan(M_F)
        Memory_F = lax.select(is_F_nan, state.Memory_F, Memory_F_update)

        Memory_CR_update = jnp.roll(state.Memory_CR, shift=1)
        Memory_CR_update = Memory_CR_update.at[0].set(M_CR)
        is_CR_nan = jnp.isnan(M_CR)
        Memory_CR = lax.select(is_CR_nan, state.Memory_CR, Memory_CR_update)

        """Update archive"""
        archive = jnp.where(compare2[:, jnp.newaxis], state.archive, batch_pop)

        return state.update(
            population=population,
            fitness=fitness,
            best_index=best_index,
            start_index=start_index,
            Memory_F=Memory_F,
            Memory_CR=Memory_CR,
            archive=archive,
        )
