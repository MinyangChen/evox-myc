import jax
from jax import lax, vmap
import jax.numpy as jnp
from evox import Algorithm, jit_class, State
from evox.operators.selection import select_rand_pbest
from evox.operators.crossover import (
    de_diff_sum_rank,
    move_n_small_numbers,
    de_arith_recom,
    de_bin_cross,
    de_exp_cross,
)

from functools import partial

from jax.experimental.host_callback import id_print


@jit_class
class LSHADE_RSP(Algorithm):
    """
    V. Stanovov, S. Akhmedova and E. Semenkin, "LSHADE Algorithm with Rank-Based Selective Pressure Strategy for Solving CEC 2017 Benchmark Problems," 
    2018 IEEE Congress on Evolutionary Computation (CEC), Rio de Janeiro, Brazil, 2018, pp. 1-8, doi: 10.1109/CEC.2018.8477977."""

    def __init__(
        self,
        lb,
        ub,
        pop_size=100,
        diff_padding_num=3,
        pop_size_min=4,
        p_const=0.085,
        k_factor=3,
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.diff_padding_num = diff_padding_num
        self.H = 5
        self.p_const = p_const
        self.k_factor = k_factor

        self.num_diff_vects = 1
        self.pop_size_min=pop_size_min

    def setup(self, key):
        state_key, init_key = jax.random.split(key, 2)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        trial_vectors = jnp.zeros(shape=(self.pop_size, self.dim))
        best_index = 1

        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            key=state_key,
            trial_vectors=trial_vectors,
            Memory_F=jnp.full(shape=(self.H,), fill_value=0.3),
            Memory_CR=jnp.full(shape=(self.H,), fill_value=0.8),
            F_vect=jnp.empty(self.pop_size),
            CR_vect=jnp.empty(self.pop_size),
            archive=population,
            CR_cutoff=0,

            pop_size_reduced=self.pop_size,
            worst_solution=population[0],
            progress=0,
            p=self.p_const
        )

    def ask(self, state):
        key, ask_one_key, choice_key, F_key, CR_key = jax.random.split(state.key, 5)
        ask_one_keys = jax.random.split(ask_one_key, self.pop_size)
        indices = jnp.arange(self.pop_size)

        FCR_ids = jax.random.choice(
            choice_key, a=self.H, shape=(self.pop_size,), replace=True
        )
        M_F_vect = state.Memory_F[FCR_ids]
        M_CR_vect = state.Memory_CR[FCR_ids]

        # Generare F. Limit F in a specific range
        F_vect = jax.random.cauchy(F_key, shape=(self.pop_size,)) * 0.1 + M_F_vect
        F_vect = jnp.clip(F_vect, jnp.zeros(self.pop_size), jnp.ones(self.pop_size))

        F_vect_temp = jnp.where(F_vect < 0.7, F_vect, 0.7)
        F_vect = lax.select(state.progress < 0.6, F_vect_temp, F_vect)

        # Generare CR. Limit CR in a specific range
        CR_vect = jax.random.normal(CR_key, shape=(self.pop_size,)) * 0.1 + M_CR_vect
        CR_vect = jnp.clip(CR_vect, jnp.zeros(self.pop_size), jnp.ones(self.pop_size))
        
        conditions_CR = [state.progress <= 0.25, (state.progress > 0.25) & (state.progress <= 0.5), state.progress > 0.5]
        values_CR = [0.7, 0.6, 0.0]
        CR_min = jnp.select(conditions_CR, values_CR)

        CR_vect = jnp.where(CR_vect > CR_min, CR_vect, CR_min)
 
        # Determine the factor of Fw
        conditions = [state.progress <= 0.2, (state.progress > 0.2) & (state.progress <= 0.4), state.progress > 0.4]
        values = [0.7, 0.8, 1.2]
        Fw_factor = jnp.select(conditions, values)

        trial_vectors = vmap(
            partial(
                self._ask_one,
                state_inner=state,
                Fw_factor=Fw_factor,
            )
        )(ask_one_key=ask_one_keys, index=indices, F=F_vect, CR=CR_vect)

        # Replace nan solutions as worst_solution. In case of evaluating nan solution.
        replace_mask = jnp.arange(self.pop_size) < state.pop_size_reduced
        trial_vectors = jnp.where(replace_mask[:, jnp.newaxis], trial_vectors, state.worst_solution)

        return trial_vectors, state.update(
            trial_vectors=trial_vectors, key=key, F_vect=F_vect, CR_vect=CR_vect
        )

    def _ask_one(self, state_inner, Fw_factor, ask_one_key, index, F, CR):
        select_key, pbest_key, crossover_key = jax.random.split(ask_one_key, 3)

        population = state_inner.population
        fitness = state_inner.fitness
        pop_size_reduced = state_inner.pop_size_reduced
        p = state_inner.p

        differential_weight = F
        cross_probability = CR


        difference_sum, _rand_vect_idx = de_diff_sum_rank(
            select_key,
            self.diff_padding_num,
            self.num_diff_vects,
            index,
            population,
            self.k_factor,
            pop_size_reduced=pop_size_reduced,  
            fitness=fitness,
        )

        pbest_vect = select_rand_pbest(pbest_key, p, population, fitness)
        current_vect = population[index]

        base_vector_prim = current_vect
        base_vector_sec = pbest_vect

        base_vector = base_vector_prim + differential_weight * Fw_factor * (
            base_vector_sec - base_vector_prim
        )

        mutation_vector = base_vector + difference_sum * differential_weight
  
        trial_vector = de_bin_cross(
            crossover_key,
            mutation_vector,
            current_vect,
            cross_probability,
        )

        # The new repair function
        compare_min = trial_vector < self.lb
        repair_min = (current_vect + self.lb) / 2
        trial_vector = jnp.where(compare_min, repair_min, trial_vector)

        compare_max = trial_vector > self.ub
        repair_max = (current_vect + self.ub) / 2
        trial_vector = jnp.where(compare_max, repair_max, trial_vector)

        return trial_vector

    def tell(self, state, trial_fitness):
        trial_mask = jnp.arange(self.pop_size) < state.pop_size_reduced
        trial_fitness = jnp.where(trial_mask, trial_fitness, jnp.inf)

        compare = trial_fitness <= state.fitness
        
        population = jnp.where(
            compare[:, jnp.newaxis], state.trial_vectors, state.population
        )
        fitness = jnp.where(compare, trial_fitness, state.fitness)

        best_index = jnp.argmin(fitness)

        """Reduce population and fitness:
        Use nan to occupy unwanted individuals in population and inf to occupy unwanted bits in fitness."""

        # Set fitness[pop_size_reduced:] as inf and pop[pop_size_reduced:] as nan
        moved_fitness, move_ids = move_n_small_numbers(fitness, state.pop_size_reduced)
        replace_mask = jnp.arange(self.pop_size) < state.pop_size_reduced
        moved_fitness = jnp.where(replace_mask, moved_fitness, jnp.inf)

        moved_population = population[move_ids]
        moved_population = jnp.where(replace_mask[:, jnp.newaxis], moved_population, jnp.nan)

        # Record the worst solution to fill nan solution in population
        max_index = jnp.nanargmax(fitness)
        worst_solution = population[max_index]

        """Update Memory_F and Memory_CR:
        Each generation's successful F and CR values are recorded, and the unsuccessful ones are not. 
        Calculate the mean of the recorded F and CR values (F uses a weighted Lehmer mean, and CR uses a weighted arithmetic mean), 
        and store them in two archive tables (Memory_F and Memory_CR)."""
        S_F_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)
        S_CR_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)
        S_delta_init = jnp.full(shape=(self.pop_size,), fill_value=jnp.nan)
        deltas = state.fitness - trial_fitness

        compare2 = trial_fitness < state.fitness
        S_F = jnp.where(compare2, state.F_vect, S_F_init)
        S_CR = jnp.where(compare2, state.CR_vect, S_CR_init)
        S_delta = jnp.where(compare2, deltas, S_delta_init)

        # The cutoff condition of CR. Once it achieves, CR will be 0 to the end.
        is_cutoff = jnp.nanmax(S_CR) <= 0
        CR_cutoff = lax.select(is_cutoff, 1, state.CR_cutoff)

        norm_delta = S_delta / jnp.nansum(S_delta)
        M_CR = (jnp.nansum(norm_delta * (S_CR**2)) / jnp.nansum(norm_delta * S_CR) + state.Memory_CR[0]) / 2
        M_F = (jnp.nansum(norm_delta * (S_F**2)) / jnp.nansum(norm_delta * S_F) + state.Memory_F[0]) / 2


        Memory_F_update = jnp.roll(state.Memory_F, shift=1)
        Memory_F_update = Memory_F_update.at[0].set(M_F)
        is_F_nan = jnp.isnan(M_F)
        Memory_F = lax.select(is_F_nan, state.Memory_F, Memory_F_update)

        Memory_CR_update = jnp.roll(state.Memory_CR, shift=1)
        Memory_CR_update = Memory_CR_update.at[0].set(M_CR)
        is_CR_nan = jnp.isnan(M_CR)
        Memory_CR = lax.select(is_CR_nan, state.Memory_CR, Memory_CR_update)

        # Ensure the last entry of Memory_F/Memory_CR be 0.9
        Memory_CR = Memory_CR.at[self.H - 1].set(0.9)
        Memory_F = Memory_F.at[self.H - 1].set(0.9)

        """Update archive"""
        archive = jnp.where(compare2[:, jnp.newaxis], state.population, state.archive)

        """Ajust pop_size"""
        pop_size_temp = self.pop_size - (self.pop_size - self.pop_size_min) * state.progress
        pop_size_reduced = pop_size_temp.astype(int)

        """Update pbest percent"""
        p = self.p_const + self.p_const * state.progress

        return state.update(
            population=moved_population,
            fitness=moved_fitness,
            best_index=best_index,
            Memory_F=Memory_F,
            Memory_CR=Memory_CR,
            archive=archive,
            CR_cutoff=CR_cutoff,

            pop_size_reduced=pop_size_reduced,
            worst_solution=worst_solution,
            p=p,
        )
