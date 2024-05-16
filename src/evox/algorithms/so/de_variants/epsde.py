import jax
from jax import lax, vmap
import jax.numpy as jnp
from evox import Algorithm, jit_class, State
from evox.operators.selection import select_rand_pbest
from evox.operators.crossover import (
    de_diff_sum,
    de_arith_recom,
    de_bin_cross,
    de_exp_cross,
)

from functools import partial
from jax.experimental.host_callback import id_print

"""Strategy codes(4 bits): [basevect_prim, basevect_sec, diff_num, cross_strategy]
basevect: 0="rand", 1="best", 2="pbest", 3="current",
cross_strategy: 0=bin, 1=exp, 2=arith """
rand_1_bin = jnp.array([0, 0, 1, 0])
rand2best_2_bin = jnp.array([0, 1, 2, 0])
rand_2_bin = jnp.array([0, 0, 2, 0])

best_2_bin = jnp.array([1, 1, 2, 0])

current2rand_1_bin = jnp.array([3, 0, 1, 0])
current2rand_1 = jnp.array([0, 0, 1, 2])  # current2rand_1 <==> rand_1_arith
current2pbest_1_bin = jnp.array([3, 2, 1, 0])

@jit_class
class EPSDE(Algorithm):
    """R. Mallipeddi, P.N. Suganthan, Q.K. Pan, M.F. Tasgetiren, Differential evolution algorithm with ensemble of parameters and mutation strategies,
    Applied Soft Computing, Volume 11, Issue 2, 2011, Pages 1679-1696, ISSN 1568-4946."""

    def __init__(
        self,
        lb,
        ub,
        pop_size=100,
        diff_padding_num=5,
        differential_weight=None,
        cross_probability=None,
        p=0.05,
    ):
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.diff_padding_num = diff_padding_num

        self.cross_probability = cross_probability
        self.differential_weight = differential_weight
        self.num_diff_vects = 1

        self.p = p
        self.F_pool = jnp.arange(4, 10) / 10.0
        self.CR_pool = jnp.arange(1, 10) / 10.0
        self.strategies = jnp.array([rand_1_bin, best_2_bin, current2rand_1_bin])

    def setup(self, key):
        state_key, init_key, strategy_key, F_key, CR_key = jax.random.split(key, 5)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        trial_vectors = jnp.zeros(shape=(self.pop_size, self.dim))
        best_index = 0

        strategies_ids = jax.random.choice(strategy_key, a=3, shape=(self.pop_size,), replace=True)
        F_ids = jax.random.choice(F_key, a=6, shape=(self.pop_size, 1), replace=True)
        CR_ids = jax.random.choice(CR_key, a=9, shape=(self.pop_size, 1), replace=True)
        strategies_vect = self.strategies[strategies_ids]
        F_vect = self.F_pool[F_ids]
        CR_vect = self.CR_pool[CR_ids]
        param_vect = jnp.concatenate((strategies_vect, F_vect, CR_vect), axis=1)

        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            key=state_key,
            trial_vectors=trial_vectors,
            param_vect=param_vect,
            S_param_vect=param_vect,
            compare=jnp.full((self.pop_size,), True)
        )

    def ask(self, state):
        key, ask_one_key, strategy_key, param_renew_key, F_key, CR_key = jax.random.split(state.key, 6)
        ask_one_keys = jax.random.split(ask_one_key, self.pop_size)
        indices = jnp.arange(self.pop_size)

        # Generare F and CR
        strategies_ids = jax.random.choice(strategy_key, a=3, shape=(self.pop_size,), replace=True)
        F_ids = jax.random.choice(F_key, a=6, shape=(self.pop_size, 1), replace=True)
        CR_ids = jax.random.choice(CR_key, a=9, shape=(self.pop_size, 1), replace=True)

        strategies_vect = self.strategies[strategies_ids]
        F_vect = self.F_pool[F_ids]
        CR_vect = self.CR_pool[CR_ids]
        param_vect_random = jnp.concatenate((strategies_vect, F_vect, CR_vect), axis=1)

        renew_mask = jax.random.choice(param_renew_key, a=2, shape=(self.pop_size,), replace=True)
        param_vect_renew = jnp.where(renew_mask[:, jnp.newaxis], state.S_param_vect, param_vect_random)

        param_vect = jnp.where(state.compare[:, jnp.newaxis], state.param_vect, param_vect_renew)

        trial_vectors = vmap(
            partial(
                self._ask_one,
                state_inner=state,
            )
        )(ask_one_key=ask_one_keys, index=indices, param=param_vect)

        return trial_vectors, state.update(
            trial_vectors=trial_vectors, key=key, F_vect=F_vect, CR_vect=CR_vect, param_vect=param_vect,
        )

    def _ask_one(self, state_inner, ask_one_key, index, param):
        select_key, pbest_key, crossover_key = jax.random.split(ask_one_key, 3)

        basevect_prim_type = (param[0]).astype(int)
        basevect_sec_type = (param[1]).astype(int)
        num_diff_vects = (param[2]).astype(int)
        cross_strategy = (param[3]).astype(int)
        F = param[4]
        CR = param[5]

        population = state_inner.population
        fitness = state_inner.fitness
        best_index = state_inner.best_index

        difference_sum, rand_vect_idx = de_diff_sum(
            select_key,
            self.diff_padding_num,
            num_diff_vects,
            index,
            population,
        )

        # Integrate all base_vect
        rand_vect = population[rand_vect_idx]
        best_vect = population[best_index]
        pbest_vect = select_rand_pbest(pbest_key, self.p, population, fitness)
        current_vect = population[index]
        vector_merge = jnp.stack((rand_vect, best_vect, pbest_vect, current_vect))

        # Select base_vect
        base_vector_prim = vector_merge[basevect_prim_type]
        base_vector_sec = vector_merge[basevect_sec_type]

        base_vector = base_vector_prim + F * (
            base_vector_sec - base_vector_prim
        )

        # Mutation
        mutation_vector = base_vector + difference_sum * F

        # Crossover
        cross_funcs = (
            de_bin_cross,
            de_exp_cross,
            lambda _key, x, y, z: de_arith_recom(x, y, z),
        )
        trial_vector = lax.switch(
            cross_strategy,
            cross_funcs,
            crossover_key,
            mutation_vector,
            current_vect,
            CR,
        )

        trial_vector = jnp.clip(trial_vector, self.lb, self.ub)

        return trial_vector

    def tell(self, state, trial_fitness):
        compare = trial_fitness < state.fitness

        population = jnp.where(
            compare[:, jnp.newaxis], state.trial_vectors, state.population
        )
        fitness = jnp.where(compare, trial_fitness, state.fitness)

        best_index = jnp.argmin(fitness)

        S_param_vect = jnp.where(compare[:, jnp.newaxis], state.param_vect, state.S_param_vect)

        return state.update(
            population=population,
            fitness=fitness,
            best_index=best_index,
            S_param_vect=S_param_vect,
            compare=compare,
        )
