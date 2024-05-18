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
class EVDE(Algorithm):
    """Wu G, Shen X, Li H, et al. Ensemble of differential evolution variants[J]. Information Sciences, 2018, 423: 172-186."""

    def __init__(
        # Basic hyper-parameters
        self,
        lb,
        ub,
        pop_size=100,
        diff_padding_num=5,
        differential_weight=None,
        cross_probability=None,
        p=0.05,

        # JaDE hyper-parameters
        c=0.1,

        # EVDE hyper-parameters
        ng = 20,
        lambda_1 = 0.1,
        lambda_2 = 0.1,
        lambda_3 = 0.1,
        lambda_4 = 0.7,
    ):
        # Basic hyper-parameters
        self.dim = lb.shape[0]
        self.lb = lb
        self.ub = ub
        self.pop_size = pop_size
        self.diff_padding_num = diff_padding_num
        self.cross_probability = cross_probability
        self.differential_weight = differential_weight
        self.num_diff_vects = 1
        self.p = p

        # JaDE hyper-parameters
        self.JaDE_strategy = current2pbest_1_bin
        self.c = c

        # CoDE hyper-parameters
        self.CoDE_FCR_pool=jnp.array([[1, 0.1], [1, 0.9], [0.8, 0.2]])
        self.CoDE_strategies = jnp.array([rand_1_bin, rand_2_bin, current2rand_1])

        # EPSDE hyper-parameters
        self.EPSDE_F_pool = jnp.arange(4, 10) / 10.0
        self.EPSDE_CR_pool = jnp.arange(1, 10) / 10.0
        self.EPSDE_strategies = jnp.array([rand_1_bin, best_2_bin, current2rand_1_bin])

        # EVDE hyper-parameters
        self.ng = ng
        self.pop_size_1 = round(lambda_1 * pop_size)
        self.pop_size_2 = round(lambda_2 * pop_size)
        self.pop_size_2_expanded = round(lambda_2 * pop_size * 3)
        self.pop_size_3 = round(lambda_3 * pop_size)
        self.pop_size_4 = round(lambda_4 * pop_size)
        self.pop_size_expanded = round((lambda_1 + lambda_2*3 + lambda_3 + lambda_4) * pop_size)
        # _expanded means the CoDE parts are 3 times larger (length=260). _standard means the original popsize (length=100)

    def setup(self, key):
        state_key, init_key, strategy_key, F_key, CR_key, reward_strategy_key, reward_F_key, reward_CR_key = jax.random.split(key, 8)
        population = jax.random.uniform(init_key, shape=(self.pop_size, self.dim))
        population = population * (self.ub - self.lb) + self.lb
        fitness = jnp.full((self.pop_size,), jnp.inf)
        best_index = 0

        # Initialize EPSDE's param_vect
        strategies_ids = jax.random.choice(strategy_key, a=3, shape=(self.pop_size_3,), replace=True)
        F_ids = jax.random.choice(F_key, a=6, shape=(self.pop_size_3, 1), replace=True)
        CR_ids = jax.random.choice(CR_key, a=9, shape=(self.pop_size_3, 1), replace=True)
        strategies_vect = self.EPSDE_strategies[strategies_ids]
        EPSDE_F_vect = self.EPSDE_F_pool[F_ids]
        EPSDE_CR_vect = self.EPSDE_CR_pool[CR_ids]
        EPSDE_param_vect = jnp.concatenate((strategies_vect, EPSDE_F_vect, EPSDE_CR_vect), axis=1)

        # Initialize reward EPSDE's param_vect
        reward_strategies_ids = jax.random.choice(reward_strategy_key, a=3, shape=(self.pop_size_4,), replace=True)
        reward_F_ids = jax.random.choice(reward_F_key, a=6, shape=(self.pop_size_4, 1), replace=True)
        reward_CR_ids = jax.random.choice(reward_CR_key, a=9, shape=(self.pop_size_4, 1), replace=True)
        reward_strategies_vect = self.EPSDE_strategies[reward_strategies_ids]
        reward_EPSDE_F_vect = self.EPSDE_F_pool[reward_F_ids]
        reward_EPSDE_CR_vect = self.EPSDE_CR_pool[reward_CR_ids]
        reward_EPSDE_param_vect = jnp.concatenate((reward_strategies_vect, reward_EPSDE_F_vect, reward_EPSDE_CR_vect), axis=1)

        return State(
            population=population,
            fitness=fitness,
            best_index=best_index,
            key=state_key,

            # JaDE's state parameters
            F_u=0.5,
            CR_u=0.5,

            EPSDE_param_vect=EPSDE_param_vect,
            EPSDE_S_param_vect=EPSDE_param_vect,
            EPSDE_compare=jnp.full((self.pop_size_3,), True),

            iter=0,
            JaDE_delta_sum=0,
            CoDE_delta_sum=0,
            EPSDE_delta_sum=0,
            best_alg_ng=0,
            reward_alg_id=0,
            reward_EPSDE_param_vect=reward_EPSDE_param_vect,
            reward_EPSDE_S_param_vect=reward_EPSDE_param_vect,
            reward_EPSDE_compare=jnp.full((self.pop_size_4,), True),
        )

    def ask(self, state):
        key, ask_one_key, strategy_key, param_renew_key, F_key, CR_key, reward_key_1, reward_key_2 = jax.random.split(state.key, 8)
        ask_one_keys = jax.random.split(ask_one_key, self.pop_size_expanded)
        indices = jnp.arange(self.pop_size_expanded)

        """Input parameters (pop_size rows with 6 column) into _ask_one.
        param_size_JaDE = lambda_1*pop_size, param_size_CoDE = lambda_2*pop_size*3, param_size_EPSDE = lambda_3*pop_size.
        param_size_reward = lambda_4*pop_size."""

        # Generare JaDE's parameters
        JaDE_F_vect = jax.random.cauchy(F_key, shape=(self.pop_size_1,)) * 0.1 + state.F_u
        JaDE_F_vect = jnp.clip(JaDE_F_vect, jnp.zeros(self.pop_size_1), jnp.ones(self.pop_size_1))
        JaDE_CR_vect = jax.random.normal(CR_key, shape=(self.pop_size_1,)) * 0.1 + state.CR_u
        JaDE_CR_vect = jnp.clip(JaDE_CR_vect, jnp.zeros(self.pop_size_1), jnp.ones(self.pop_size_1))
        JaDE_param_vect = jnp.concatenate((jnp.tile(self.JaDE_strategy, (self.pop_size_1, 1)), JaDE_F_vect[:, jnp.newaxis], JaDE_CR_vect[:, jnp.newaxis]), axis=1)

        # Generare CoDE's parameters
        CoDE_FCR_ids = jax.random.choice(param_renew_key, a=3, shape=(self.pop_size_2_expanded,), replace=True)
        CoDE_FCR_vect = self.CoDE_FCR_pool[CoDE_FCR_ids]
        CoDE_strategies_vect = jnp.concatenate(
            (jnp.tile(self.CoDE_strategies[0], (self.pop_size_2, 1)), 
             jnp.tile(self.CoDE_strategies[1], (self.pop_size_2, 1)), 
             jnp.tile(self.CoDE_strategies[2], (self.pop_size_2, 1))), 
             axis=0
        )

        CoDE_param_vect = jnp.concatenate((CoDE_strategies_vect, CoDE_FCR_vect), axis=1)

        # Generare EPSDE's parameters
        EPSDE_strategies_ids = jax.random.choice(strategy_key, a=3, shape=(self.pop_size_3,), replace=True)
        EPSDE_F_ids = jax.random.choice(F_key, a=6, shape=(self.pop_size_3, 1), replace=True)
        EPSDE_CR_ids = jax.random.choice(CR_key, a=9, shape=(self.pop_size_3, 1), replace=True)

        EPSDE_strategies_vect = self.EPSDE_strategies[EPSDE_strategies_ids]
        EPSDE_F_vect = self.EPSDE_F_pool[EPSDE_F_ids]
        EPSDE_CR_vect = self.EPSDE_CR_pool[EPSDE_CR_ids]
        EPSDE_param_vect_random = jnp.concatenate((EPSDE_strategies_vect, EPSDE_F_vect, EPSDE_CR_vect), axis=1)

        EPSDE_renew_mask = jax.random.choice(param_renew_key, a=2, shape=(self.pop_size_3,), replace=True)
        EPSDE_param_vect_renew = jnp.where(EPSDE_renew_mask[:, jnp.newaxis], state.EPSDE_S_param_vect, EPSDE_param_vect_random)

        EPSDE_param_vect = jnp.where(state.EPSDE_compare[:, jnp.newaxis], state.EPSDE_param_vect, EPSDE_param_vect_renew)

        # Generare Reward's parameters
        reward_J_F_vect = jax.random.cauchy(reward_key_1, shape=(self.pop_size_4,)) * 0.1 + state.F_u
        reward_J_F_vect = jnp.clip(reward_J_F_vect, jnp.zeros(self.pop_size_4), jnp.ones(self.pop_size_4))
        reward_J_CR_vect = jax.random.normal(reward_key_1, shape=(self.pop_size_4,)) * 0.1 + state.CR_u
        reward_J_CR_vect = jnp.clip(reward_J_CR_vect, jnp.zeros(self.pop_size_4), jnp.ones(self.pop_size_4))
        reward_J_param_vect = jnp.concatenate((jnp.tile(self.JaDE_strategy, (self.pop_size_4, 1)), reward_J_F_vect[:, jnp.newaxis], reward_J_CR_vect[:, jnp.newaxis]), axis=1)


        reward_EPSDE_strategies_ids = jax.random.choice(reward_key_1, a=3, shape=(self.pop_size_4,), replace=True)
        reward_EPSDE_F_ids = jax.random.choice(reward_key_1, a=6, shape=(self.pop_size_4, 1), replace=True)
        reward_EPSDE_CR_ids = jax.random.choice(reward_key_2, a=9, shape=(self.pop_size_4, 1), replace=True)

        reward_EPSDE_strategies_vect = self.EPSDE_strategies[reward_EPSDE_strategies_ids]
        reward_EPSDE_F_vect = self.EPSDE_F_pool[reward_EPSDE_F_ids]
        reward_EPSDE_CR_vect = self.EPSDE_CR_pool[reward_EPSDE_CR_ids]
        reward_EPSDE_param_vect_random = jnp.concatenate((reward_EPSDE_strategies_vect, reward_EPSDE_F_vect, reward_EPSDE_CR_vect), axis=1)

        reward_EPSDE_renew_mask = jax.random.choice(param_renew_key, a=2, shape=(self.pop_size_4,), replace=True)
        reward_EPSDE_param_vect_renew = jnp.where(reward_EPSDE_renew_mask[:, jnp.newaxis], state.reward_EPSDE_S_param_vect, reward_EPSDE_param_vect_random)

        reward_EPSDE_param_vect = jnp.where(state.reward_EPSDE_compare[:, jnp.newaxis], state.reward_EPSDE_param_vect, reward_EPSDE_param_vect_renew)
        
        reward_param_vect = lax.select(state.reward_alg_id==0, reward_J_param_vect, reward_EPSDE_param_vect)

        # Merge all the param_vect
        param_vect_expanded = jnp.concatenate(
            (JaDE_param_vect, CoDE_param_vect, EPSDE_param_vect, reward_param_vect), axis=0)

        # The trial_vectors has 260 vectors (if pop_size = 100)
        trial_vectors = vmap(
            partial(
                self._ask_one,
                state_inner=state,
            )
        )(ask_one_key=ask_one_keys, index=indices, param=param_vect_expanded)

        return trial_vectors, state.update(
            trial_vectors=trial_vectors, key=key, param_vect_expanded=param_vect_expanded,
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
        """Split expanded pop/fit into different DE variants"""
        trial_fitness_expanded = trial_fitness
        trial_vectors_expanded = state.trial_vectors


        JaDE_trial_fitness = trial_fitness_expanded[0: self.pop_size_1]
        JaDE_trial_vectors = trial_vectors_expanded[0: self.pop_size_1]

        CoDE_trial_fitness = trial_fitness_expanded[self.pop_size_1: self.pop_size_1 + self.pop_size_2_expanded]
        CoDE_trial_vectors = trial_vectors_expanded[self.pop_size_1: self.pop_size_1 + self.pop_size_2_expanded]

        EPSDE_trial_fitness = trial_fitness_expanded[self.pop_size_1 + self.pop_size_2_expanded: self.pop_size_1 + self.pop_size_2_expanded + self.pop_size_3]
        EPSDE_trial_vectors = trial_vectors_expanded[self.pop_size_1 + self.pop_size_2_expanded: self.pop_size_1 + self.pop_size_2_expanded + self.pop_size_3]

        reward_trial_fitness = trial_fitness_expanded[self.pop_size_1 + self.pop_size_2_expanded + self.pop_size_3:]
        reward_trial_vectors = trial_vectors_expanded[self.pop_size_1 + self.pop_size_2_expanded + self.pop_size_3:]

        """Convert expanded trial_fitness to standard trial_fitness (length=100)."""
        indices = jnp.arange(self.pop_size_2_expanded).reshape(3, self.pop_size_2)
        trans_fit = CoDE_trial_fitness[indices]
        min_indices = jnp.argmin(trans_fit, axis=0)

        min_indices_global = indices[min_indices, jnp.arange(self.pop_size_2)]
        CoDE_trial_fitness_standard = CoDE_trial_fitness[min_indices_global]
        CoDE_trial_vectors_standard = CoDE_trial_vectors[min_indices_global]

        trial_fitness_standard = jnp.concatenate((JaDE_trial_fitness, CoDE_trial_fitness_standard, EPSDE_trial_fitness, reward_trial_fitness), axis=0)
        trial_vectors_standard = jnp.concatenate((JaDE_trial_vectors, CoDE_trial_vectors_standard, EPSDE_trial_vectors, reward_trial_vectors), axis=0)

        """Standard selection"""
        compare = trial_fitness_standard <= state.fitness

        population = jnp.where(
            compare[:, jnp.newaxis], trial_vectors_standard, state.population
        )
        fitness = jnp.where(compare, trial_fitness_standard, state.fitness)

        best_index = jnp.argmin(fitness)

        """Making adpation for each DE variants"""
        # JaDE adpation
        FCR_expanded = state.param_vect_expanded[:, 4:]
        JaDE_FCR = FCR_expanded[0: self.pop_size_1]
        JaDE_compare = JaDE_trial_fitness < state.fitness[0: self.pop_size_1]

        S_F_init = jnp.full(shape=(self.pop_size_1,), fill_value=jnp.nan)
        S_CR_init = jnp.full(shape=(self.pop_size_1,), fill_value=jnp.nan)

        S_F = jnp.where(JaDE_compare, JaDE_FCR[:, 0], S_F_init)
        S_CR = jnp.where(JaDE_compare, JaDE_FCR[:, 1], S_CR_init)

        no_success = jnp.all(~JaDE_compare)

        F_u_temp = (1 - self.c) * state.F_u + self.c * (
            jnp.nansum(S_F**2) / jnp.nansum(S_F)
        )
        F_u = lax.select(no_success, state.F_u, F_u_temp)

        CR_u_temp = (1 - self.c) * state.CR_u + self.c * jnp.nanmean(S_CR)
        CR_u = lax.select(no_success, state.CR_u, CR_u_temp)

        # EPSDE adpation
        EPSDE_FCR = state.param_vect_expanded[self.pop_size_1 + self.pop_size_2_expanded: self.pop_size_1 + self.pop_size_2_expanded + self.pop_size_3]
        EPSDE_compare = EPSDE_trial_fitness < state.fitness[self.pop_size_1+self.pop_size_2: self.pop_size_1+self.pop_size_2+self.pop_size_3]
        EPSDE_S_param_vect = jnp.where(EPSDE_compare[:, jnp.newaxis], EPSDE_FCR, state.EPSDE_S_param_vect)

        reward_EPSDE_FCR = state.param_vect_expanded[self.pop_size_1 + self.pop_size_2_expanded + self.pop_size_3: ]
        reward_EPSDE_compare = reward_trial_fitness < state.fitness[self.pop_size_1+self.pop_size_2+self.pop_size_3: ]
        reward_EPSDE_S_param_vect = jnp.where(reward_EPSDE_compare[:, jnp.newaxis], reward_EPSDE_FCR, state.reward_EPSDE_S_param_vect)

        """Determine the best DE variant"""
        JaDE_bestfit = jnp.min(fitness[0: self.pop_size_1])
        old_JaDE_bestfit = jnp.min(state.fitness[0: self.pop_size_1])
        JaDE_delta = old_JaDE_bestfit - JaDE_bestfit
        JaDE_delta_sum = JaDE_delta + state.JaDE_delta_sum

        CoDE_bestfit = jnp.min(fitness[self.pop_size_1: self.pop_size_1+self.pop_size_2])
        old_CoDE_bestfit = jnp.min(state.fitness[self.pop_size_1: self.pop_size_1+self.pop_size_2])
        CoDE_delta = old_CoDE_bestfit - CoDE_bestfit
        CoDE_delta_sum = CoDE_delta + state.CoDE_delta_sum

        EPSDE_bestfit = jnp.min(fitness[self.pop_size_1+self.pop_size_2: self.pop_size_1+self.pop_size_2+self.pop_size_3])
        old_EPSDE_bestfit = jnp.min(state.fitness[self.pop_size_1+self.pop_size_2: self.pop_size_1+self.pop_size_2+self.pop_size_3])
        EPSDE_delta = old_EPSDE_bestfit - EPSDE_bestfit
        EPSDE_delta_sum = EPSDE_delta + state.EPSDE_delta_sum

        cheak_delta = jnp.mod(state.iter, self.ng) == 0

        best_alg_gen = jnp.argmax(jnp.array([JaDE_delta_sum / self.pop_size_1, CoDE_delta_sum / self.pop_size_2_expanded, EPSDE_delta_sum / self.pop_size_3]))
        best_alg_gen_float = (best_alg_gen).astype(float)
        best_alg_ng = lax.select(cheak_delta, best_alg_gen_float, jnp.nan)

        no_change_reward = jnp.isnan(best_alg_ng)
        select_alg_id = lax.select(best_alg_ng==0, 0, 2)
        reward_alg_id = lax.select(no_change_reward, state.reward_alg_id, select_alg_id)

        JaDE_delta_sum = lax.select(cheak_delta, 0.0, JaDE_delta_sum)
        CoDE_delta_sum = lax.select(cheak_delta, 0.0, CoDE_delta_sum)
        EPSDE_delta_sum = lax.select(cheak_delta, 0.0, EPSDE_delta_sum)

        return state.update(
            population=population,
            fitness=fitness,
            best_index=best_index,

            F_u=F_u,
            CR_u=CR_u,
            EPSDE_S_param_vect=EPSDE_S_param_vect,
            EPSDE_compare=EPSDE_compare,

            JaDE_delta_sum=JaDE_delta_sum,
            CoDE_delta_sum=CoDE_delta_sum,
            EPSDE_delta_sum=EPSDE_delta_sum,
            best_alg_ng=best_alg_ng,
            reward_alg_id=reward_alg_id,
            reward_EPSDE_S_param_vect=reward_EPSDE_S_param_vect,
            reward_EPSDE_compare=reward_EPSDE_compare,
        )
