import jax
import jax.numpy as jnp
from jax import vmap, lax
from jax.experimental.host_callback import id_print

from evox import (
    Problem,
    State,
    jit_class,
)
from evox.utils import *

@jit_class
class MetaAlg(Problem): 
    def __init__(self, batch_size, num_runs, problem, iter_DE, base_algorithm):
        super().__init__()
        self.base_algorithm = base_algorithm
        self.problem = problem
        self.iter_DE = iter_DE
        self.batch_size = batch_size
        self.num_runs = num_runs

    def setup(self, key):
        return State(key=key, run_keys=key) 

    def evaluate(self, state, x):  
        run_keys = jax.random.split(state.key, num=self.num_runs)

        iter_DE = lax.select(state.power_up, self.iter_DE * 5, self.iter_DE)
        # iter_DE = self.iter_DE

        def one_step(i, substate_minfit):
            sub_state, min_fit = substate_minfit
            sub_pop, sub_state = self.base_algorithm.ask(sub_state)
            sub_fit, sub_state = self.problem.evaluate(sub_state, sub_pop)
            sub_state = self.base_algorithm.tell(sub_state, sub_fit)

            min_fit = jnp.nanmin(sub_state.fitness)
            return (sub_state, min_fit)

        def reparam_and_eva(param_vect, run_key): 
            self.base_algorithm.differential_weight = param_vect[0]
            self.base_algorithm.cross_probability = param_vect[1]
            self.base_algorithm.basevect_prim_type = jnp.floor(param_vect[2]).astype(int)
            self.base_algorithm.basevect_sec_type = jnp.floor(param_vect[3]).astype(int)
            self.base_algorithm.num_diff_vects = jnp.floor(param_vect[4]).astype(int)
            self.base_algorithm.cross_strategy = jnp.floor(param_vect[5]).astype(int)
            
            sub_state = self.base_algorithm.setup(run_key)
            # sub_state = state.get_child_state("base_algorithm")

            sub_state, min_fit= lax.fori_loop(0, iter_DE, body_fun=one_step, init_val=(sub_state, jnp.inf))

            return min_fit

        tile_x = jnp.tile(x[:, jnp.newaxis, :], (1, self.num_runs, 1))
        tile_runkeys = jnp.tile(run_keys, (self.batch_size, 1, 1))

        fitness_all = vmap(vmap(reparam_and_eva))(tile_x, tile_runkeys)
        fitness = jnp.min(fitness_all, axis=1) # min can be mean/median

        return fitness, state.update(run_keys=run_keys)

        

