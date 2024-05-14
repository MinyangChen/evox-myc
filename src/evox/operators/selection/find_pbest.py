from evox.core.state import State
import jax
import jax.numpy as jnp
from jax import jit

from functools import partial
from evox.operators.crossover import move_n_small_numbers
from jax.experimental.host_callback import id_print


@jit 
def select_rand_pbest(key, percent, population, fitness):
    # Randomly select a member in top 100p% best individuals. p can be dynamic.
    pop_size = population.shape[0]
    top_p_num = jnp.floor(pop_size * percent).astype(int)  # p ranges in 5% - 20%.

    _moved_fitness, moved_ids = move_n_small_numbers(fitness, top_p_num)
    moved_population = population[moved_ids]

    random_ids = jax.random.choice(key, pop_size, shape=(pop_size,), replace=False)
    moved_random_ids, _moved_ids = move_n_small_numbers(random_ids, top_p_num)

    pbest_index = moved_random_ids[0]
    pbest_vect = moved_population[pbest_index]

    return pbest_vect