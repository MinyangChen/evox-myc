from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, random

from evox import jit_class
from jax.experimental.host_callback import id_print


def _de_mutation(x1, x2, x3, F):
    mutated_pop = x1 + F * (x2 - x3)
    return mutated_pop


def _de_crossover(key, new_x, x, CR):
    batch, dim = x.shape
    random_crossover = random.uniform(key, shape=(batch, dim))
    mask = random_crossover < CR
    return jnp.where(mask, new_x, x)


@jit
def differential_evolve(key, x1, x2, x3, F, CR):
    key, de_key = random.split(key)
    mutated_pop = _de_mutation(x1, x2, x3, F)

    children = _de_crossover(de_key, mutated_pop, x1, CR)
    return children


@jit_class
class DifferentialEvolve:
    def __init__(self, F=0.5, CR=1):
        """
        Parameters
        ----------
        F
            The scaling factor
        CR
            The probability of crossover
        """
        self.F = F
        self.CR = CR

    def __call__(self, key, p1, p2, p3):
        return differential_evolve(key, p1, p2, p3, self.F, self.CR)

@jit
def move_n_small_numbers(array, n):
    # Move the first n small numbers in the array to the first n bits of the array and the rest to the back.
    # The relative order is not changed. n can be a dynamic value.
    sorted_array = jnp.sort(array)
    nth_element = sorted_array[n-1]

    condition = array > nth_element
    sorted_indices = jnp.argsort(condition, kind='stable')
    moved_array = array[sorted_indices]

    return moved_array, sorted_indices


@partial(jit, static_argnames=["diff_padding_num", "replace"])
def de_diff_sum(
    key, diff_padding_num, num_diff_vects, index, population, pop_size_reduced=None, replace=False
):
    pop_size, dim = population.shape

    if pop_size_reduced is None:
        pop_size_fixed = pop_size
    else:
        pop_size_fixed = pop_size_reduced

    """Make differences' indices'"""
    # Randomly select 1 random individual (first index) and (num_diff_vects * 2) difference individuals
    permut = jax.random.choice(key, pop_size, shape=(pop_size,), replace=replace)
    moved_array, _moved_ids = move_n_small_numbers(permut, pop_size_fixed)
    random_choice = moved_array[0: diff_padding_num]

    random_choice = jnp.where(
        random_choice == index, pop_size_fixed - 1, random_choice
    )  # Ensure that selected indices != index

    """Padding: Take the first select_len individuals in the population, and set the rest individuals to 0"""
    pop_permut = population[random_choice]
    select_len = num_diff_vects * 2 + 1
    permut_mask = jnp.arange(diff_padding_num) < select_len
    pop_permut_padding = jnp.where(permut_mask[:, jnp.newaxis], pop_permut, 0)

    """Make difference"""
    diff_vects = pop_permut_padding[1:, :]
    subtrahend_ids = jnp.arange(1, diff_padding_num, 2)
    difference_sum = jnp.sum(diff_vects.at[subtrahend_ids, :].multiply(-1), axis=0)

    rand_vect_idx = random_choice[0]
    return difference_sum, rand_vect_idx

@partial(jit, static_argnames=["diff_padding_num", "num_diff_vects", "replace"])
def de_diff_sum_archive(
    key, diff_padding_num, num_diff_vects, index, population, archive, pop_size_reduced=None, replace=False
):
    pop_size, dim = population.shape
    if pop_size_reduced is None:
        pop_size_fixed = pop_size
    else:
        pop_size_fixed = pop_size_reduced

    """Make differences' indices'"""
    # Randomly select 1 random individual (first index) and (num_diff_vects * 2) difference individuals' indices
    key_select, key_archive = jax.random.split(key)
    population_archive = jnp.vstack((population, archive))

    # Move nan to the last position in population_archive
    isnan_in_pop_arc = jnp.isnan(jnp.sum(population_archive, axis=1))
    _moved_array, sorted_indices = move_n_small_numbers(isnan_in_pop_arc, pop_size_fixed*2)
    population_archive_moved = population_archive[sorted_indices]
    
    # select indices from [0 - 2*pop_szie) as subtrahend

    permut_subtrahend = jax.random.choice(key, pop_size*2, shape=(pop_size*2,), replace=replace)
    moved_array_subtrahend, _moved_ids = move_n_small_numbers(permut_subtrahend, pop_size_fixed*2)
    base_subtrahend_ids = moved_array_subtrahend[0: diff_padding_num]
    
    # select indices from [0 - 2*pop_szie) as minuend, and the subtrahend part ([2, 4, 6, 8]) will be repalced by subtrahend_ids

    permut_base = jax.random.choice(key, pop_size, shape=(pop_size,), replace=replace)
    moved_array_base, _moved_ids = move_n_small_numbers(permut_base, pop_size_fixed)
    base_ids = moved_array_base[0: diff_padding_num]

    even_ids = jnp.arange(2, diff_padding_num, 2)
    diff_member_ids = base_ids.at[even_ids].set(base_subtrahend_ids[even_ids])

    diff_member_ids = jnp.where(
        diff_member_ids == index, pop_size_fixed - 1, diff_member_ids
    )  # Ensure that selected indices != index

    """Padding: Take the first select_len individuals in the population, and set the rest individuals to 0"""
    pop_permut = population_archive_moved[diff_member_ids]
    select_len = num_diff_vects * 2 + 1
    permut_mask = jnp.arange(diff_padding_num) < select_len
    pop_permut_padding = jnp.where(permut_mask[:, jnp.newaxis], pop_permut, 0)

    """Make difference"""
    diff_vects = pop_permut_padding[1:, :]
    subtrahend_ids = jnp.arange(1, diff_padding_num, 2)
    difference_sum = jnp.sum(diff_vects.at[subtrahend_ids, :].multiply(-1), axis=0)

    rand_vect_idx = diff_member_ids[0]
    
    return difference_sum, rand_vect_idx

@partial(jit, static_argnames=["diff_padding_num", "replace"])
def de_diff_sum_rank(
    key, diff_padding_num, num_diff_vects, index, population, k_factor, fitness, pop_size_reduced=None, replace=False
):
    pop_size, dim = population.shape

    if pop_size_reduced is None:
        pop_size_fixed = pop_size
    else:
        pop_size_fixed = pop_size_reduced

    """Make differences' indices'"""
    # Randomly select 1 random individual (first index) and (num_diff_vects * 2) difference individuals
    ids = jnp.arange(pop_size)
    sorted_indices = jnp.argsort(fitness)
    ranks = jnp.argsort(sorted_indices)

    rank_w = k_factor * (pop_size - ranks) + 1

    rank_w_s = jnp.sort(rank_w)
    nth_element = rank_w_s[pop_size-pop_size_reduced]
    rank_w_zero = jnp.where(rank_w < nth_element, 0, rank_w)
    p_weights = rank_w_zero / jnp.sum(rank_w_zero)

    random_choice = jax.random.choice(key, ids, shape=(diff_padding_num,), p=p_weights, replace=replace)
    
    random_choice = jnp.where(
        random_choice == index, pop_size_fixed - 1, random_choice
    )  # Ensure that selected indices != index

    """Padding: Take the first select_len individuals in the population, and set the rest individuals to 0"""
    pop_permut = population[random_choice]
    select_len = num_diff_vects * 2 + 1
    permut_mask = jnp.arange(diff_padding_num) < select_len
    pop_permut_padding = jnp.where(permut_mask[:, jnp.newaxis], pop_permut, 0)

    """Make difference"""
    diff_vects = pop_permut_padding[1:, :]
    subtrahend_ids = jnp.arange(1, diff_padding_num, 2)
    difference_sum = jnp.sum(diff_vects.at[subtrahend_ids, :].multiply(-1), axis=0)

    rand_vect_idx = random_choice[0]
    return difference_sum, rand_vect_idx

@jit
def de_bin_cross(key, mutation_vector, current_vect, CR):
    # Binary crossover: dimension-by-dimension crossover
    # , based on cross_probability to determine the crossover needed for that dimension.
    R_key, mask_key = jax.random.split(key, 2)
    dim = mutation_vector.shape[0]
    R = jax.random.choice(
        R_key, dim
    )  # R is the jrand, i.e. the dimension that must be changed in the crossover
    mask = jax.random.uniform(mask_key, shape=(dim,)) < CR
    mask = mask.at[R].set(True)

    trial_vector = jnp.where(
        mask,
        mutation_vector,
        current_vect,
    )
    return trial_vector


@jit
def de_exp_cross(key, mutation_vector, current_vect, CR):
    # Exponential crossover: Cross the n-th to (n+l-1)-th dimension of the vector,
    # and if n+l-1 exceeds the maximum dimension dim, then make it up from the beginning

    n_key, l_key = jax.random.split(key, 2)
    dim = mutation_vector.shape[0]
    n = jax.random.choice(n_key, jnp.arange(dim))

    # Generate l according to CR. n is the starting dimension to be crossover, and l is the crossover length
    l = jnp.minimum(jax.random.geometric(l_key, CR), dim) - 1
    # Generate mask by n and l
    mask = jnp.arange(dim) < l
    mask = jnp.roll(mask, n, axis=0)
    trial_vector = jnp.where(
        mask,
        mutation_vector,
        current_vect,
    )
    return trial_vector


@jit
def de_arith_recom(mutation_vector, current_vect, K):
    # K can take CR
    trial_vector = current_vect + K * (mutation_vector - current_vect)
    return trial_vector
