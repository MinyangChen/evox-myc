import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import vmap
from functools import partial
import time
from evox import (
    Algorithm,
    Problem,
    State,
    algorithms,
    jit_class,
    monitors,
    pipelines,
    problems,
)
from evox.utils import compose
from evox.problems.metade import *
from evox.problems.CEC2022_benchmark import CEC2022

"""Settings"""
## Problem setting
D = 10          #* 10D 20D
FUNC_LIST = jnp.arange(12) + 1
# FUNC_LIST = [6]
## Meta setting
BATCH_SIZE = 100
NUM_RUNS = 1
MAX_TIME = 30   #* 10D:30, 20D:60
runs = 33 # 33
key_start = 42

## Outer optimizer setting
STEPS = 999999
POP_SIZE = BATCH_SIZE

## base algorithm setting
BASE_ALG_POP_SIZE = 100
BASE_ALG_STEPS = 500  # 10D:500, 20D:800

## DE parameter boundary setting
tiny_num = 1e-5     
param_lb = jnp.array([0, 0, 0,         0,          1,            0])
param_ub = jnp.array([1, 1, 4-tiny_num, 4-tiny_num, 5-tiny_num, 3-tiny_num])

"""Run"""

algorithm = algorithms.DE(
    lb=param_lb,
    ub=param_ub,
    pop_size=POP_SIZE,
    base_vector="rand", differential_weight=0.5, cross_probability=0.9
)

BatchDE = create_batch_algorithm(algorithms.ParamDE, BATCH_SIZE, NUM_RUNS)
batch_de = BatchDE(
    lb=jnp.full((D,), -100),
    ub=jnp.full((D,), 100),
    pop_size=BASE_ALG_POP_SIZE,
)

for txt_name in ['experiment/result_meta.txt', 'experiment/result_history_hyperpop.txt']:
    with open(txt_name, 'w') as f:
        f.write(f'Problem_Dim: {D}, ')
        f.write(f'Time: {MAX_TIME}, ')
        f.write(f'Outer_Optimizer: {type(algorithm).__name__}, ')
        f.write(f'Outer_Popsize: {POP_SIZE}, ')
        f.write(f'Outer_Iters: {STEPS}, ')
        f.write(f'DE_Runs: {NUM_RUNS}, ')
        f.write(f'DE_Popsize: {BASE_ALG_POP_SIZE}, ')
        f.write(f'DE_Iters: {BASE_ALG_STEPS}\n\n')

for func_num in FUNC_LIST:
    base_problem = CEC2022.create(func_num)
    decoder = decoder_de
    key = jax.random.PRNGKey(key_start)
    with open('experiment/result_meta.txt', 'a') as f:
        f.write(f'{type(base_problem).__name__}  ')
    for run_num in range(runs):
        monitor = monitors.FitnessMonitor()
        print(type(base_problem).__name__)
        meta_problem = MetaDE(
            batch_de,
            base_problem,
            batch_size=BATCH_SIZE,
            num_runs=NUM_RUNS,
            base_alg_steps=BASE_ALG_STEPS
        )

        pipeline = pipelines.StdPipeline(
            algorithm=algorithm,
            problem=meta_problem,
            pop_transform=decoder,
            fitness_transform=monitor.update,
        )
        key, _ = jax.random.split(key)
        state = pipeline.init(key)

        start_time = time.time()
        power_up = 0
        last_iter = 0
        for i in range(STEPS):
            state = state.update_child("problem", {"power_up": power_up})
            state = pipeline.step(state)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time > MAX_TIME - 1:
                power_up = 1
                if last_iter:
                    break
                last_iter = 1
            steps_iter = i + 1
        print(f"min fitness: {monitor.get_min_fitness()}")
        print(f"Steps: {steps_iter}")
        print(f"Time: {elapsed_time} s")
        fit_history = monitor.get_history()

        """Print and record"""
        if run_num >= 2:
            fitness_final = state.get_child_state("algorithm").fitness
            best_ids = jnp.argmin(fitness_final)
            best_fit = fitness_final[best_ids]

            pop_final = state.get_child_state("algorithm").population
            best_params = pop_final[best_ids]

            best_fit_record = monitor.get_min_fitness()
            with open('experiment/result_meta.txt', 'a') as f:
                f.write(f'{best_fit_record} ')
    FEs = (steps_iter * BATCH_SIZE) * NUM_RUNS * (BASE_ALG_STEPS * BASE_ALG_POP_SIZE) + 5 * (BASE_ALG_STEPS * BASE_ALG_POP_SIZE)
    with open('experiment/result_meta.txt', 'a') as f:
        f.write(f'{FEs}')
        f.write('\n')

    with open('experiment/result_history_hyperpop.txt', 'a') as f:
        f.write(f'{type(base_problem).__name__}\n')
        f.write(f'{best_params}\n')
        f.write(f'{fit_history}\n\n')