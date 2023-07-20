import jax
import jax.numpy as jnp
from jax import vmap, lax
from jax.experimental.host_callback import id_print
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
from evox.utils import *
from evox.problems.CEC2022_benchmark import CEC2022

"""The following is the main procedure"""

func_list = jnp.arange(12) + 1
# func_list = [11, 9]
D = 10
num_runs = 1
iter_DE = 500  ## 10D:500, 20D:800, 
iter_outer = 999999

de_popsize = 100
pop_size = 100
# batch_size = int(pop_size/2)
batch_size = pop_size

key_start_list = jnp.arange(3) +5# 33
max_time = 30   ## 10D:30, 20D:60

#tiny_num prevents variables from being taken to the upper bound (outside the strategy parameter range)
tiny_num = 1e-5     
# Variables: F, CR, base_vector, num_differences, cross_category
# lb=jnp.array([0, 0, 0])
# ub=jnp.array([2, 5, 5])
# lb = jnp.array([0, 0, 1,         1, 0])
# ub = jnp.array([1, 1, 1.99, 1.99, 0.99])
# lb = jnp.array([0, 0, 0,          0,            0])
# ub = jnp.array([1, 1, 5-tiny_num, 5-tiny_num, 2-tiny_num])
lb = jnp.array([0, 0, 0,         0,          1,            0])
ub = jnp.array([1, 1, 4-tiny_num, 4-tiny_num, 5-tiny_num, 3-tiny_num])

algorithm = algorithms.DE(lb=lb, ub=ub, pop_size=pop_size, base_vector="rand", differential_weight=0.5, cross_probability=0.9)
# algorithm = algorithms.CSO(lb=lb, ub=ub, pop_size=pop_size)
# algorithm = algorithms.PSO(lb=lb, ub=ub, pop_size=pop_size)
# algorithm = algorithms.SimpleES(lb=lb, ub=ub, pop_size=pop_size)

base_algorithm = algorithms.ParamDE(jnp.full((D,), -100), jnp.full((D,), 100), pop_size=de_popsize)

for txt_name in ['experiment/result_meta.txt', 'experiment/result_history_hyperpop.txt']:
    with open(txt_name, 'w') as f:
            f.write(f'Problem_Dim: {D}, ')
            f.write(f'Time: {max_time}, ')
            f.write(f'Outer_Optimizer: {type(algorithm).__name__}, ')
            f.write(f'Outer_Popsize: {batch_size}, ')
            f.write(f'Outer_Iters: {iter_outer}, ')
            f.write(f'DE_Runs: {num_runs}, ')
            f.write(f'DE_Popsize: {de_popsize}, ')
            f.write(f'DE_Iters: {iter_DE}\n\n')


"""————————————————————————————————————————————"""
time_all = jnp.array([])
for func_num in func_list:
    problem_instance = CEC2022.create(func_num)
    print(type(problem_instance).__name__)
    with open('experiment/result_meta.txt', 'a') as f:
        f.write(f'{type(problem_instance).__name__}  ')
    for key_start in key_start_list:
        start_time = time.time()
        power_up = 0
        last_iter = 0
        # problem_instance = problems.classic.Ackley()
        meta_problem = problems.MetaAlg(batch_size=batch_size, num_runs=num_runs, problem=problem_instance, iter_DE=iter_DE, base_algorithm=base_algorithm)
        
        monitor = monitors.FitnessMonitor()
        pipeline = pipelines.StdPipeline(
            algorithm=algorithm,
            problem=meta_problem,
            fitness_transform=monitor.update,
        )

        state = pipeline.init(jax.random.PRNGKey(key_start))

        for i in range(iter_outer):
            state = state.update_child("problem", {"power_up": power_up})
            # print(f"min fitness: {monitor.get_min_fitness()}")
            state = pipeline.step(state)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if elapsed_time > max_time - 1:
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
        if key_start >= 2:
            fitness_final = state.get_child_state("algorithm").fitness
            best_ids = jnp.argmin(fitness_final)
            best_fit = fitness_final[best_ids]

            pop_final = state.get_child_state("algorithm").population
            best_params = pop_final[best_ids]
            best_params_print = best_params.at[2:].set(jnp.floor(best_params[2:]))
            # print(jnp.round(best_params_print, 2))

            best_fit_record = monitor.get_min_fitness()
            with open('experiment/result_meta.txt', 'a') as f:
                f.write(f'{best_fit_record} ')

    FEs = (steps_iter * batch_size) * num_runs * (iter_DE * de_popsize) + iter_DE * 5 * de_popsize
    with open('experiment/result_meta.txt', 'a') as f:
        f.write(f'{FEs}')
        f.write('\n')
        # f.write(f'Time:{elapsed_time} s\n  ')

    with open('experiment/result_history_hyperpop.txt', 'a') as f:
        f.write(f'{type(problem_instance).__name__}\n')
        f.write(f'{best_params}\n')
        f.write(f'{fit_history}\n\n')