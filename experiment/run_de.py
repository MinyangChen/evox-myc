from evox import algorithms, pipelines
from evox.monitors import FitnessMonitor
from evox.problems.CEC2022_benchmark import CEC2022
import jax
import jax.numpy as jnp
import time
 
func_list = jnp.arange(12) + 1
# func_list = [4, 8]
D = 10
steps = 9999999
pop_size = 100
# key_start = 42
key_start_list = jnp.arange(33)# +5 # 33
max_time = 30   ## 10D:30, 20D:60

optimizer =algorithms.DE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, base_vector="rand", num_difference_vectors=1, )
# optimizer =algorithms.SaDE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )
# optimizer =algorithms.JaDE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )
# optimizer =algorithms.CoDE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )
# optimizer =algorithms.ParamDE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )


for txt_name in ['experiment/result_de.txt', 'experiment/result_de_history.txt']:
    with open(txt_name, 'w') as f:
            f.write(f'Problem_Dim: {D}, ')
            f.write(f'Time: {max_time}, ')
            f.write(f'Op timizer: {type(optimizer).__name__}, ')
            f.write(f'Popsize: {pop_size}, ')
            f.write(f'Iters: {steps}\n\n')

"""Run the algorithm"""
time_all = jnp.array([])

for func_num in func_list:
    problem = CEC2022.create(func_num)
    print(type(problem).__name__)
    with open('experiment/result_de.txt', 'a') as f:
        f.write(f'{type(problem).__name__}  ')
    key = jax.random.PRNGKey(42)
    # key = jax.random.split(key)

    for key_start in key_start_list:
        start_time = time.time() 
        monitor = FitnessMonitor()

        # create a pipeline
        pipeline = pipelines.StdPipeline(algorithm=optimizer, problem=problem, fitness_transform=monitor.update)

        # init the pipeline
        key, _ = jax.random.split(key)
        state = pipeline.init(key)

        # run the pipeline for 100 steps
        for i in range(steps):
            state = pipeline.step(state)
            # print(f"min fitness: {monitor.get_min_fitness()}")
            steps_iter = i + 1
            end_time = time.time() 
            elapsed_time = end_time - start_time   
            if elapsed_time >= max_time:
                break
        print(f"min fitness: {monitor.get_min_fitness()}")
        print(f"Steps: {steps_iter}")
        print(f"Time: {elapsed_time} s")

        """Print and record"""
        if key_start >= 2:
            fitness_final = state.get_child_state("algorithm").fitness
            best_ids = jnp.argmin(fitness_final)
            best_fit = fitness_final[best_ids]
            history = monitor.get_history()
            pop_final = state.get_child_state("algorithm").population
            best_solution = pop_final[best_ids]

            with open('experiment/result_de.txt', 'a') as f:
                f.write(f'{best_fit} ')
    FEs = steps_iter * pop_size
    with open('experiment/result_de.txt', 'a') as f:
        f.write(f'{FEs}')
        f.write('\n')


    num_samples = 100
    history = jnp.array(history)
    indices = jnp.linspace(0, len(history)-1, num_samples, dtype=int)
    sampled_arr = history[indices]
    sampled_arr = sampled_arr.tolist()
    with open('experiment/result_de_history.txt', 'a') as f:
        f.write(f'{type(problem).__name__}\n')
        f.write(f'{sampled_arr}\n\n')