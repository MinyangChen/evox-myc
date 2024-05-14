import sys
sys.path.append('/home/chenmiy/DE/evox-myc/src')

from evox import algorithms, workflows
from evox.monitors import StdSOMonitor
from evox.problems.numerical.cec2022_so import CEC2022TestSuit
import jax
import jax.numpy as jnp
import time

func_list = jnp.arange(12) + 1
# func_list = [6]
D = 20
steps = 9999999
pop_size = 100
# key_start = 42
runs = 2 # number of independent runs. should be an even number
max_time = 60   ## 60
num_samples = 100 # history sample num
key = jax.random.PRNGKey(42)

# optimizer =algorithms.de_variants.DE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, base_vector="rand", num_difference_vectors=1, )
# optimizer =algorithms.de_variants.SaDE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )
# optimizer =algorithms.de_variants.JaDE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )
# optimizer =algorithms.de_variants.CoDE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )
# optimizer =algorithms.de_variants.SHADE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )
optimizer =algorithms.de_variants.LSHADE(lb=jnp.full(shape=(D,), fill_value=-100), ub=jnp.full(shape=(D,), fill_value=100), pop_size=pop_size, )

def sample_history(num_samples, fit_history):
    fit_history = jnp.array(fit_history)
    float_indices = jnp.linspace(0, len(fit_history)-1, num_samples)
    indices = jnp.floor(float_indices).astype(int)
    sampled_arr = fit_history[indices]
    sampled_arr = sampled_arr.tolist()
    return sampled_arr


for txt_name in ['run/result_de.txt', 'run/result_de_history.txt']:
    with open(txt_name, 'w') as f:
            f.write(f'Problem_Dim: {D}, ')
            f.write(f'Time: {max_time}, ')
            f.write(f'Op timizer: {type(optimizer).__name__}, ')
            f.write(f'Popsize: {pop_size}, ')
            f.write(f'Iters: {steps}\n\n')

"""Run the algorithm"""
time_all = jnp.array([])

for func_num in func_list:
    problem = CEC2022TestSuit.create(int(func_num))
    print(type(problem).__name__)
    
    with open('run/result_de.txt', 'a') as f:
        f.write(f'{type(problem).__name__}  ')
    with open('run/result_de_history.txt', 'a') as f:
        f.write(f'{type(problem).__name__}  ')

    best_fit_history_all = []   # history of all runs
    best_fit_all = []           # best fit of all runs

    for run_num in range(runs):
        monitor = StdSOMonitor(record_fit_history=False)

        # create a pipeline
        workflow = workflows.StdWorkflow(algorithm=optimizer, problem=problem, monitor=monitor,)

        # init the pipeline
        key, _ = jax.random.split(key)
        state = workflow.init(key)

        bestfit_history = []
        start_time = time.time() 
        # run the pipeline for 100 steps
        for i in range(steps):
            state = workflow.step(state)
            steps_iter = i + 1

            bestfit_step = monitor.get_best_fitness().item() # record best fitness history
            bestfit_history.append(bestfit_step)
            # print(bestfit_step) ###

            end_time = time.time() 
            elapsed_time = end_time - start_time   

            # Update progress into algorithm
            progress = elapsed_time / max_time
            alg_state = state.get_child_state("algorithm")
            alg_state = alg_state.update(progress=progress)
            state = state.update_child("algorithm", alg_state) 
            if elapsed_time >= max_time:
                break
        """Record and print"""
        bestfit_history = sample_history(num_samples, bestfit_history)
        best_fit = monitor.get_best_fitness()
        print(f"min fitness: {best_fit}")
        print(f"Steps: {steps_iter} Runs: {run_num}")
        print(f"Time: {elapsed_time} s\n")

        if run_num >= 1:
            best_fit_all.append(best_fit)
            best_fit_history_all.append(bestfit_history)

            with open('run/result_de.txt', 'a') as f:
                f.write(f'{best_fit} ')


    FEs = steps_iter * pop_size
    with open('run/result_de.txt', 'a') as f:
        f.write(f'{FEs}\n')

    # find the median run for history
    sorted_best_fit_all = sorted(best_fit_all)
    median_value  = sorted_best_fit_all[len(sorted_best_fit_all) // 2]
    median_index = best_fit_all.index(median_value)

    best_fit_history_median = best_fit_history_all[median_index]

    with open('run/result_de_history.txt', 'a') as f:
        f.write(f'{best_fit_history_median}\n')