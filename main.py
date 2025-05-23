import numpy as np
import matplotlib.pyplot as plt
import time

def visualization_with_variance(file_name):
    x_values = []
    y_values = []
    variances = []

    try:
        with open(file_name, 'r') as file:
            for line in file:
                x, y, var = map(float, line.strip().split(','))
                x_values.append(x)
                y_values.append(y)
                variances.append(var)
    except FileNotFoundError:
        print(f"Klaida: Failas '{file_name}' nerastas.")
        return

    y_values = np.array(y_values)
    variances = np.array(variances)


    plt.plot(x_values, y_values, label='Average number of iterations', color = 'orange')
    plt.fill_between(x_values, np.maximum(y_values - variances, 0), y_values + variances, alpha=0.2, color='orange', label='standard deviation')
    plt.xlabel('Mutation rate')
    plt.ylabel('Average number of Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()


def real_to_binary(x, min_val=-11, max_val=11, bits=16):
    scaled = (x - min_val) / (max_val - min_val)
    int_val = int(scaled * (2**bits - 1))
    return bin(int_val)[2:].zfill(bits)

def binary_to_real(x, min_val=-11, max_val=11, bits=16):
    if x.startswith('0b'):
        x = x[2:]
    try:
        int_val = int(x, 2)
    except ValueError:
        return 0
    scaled = int_val / (2**bits - 1)
    return scaled * (max_val - min_val) + min_val

def fitness(x):
    if (x > -11) and (x < 11):
        y = (x**2 + 1) * np.cos(15 * x**2)
        return y
    else:
        return 0

def crossover(parents, population_size):
    n = len(parents)
    children = []
    while len(children) < population_size:
        for i in range(0, n, 2):
            if i + 1 < n:
                parent1 = parents[i]
                parent2 = parents[i + 1]
                crossover_point = int(np.random.uniform(1, len(parent1) - 1))
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                children.extend([child1, child2])
                if len(children) >= population_size:
                    break
            else:
                children.append(parents[i])
                if len(children) >= population_size:
                    break
    return children[:population_size]

def local_search(parent, fitness_function, step_size=0.05, iterations=5):
    real_parent = binary_to_real(parent)
    best_parent = real_parent
    best_fitness = fitness_function(real_parent)

    for _ in range(iterations):
        for i in [-step_size, step_size]:
            neighbor = real_parent + i
            neighbor_fitness = fitness_function(neighbor)

            if neighbor_fitness > best_fitness:
                best_fitness = neighbor_fitness
                best_parent = neighbor

    return real_to_binary(best_parent)

def mutate(parents, mutation_rate):
    mutated_parents = []
    for parent in parents:
        mutated_parent = ""
        for bit in parent:
            if np.random.random() < mutation_rate:
                mutated_parent += '1' if bit == '0' else '0'
            else:
                mutated_parent += bit
        mutated_parents.append(mutated_parent)
    return mutated_parents

def _get_fittest_parents(parents, fitness_function, top_k=20):
    real_parents = [binary_to_real(p) for p in parents]
    _fitness = np.array([fitness_function(p) for p in real_parents])
    PFitness = list(zip(parents, _fitness))
    PFitness.sort(key=lambda x: x[1], reverse=True)
    top_parents = [p[0] for p in PFitness[:top_k]]
    best_fitness = PFitness[0][1]
    return top_parents, round(best_fitness, 7)

def memetic_algorithm(initial_parents, fitness_function, mutation_rate, population, generations=1000000):
    History = []
    parents = initial_parents.copy()
    best_parent, best_fitness = _get_fittest_parents(parents, fitness_function, top_k=20)[0][0], _get_fittest_parents(parents, fitness_function, top_k=20)[1]

    x = np.linspace(start=-20, stop=20, num=200)
    plt.ion()

    for i in range(1, generations):
        top_parents, curr_fitness = _get_fittest_parents(parents, fitness_function, top_k=20)
        parents = crossover(top_parents, population)
        parents = mutate(parents, mutation_rate)

        parents = [local_search(parent, fitness_function) for parent in parents]

        if curr_fitness > best_fitness:
            best_fitness = curr_fitness
            best_parent = top_parents[0]

        History.append((i, best_fitness))

        #36.5981486
        if curr_fitness >= 121.6176:
           print("iteracija: ", i)
           break

        real_parents = [binary_to_real(p) for p in parents]

    print('generation {}| best fitness {}| best_parent {} | mutation_rate {}'.format(i, best_fitness, binary_to_real(best_parent), mutation_rate))
    return best_parent, best_fitness, History, i

#------------------------------------------------------------------------------------------------------------------
# generuoja duomenis duomenis ir saugo į failą (mutation rate | iterations | standard deviation)
# start_time = time.perf_counter()
#
# results = []
#
# for mutation_rate in np.arange(0.09, 0.92, 0.01):
#     scores = []
#     for i in range(100):
#         initial_parents = [real_to_binary(np.random.uniform(-10, 10)) for _ in range(100)]
#         best_parent, best_fitness, history, iteration = memetic_algorithm(initial_parents, fitness, mutation_rate, 100)
#         scores.append(iteration)
#
#     average_iterations = np.mean(scores)
#     variance = np.std(scores)
#     results.append((round(mutation_rate, 3), average_iterations, variance))
#
# with open("results_variance.txt", "w") as file:
#     for mutation_rate, average_iterations, variance in results:
#         file.write(f"{mutation_rate}, {average_iterations}, {variance}\n")
#
# end_time = time.perf_counter()
# execution_time = end_time - start_time
#
# print(f"Programos vykdymo laikas: {execution_time} sekundės")


#duomenu vizualizacija
visualization_with_variance("results_variance200.txt")
visualization_with_variance("results_variance100.txt")



#------------------------------------------------------------------------------------------------------------------
# generuoja duomenis ir saugo i faila (iterations | time) pasirenkamas optimaliausias mutation rate ir vykdoma 1000 kartu

# times = []
# iterations = []
#
# for _ in range(1000):
#     start_time = time.perf_counter()
#     initial_parents = [real_to_binary(np.random.uniform(-10, 10)) for _ in range(200)]
#     best_parent, best_fitness, history, iteration = memetic_algorithm(initial_parents, fitness, 0.9, 200)
#     end_time = time.perf_counter()
#     finish_time = end_time - start_time
#     times.append(finish_time)
#     iterations.append(iteration)
#
#
# with open("MAresults200TnI.txt", "w") as file:
#     for i in range(len(times)):
#         file.write(f"{iterations[i]}, {times[i]}\n")
#
# times1 = []
# iterations1 = []
#
# for _ in range(1000):
#     start_time = time.perf_counter()
#     initial_parents = [real_to_binary(np.random.uniform(-10, 10)) for _ in range(100)]
#     best_parent, best_fitness, history, iteration = memetic_algorithm(initial_parents, fitness, 0.9, 100)
#     end_time = time.perf_counter()
#     finish_time = end_time - start_time
#     times1.append(finish_time)
#     iterations1.append(iteration)
#
#
# with open("MAresults100TnI.txt", "w") as file:
#     for i in range(len(times1)):
#         file.write(f"{iterations1[i]}, {times1[i]}\n")





#boxplot