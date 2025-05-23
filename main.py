from os import times_result

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
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


    plt.plot(x_values, y_values, label='Average number of iterations')
    plt.fill_between(x_values, np.maximum(y_values - variances, 0), y_values + variances, alpha=0.2, color='b', label='Standard deviation')
    plt.xlabel('Mutation rate')
    plt.ylabel('Average number of Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

def generuoti_boxplot_is_failo(failo_pavadinimas):
    try:
        with open(failo_pavadinimas, 'r') as failas:
            duomenys = []
            for eilute in failas:
                eilute = eilute.strip()
                if eilute:  # Praleidžiamos tuščios eilutės
                    try:
                        iteracijos, laikas = map(float, eilute.split(','))
                        duomenys.append((iteracijos, laikas))
                    except ValueError:
                        print(f"Įspėjimas: neteisingas duomenų formatas eilutėje: {eilute}")

        if not duomenys:
            print("Klaida: faile nerasta tinkamų duomenų.")
            return

        iteracijos = [d[0] for d in duomenys]
        laikai = [d[1] for d in duomenys]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Iteracijų palyginimas
        axes[0].boxplot(iteracijos, patch_artist=True)
        axes[0].set_title("Iteracijų paskirstymas")
        axes[0].set_ylabel("Iteracijų skaičius")

        # Vykdymo laiko palyginimas
        axes[1].boxplot(laikai, patch_artist=True)
        axes[1].set_title("Vykdymo laiko paskirstymas")
        axes[1].set_ylabel("Laikas (s)")

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Klaida: failas '{failo_pavadinimas}' nerastas.")
    except Exception as e:
        print(f"Įvyko klaida: {e}")

def generuoti_boxplot(failo_pavadinimas1, failo_pavadinimas2, failo_pavadinimas3, failo_pavadinimas4):

    def skaityti_duomenis_is_failo(failo_pavadinimas):
        try:
            with open(failo_pavadinimas, 'r') as failas:
                duomenys = []
                for eilute in failas:
                    eilute = eilute.strip()
                    if eilute:
                        try:
                            iteracijos, laikas = map(float, eilute.split(','))
                            duomenys.append((iteracijos, laikas))
                        except ValueError:
                            print(f"Įspėjimas: neteisingas duomenų formatas eilutėje: {eilute} faile {failo_pavadinimas}")
                return duomenys
        except FileNotFoundError:
            print(f"Klaida: failas '{failo_pavadinimas}' nerastas.")
            return None
        except Exception as e:
            print(f"Įvyko klaida skaitant failą '{failo_pavadinimas}': {e}")
            return None

    duomenys1 = skaityti_duomenis_is_failo(failo_pavadinimas1)
    duomenys2 = skaityti_duomenis_is_failo(failo_pavadinimas2)
    duomenys3 = skaityti_duomenis_is_failo(failo_pavadinimas3)
    duomenys4 = skaityti_duomenis_is_failo(failo_pavadinimas4)

    if duomenys1 is None or duomenys2 is None:
        return

    iteracijos1 = [d[0] for d in duomenys1]
    laikai1 = [d[1] for d in duomenys1]
    iteracijos2 = [d[0] for d in duomenys2]
    laikai2 = [d[1] for d in duomenys2]
    iteracijos3 = [d[0] for d in duomenys3]
    laikai3 = [d[1] for d in duomenys3]
    iteracijos4 = [d[0] for d in duomenys4]
    laikai4 = [d[1] for d in duomenys4]

    print()

    u_statistic, p_value = stats.mannwhitneyu(iteracijos4, iteracijos3, alternative='less')

    print(f"Mano-Vitnio U statistika: {u_statistic}")
    print(f"P reikšmė (vienpusis testas): {p_value}")

    alpha = 0.05  #reikšmingumo lygis

    if p_value < alpha:
        print(
            "Atmestame nulinę hipotezę. Yra statistiškai reikšmingas įrodymas, kad 200 populiacijos mediana yra mažesnė nei 100 populiacijos mediana.")
    else:
        print(
            "Nepavyko atmesti nulinės hipotezės. Nėra statistiškai reikšmingo įrodymo, kad 200 populiacijos mediana yra mažesnė nei 100 populiacijos mediana.")

    u_statistic, p_value = stats.mannwhitneyu(laikai4, laikai3, alternative='two-sided')

    print(f"Mano-Vitnio U statistika: {u_statistic}")
    print(f"P reikšmė (dvipusis testas): {p_value}")

    if p_value < alpha:
        print(
            "Atmestame nulinę hipotezę. Yra statistiškai reikšmingas įrodymas, kad medianos yra skirtingos.")
    else:
        print(
            "Nepavyko atmesti nulinės hipotezės. Nėra statistiškai reikšmingo įrodymo, kad medianos yra statistiškai reikšmingai skirtingos.")
    # surasti statistini testa, naudojant scipy, kuris patikrina ar tikrai 200 pop mediana yra mazesne negu 100 pop mediana (ar 100 pop statistiskai reiksmingai didesne nei 200 pop)
    # su laiku patikrinti ar mediana lygi
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Iteracijų palyginimas
    axes[0].boxplot([iteracijos1, iteracijos2, iteracijos3, iteracijos4], tick_labels=["GA 100 Populiacija", "GA 200 Populiacija", "MA 100 Populiacija", "MA 200 Populiacija"], patch_artist=True)
    axes[0].set_title("Iteracijų palyginimas")
    axes[0].set_ylabel("Iteracijų skaičius")

    # Vykdymo laiko palyginimas
    axes[1].boxplot([laikai1, laikai2, laikai3, laikai4], tick_labels=["GA 100 Populiacija", "GA 200 Populiacija","MA 100 Populiacija", "MA 200 Populiacija"], patch_artist=True)
    axes[1].set_title("Vykdymo laiko palyginimas")
    axes[1].set_ylabel("Laikas (s)")

    plt.tight_layout()
    plt.show()

def real_to_binary(x, min_val=-11, max_val=11, bits=16):
    scaled = (x - min_val) / (max_val - min_val)
    int_val = int(scaled * (2**bits - 1))
    return bin(int_val)[2:].zfill(bits)

def binary_to_real(x, min_val=-11, max_val=11, bits=16):
    int_val = int(x, 2)
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

def genetic_algorithm(initial_parents, fitness_function, mutation_rate, population, generations=1000000):
    History = []
    parents = initial_parents.copy()
    best_parent, best_fitness = _get_fittest_parents(parents, fitness_function, top_k=20)[0][0], _get_fittest_parents(parents, fitness_function, top_k=20)[1]

    x = np.linspace(start=-20, stop=20, num=200)
    plt.ion()

    for i in range(1, generations):
        top_parents, curr_fitness = _get_fittest_parents(parents, fitness_function, top_k=20)
        parents = crossover(top_parents, population)
        parents = mutate(parents, mutation_rate)

        if curr_fitness > best_fitness:
            best_fitness = curr_fitness
            best_parent = top_parents[0]

        if i % 10 == 0:
            print('generation {}| best_fitness {}| current_fitness {}| current_parent {}'.format(i, best_fitness, curr_fitness, binary_to_real(best_parent)))
        History.append((i, best_fitness))

        if curr_fitness >= 121.6176:
           # print("iteracija: ", i)
           break

    #     real_parents = [binary_to_real(p) for p in parents]
    #     plt.clf()
    #     plt.plot(x, [fitness_function(val) for val in x])
    #     plt.scatter(real_parents, [fitness_function(val) for val in real_parents])
    #     plt.scatter(binary_to_real(best_parent), fitness_function(binary_to_real(best_parent)), marker='.', c='b', s=200)
    #     plt.draw()
    #     plt.pause(0.1)
    #
    # plt.ioff()
    # plt.show()
    print('generation {}| best fitness {}| best_parent {} | mutation_rate {}'.format(i, best_fitness, binary_to_real(best_parent), mutation_rate))
    return best_parent, best_fitness, History, i

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


#------------------------------------------------------------------------------------------------
#atliekamas genetinis algoritmas tol kol suras optimalu sprendini
# initial_parents = [real_to_binary(np.random.uniform(-10, 10)) for _ in range(100)]
# best_parent, best_fitness, history, iteration = genetic_algorithm(initial_parents, fitness, 0.1, 200)



#------------------------------------------------------------------------------------------------
# generuoja duomenis duomenis ir saugo į failą (mutation rate | iterations | standard deviation)
# start_time = time.perf_counter()
#
# results = []
#
# for mutation_rate in np.arange(0.1, 0.91, 0.01):
#     scores = []
#     for i in range(100):
#         initial_parents = [real_to_binary(np.random.uniform(-10, 10)) for _ in range(700)]
#         best_parent, best_fitness, history, iteration = genetic_algorithm(initial_parents, fitness, mutation_rate)
#         scores.append(iteration)
#
#     average_iterations = np.mean(scores)
#     variance = np.std(scores)
#     results.append((round(mutation_rate, 3), average_iterations, variance))
#
# with open("results_variance600.txt", "w") as file:
#     for mutation_rate, average_iterations, variance in results:
#         file.write(f"{mutation_rate}, {average_iterations}, {variance}\n")
#
# end_time = time.perf_counter()
# execution_time = end_time - start_time
#
# print(f"Programos vykdymo laikas: {execution_time} sekundės")
# visualization_with_variance("results_variance800.txt")



#------------------------------------------------------------------------------------------------------------------
# generuoja duomenis ir saugo i faila (iterations | time) pasirenkamas optimaliausias mutation rate ir vykdoma 1000 kartu
# times = []
# iterations = []
#
# for _ in range(100):
#     start_time = time.perf_counter()
#     initial_parents = [real_to_binary(np.random.uniform(-10, 10)) for _ in range(200)]
#     best_parent, best_fitness, history, iteration = genetic_algorithm(initial_parents, fitness, 0.17, 200)
#     end_time = time.perf_counter()
#     finish_time = end_time - start_time
#     times.append(finish_time)
#     iterations.append(iteration)
#
#
# with open("results20TnI.txt", "w") as file:
#     for i in range(len(times)):
#         file.write(f"{iterations[i]}, {times[i]}\n")
#
# times1 = []
# iterations1 = []
#
# for _ in range(100):
#     start_time = time.perf_counter()
#     initial_parents = [real_to_binary(np.random.uniform(-10, 10)) for _ in range(100)]
#     best_parent, best_fitness, history, iteration = genetic_algorithm(initial_parents, fitness, 0.11, 100)
#     end_time = time.perf_counter()
#     finish_time = end_time - start_time
#     times1.append(finish_time)
#     iterations1.append(iteration)
#
#
# with open("results10TnI.txt", "w") as file:
#     for i in range(len(times1)):
#         file.write(f"{iterations1[i]}, {times1[i]}\n")




#generuoti_boxplot("results100TnI.txt", "results200TnI.txt", "MAresults100TnI.txt", "MAresults200TnI.txt")


#visualization_with_variance("results_variance100.txt")
#visualization_with_variance("results_variance200.txt")

