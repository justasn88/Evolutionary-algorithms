# Evolutionary-algorithms
This repository contains Python implementations of Genetic Algorithm (GA) and Memetic Algorithm (MA) for optimizing a mathematical function.


`genetic_algorithm()` - Runs the standard genetic algorithm:
```python
initial_parents = [real_to_binary(np.random.uniform(-10, 10)) for _ in range(100)]
best_parent, best_fitness, history, iteration = genetic_algorithm(initial_parents, fitness, 0.1, 100)```

memetic_algorithm() - Runs the memetic algorithm algorithm:
```python
initial_parents = [real_to_binary(np.random.uniform(-10, 10)) for _ in range(100)]
best_parent, best_fitness, history, iteration = memetic_algorithm(initial_parents, fitness, 0.1, 100)```
