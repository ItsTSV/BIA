import numpy as np


class DifferentialEvolution:
    def __init__(self, solution, objective_function, plotter, population_size, generations, crossover_rate=0.9,
                 mutation_factor=0.5):
        # Parameters
        self.solution = solution
        self.objective_function = objective_function
        self.plotter = plotter
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_factor = mutation_factor

        # Data
        self.best_solution = None
        self.best_fitness = float('inf')
        self.points = []

    def search(self, visualize=False):
        # Generate initial population
        population = np.random.uniform(self.solution.lower_bound, self.solution.upper_bound,
                                       (self.population_size, self.solution.dimension))

        for generation in range(self.generations):
            new_population = np.copy(population)

            for index, target in enumerate(population):
                # Select three random indices
                indices_a, indices_b, indices_c = np.random.choice(self.population_size, 3, replace=False)

                # Generate mutation vector
                mutation_vector = population[indices_a] + self.mutation_factor * (population[indices_b] - population[indices_c])

                # Generate crossover vector
                crossover_vector = np.copy(target)

                # Do crossover
                for i in range(self.solution.dimension):
                    if np.random.uniform() < self.crossover_rate:
                        crossover_vector[i] = mutation_vector[i]

                # Clip values
                crossover_vector = np.clip(crossover_vector, self.solution.lower_bound, self.solution.upper_bound)

                # Evaluation
                crossover_fitness = self.objective_function(crossover_vector)
                target_fitness = self.objective_function(target)

                if crossover_fitness < target_fitness:
                    new_population[index] = crossover_vector
                else:
                    new_population[index] = target

                if target_fitness < self.best_fitness:
                    self.best_solution = target
                    self.best_fitness = target_fitness
                    self.points.append(target)
                    print(f'Generation {generation}; found new best solution with fitness: {self.best_fitness}')

                    if visualize:
                        self.plotter.plot_function(self.objective_function, self.points)

            population = new_population

        print(f"Best solution found: {self.best_solution} with fitness: {self.best_fitness}\n")
        self.plotter.plot_function(self.objective_function, self.points, wait=True)
        return self.best_solution, self.best_fitness
