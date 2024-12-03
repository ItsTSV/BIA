import numpy as np
import matplotlib.pyplot as plt


class Firefly:

    def __init__(self, solution, objective_function, plotter=None, light_absorption=0.01, attractiveness=1.0, alpha=0.3,
                 population_size=20, max_generations=20):
        # Parameters
        self.solution = solution
        self.objective_function = objective_function
        self.plotter = plotter
        self.light_absorption = light_absorption
        self.attractiveness = attractiveness
        self.alpha = alpha
        self.population_size = population_size
        self.max_generations = max_generations

        # Data
        self.best_solution = None
        self.best_fitness = float("inf")
        self.points = []

    def search(self, visualize=False):
        # Initial population
        population = np.random.uniform(self.solution.lower_bound, self.solution.upper_bound,
                                       (self.population_size, self.solution.dimension))

        # Evaluate and pick the best solution
        fitnesses = np.array([self.objective_function(ind) for ind in population])
        self.best_solution = population[np.argmin(fitnesses)]
        self.best_fitness = np.min(fitnesses)

        # Visualize, because the algorithm converges fast
        if visualize:
            self.points = list(population) + [self.best_solution]
            self.plotter.plot_function(self.objective_function, self.points)

        # Main loop
        for generation in range(self.max_generations):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    # Only move if firefly j is brighter
                    if fitnesses[j] < fitnesses[i]:
                        distance = np.linalg.norm(population[j] - population[i])
                        attractiveness = self.attractiveness * np.exp(-self.light_absorption * distance ** 2)

                        # Move firefly i towards firefly j
                        epsilon = np.random.normal(0, 1, self.solution.dimension)
                        population[i] += attractiveness * (population[j] - population[i]) + self.alpha * epsilon

                        # Clip so it does not wander to second monitor ;)
                        population[i] = np.clip(population[i], self.solution.lower_bound, self.solution.upper_bound)

                        # Evaluate
                        fitnesses[i] = self.objective_function(population[i])

                        # Update best solution
                        if fitnesses[i] < self.best_fitness:
                            self.best_solution = population[i]
                            self.best_fitness = fitnesses[i]
                            print(f"Generation {generation} -- found new best solution {self.best_solution}"
                                  f" with fitness {self.best_fitness}")

            # Visualize current population + best solution
            if visualize:
                self.points = list(population) + [self.best_solution]
                self.plotter.plot_function(self.objective_function, self.points)

        # Print best solution
        print(f"Best solution found: {self.best_solution} with fitness {self.best_fitness}\n")
        if visualize:
            self.plotter.plot_function(self.objective_function, self.points, wait=True)

        return self.best_solution, self.best_fitness
