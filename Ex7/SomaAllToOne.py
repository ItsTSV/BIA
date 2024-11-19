import numpy as np


class SomaAllToOne:
    def __init__(self, solution, objective_function, plotter=None, path_length=2.5, step=0.4, prt=0.5,
                 population_size=20, migrations=40, min_div=0):
        # Parameters
        self.solution = solution
        self.objective_function = objective_function
        self.plotter = plotter
        self.path_length = path_length
        self.step = step
        self.prt = prt
        self.population_size = population_size
        self.migrations = migrations
        self.min_div = min_div

        # Data
        self.best_solution = None
        self.best_fitness = float("inf")
        self.points = []

    def search(self, visualize=False):
        # Generate initial population
        population = np.random.uniform(self.solution.lower_bound, self.solution.upper_bound,
                                       (self.population_size, self.solution.dimension))

        # Evaluate initial population
        fitnesses = np.array([self.objective_function(individual) for individual in population])

        # Find the best individual
        self.best_solution = population[np.argmin(fitnesses)]
        self.best_fitness = min(fitnesses)

        # Main search loop
        for migration in range(self.migrations):
            for i in range(self.population_size):
                # Skip leader
                if np.array_equal(population[i], self.best_solution):
                    continue

                # Generate random perturbation
                perturbation = np.random.uniform(0, 1, self.solution.dimension)
                perturbation = [1 if x < self.prt else 0 for x in perturbation]

                # Initialize temporary solution
                temp_solution = np.copy(population[i])

                # Jump towards leader
                for step_factor in np.arange(self.step, self.path_length + self.step, self.step):
                    # Calculate new position
                    new_position = population[i] + step_factor * (self.best_solution - population[i]) * perturbation

                    # Clip
                    new_position = np.clip(new_position, self.solution.lower_bound, self.solution.upper_bound)

                    # Evaluate the new position
                    new_fitness = self.objective_function(new_position)

                    # Update if the new position is better
                    if new_fitness < fitnesses[i]:
                        temp_solution = new_position
                        fitnesses[i] = new_fitness

                # Update the individuals position to best one found
                population[i] = temp_solution

            # Find the new best solution in the population
            current_best_index = np.argmin(fitnesses)
            current_best_fitness = fitnesses[current_best_index]

            if current_best_fitness < self.best_fitness:
                print(f"Migration {migration} - Best individual {self.best_solution} with fitness {self.best_fitness}")
                self.best_solution = population[current_best_index]
                self.best_fitness = current_best_fitness

            # Visualization
            if visualize:
                self.points = [individual for individual in population if
                               not np.array_equal(individual, self.best_solution)]
                self.points.append(self.best_solution)
                self.plotter.plot_function(self.objective_function, self.points)

        print(f"Best solution: {self.best_solution} with fitness {self.best_fitness}")
        if self.plotter:
            self.plotter.plot_function(self.objective_function, self.points, wait=True)

        return self.best_solution, self.best_fitness
