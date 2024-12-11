import numpy as np


class TeachingLearningOptimization:
    def __init__(self, solution, objective_function, plotter=None, population_size=20, max_generations=20):
        # Parameters
        self.solution = solution
        self.objective_function = objective_function
        self.plotter = plotter
        self.population_size = population_size
        self.max_generations = max_generations

        # Data
        self.best_solution = None
        self.best_fitness = float("inf")
        self.points = []

    def search(self, visualize=False):
        # Initialize population
        population = np.random.uniform(self.solution.lower_bound, self.solution.upper_bound,
                                        (self.population_size, self.solution.dimension))

        # Evaluate initial population, choose best individual
        fitnesses = np.array([self.objective_function(individual) for individual in population])
        best_index = np.argmin(fitnesses)

        # Update best solution
        if fitnesses[best_index] < self.best_fitness:
            self.best_solution = population[best_index]
            self.best_fitness = fitnesses[best_index]

        for generation in range(self.max_generations):
            # Teacher phase -- Calculate mean, select the teacher (the best individual)
            mean_population = np.mean(population, axis=0)
            teacher = population[best_index]

            # Generate variables -- Teacher factor t_f [1, 2], random uniform r [0, 1]
            teacher_factor = np.random.randint(1, 3)
            r = np.random.uniform(0, 1, self.solution.dimension)

            # Teacher phase -- Update population by moving individuals from mean towards teacher
            for i in range(self.population_size):
                # No need to update the teacher
                if i == best_index:
                    continue

                # Calculate difference based on the equation from the slides
                difference = r * (teacher - teacher_factor * mean_population)

                # Move the individual towards the teacher, clip the solution so the individuals stay inside the bounds
                new_solution = population[i] + difference
                new_solution = np.clip(new_solution, self.solution.lower_bound, self.solution.upper_bound)

                # If the fitness is better after the move, replace the old individual in the population with the new one
                new_fitness = self.objective_function(new_solution)
                if new_fitness < fitnesses[i]:
                    population[i] = new_solution
                    fitnesses[i] = new_fitness

            # Learner phase
            for i in range(self.population_size):
                # For each learner, randomly select a partner (can't be the same individual)
                partner_index = np.random.choice([j for j in range(self.population_size) if j != i])

                # Do the ugly if-else from the slides -- if the fitness of the partner is better, move towards it
                if fitnesses[i] < fitnesses[partner_index]:
                    new_solution = population[i] + r * (population[i] - population[partner_index])
                else:
                    new_solution = population[i] + r * (population[partner_index] - population[i])

                # Clip the solution (again) and evaluate the new fitness
                new_solution = np.clip(new_solution, self.solution.lower_bound, self.solution.upper_bound)
                new_fitness = self.objective_function(new_solution)

                # If the new fitness is better, replace the old individual in the population with the new one
                if new_fitness < fitnesses[i]:
                    population[i] = new_solution
                    fitnesses[i] = new_fitness

            # Calculate fitnesses and select the best individual for the next generation
            fitnesses = np.array([self.objective_function(individual) for individual in population])
            best_index = np.argmin(fitnesses)

            # If the best individual in the population is better than current best solution, update the best solution
            if fitnesses[best_index] < self.best_fitness:
                print(f"Generation {generation}: Found new best solution {population[best_index]} with fitness {fitnesses[best_index]}")
                self.best_solution = population[best_index]
                self.best_fitness = fitnesses[best_index]

            # Visualization
            if visualize:
                self.points = [individual for individual in population if
                               not np.array_equal(individual, self.best_solution)]
                self.points.append(self.best_solution)
                self.plotter.plot_function(self.objective_function, self.points)

        print(f"Best solution found: {self.best_solution} with fitness {self.best_fitness}\n")
        if self.plotter:
            self.plotter.plot_function(self.objective_function, self.points, wait=True)

        return self.best_solution, self.best_fitness
