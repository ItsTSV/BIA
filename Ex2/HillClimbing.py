import numpy as np


class HillClimbing:
    def __init__(self, solution, objective_function, plotter, max_iterations=100, exploration_range=0.1,
                 neighbourhood_size=4):
        self.solution = solution
        self.objective_function = objective_function
        self.plotter = plotter
        self.max_iterations = max_iterations
        self.best_solution = None
        self.exploration_range = exploration_range
        self.neighbourhood_size = neighbourhood_size
        self.points = []

    def search(self, visualize=False):
        # Generate random candidate solution within bounds
        self.best_solution = np.random.uniform(self.solution.lower_bound, self.solution.upper_bound,
                                               self.solution.dimension)
        self.solution.fitness = self.objective_function(self.best_solution)
        self.points.append(self.best_solution)

        # Start search for a given number of iterations
        for iteration in range(0, self.max_iterations, self.neighbourhood_size):
            # Use the current best solution, explore solutions in its range
            candidate_solutions = [self.best_solution + np.random.uniform(-self.exploration_range, self.exploration_range,
                                                       self.solution.dimension) for _ in range(self.neighbourhood_size)]

            # Evaluate the fitness of the candidate solution
            fitnesses = [self.objective_function(candidate_solution) for candidate_solution in candidate_solutions]

            # If the fitness is better than the best one, update the best solution
            for candidate_solution, fitness in zip(candidate_solutions, fitnesses):
                if fitness < self.solution.fitness:
                    self.solution.fitness = fitness
                    self.best_solution = candidate_solution
                    self.points.append(candidate_solution)
                    print(f"Iteration {iteration}: Model fitness = {self.solution.fitness}")

                    if visualize:
                        self.plotter.plot_function(self.objective_function, self.points)

        print(f"Best solution found: {self.best_solution} with fitness: {self.solution.fitness}\n")
        self.plotter.plot_function(self.objective_function, self.points, wait=True)
        return self.best_solution, self.solution.fitness
