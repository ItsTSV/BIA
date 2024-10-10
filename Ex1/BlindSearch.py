import numpy as np


class BlindSearch:
    def __init__(self, solution, objective_function, plotter, max_iterations=100):
        self.solution = solution
        self.objective_function = objective_function
        self.plotter = plotter
        self.max_iterations = max_iterations
        self.best_solution = None
        self.points = []

    def search(self, visualize=False):
        # Start search for a given number of iterations
        for iteration in range(self.max_iterations):
            # Randomly sample a point within the bounds
            candidate_solution = np.random.uniform(self.solution.lower_bound, self.solution.upper_bound, self.solution.dimension)

            # Evaluate the fitness of the candidate solution
            fitness = self.objective_function(candidate_solution)

            # If the fitness is better than the best one, update the best solution
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


