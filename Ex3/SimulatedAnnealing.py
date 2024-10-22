import numpy as np

class SimulatedAnnealing:
    def __init__(self, solution, objective_function, plotter, temperature_max=1000,
                 temperature_min=0.5, cooling_rate=0.98, exploration_range=1):
        # Generic parameters
        self.solution = solution
        self.objective_function = objective_function
        self.plotter = plotter
        self.best_solution = None
        self.best_fitness = float('inf')
        # Simulated Annealing parameters
        self.temperature = temperature_max
        self.minimal_temperature = temperature_min
        self.cooling_rate = cooling_rate
        self.exploration_range = exploration_range
        # Visualization stuff
        self.points = []

    def search(self, visualize=False):
        # Initialize solution
        current_solution = np.random.uniform(self.solution.lower_bound, self.solution.upper_bound,
                                               self.solution.dimension)
        current_fitness = self.objective_function(current_solution)
        self.best_solution = current_solution
        self.solution.fitness = current_fitness
        self.points.append(self.best_solution)

        # Loop until temperature is above minimal
        while self.temperature > self.minimal_temperature:
            # Generate and evaluate new candidate solution
            candidate_solution = current_solution + np.random.uniform(-self.exploration_range, self.exploration_range,
                                                                      self.solution.dimension)
            candidate_solution = np.clip(candidate_solution, self.solution.lower_bound, self.solution.upper_bound)
            candidate_fitness = self.objective_function(candidate_solution)

            # Perform the cursed formula from the slides
            fitness_difference = candidate_fitness - current_fitness
            if fitness_difference < 0 or np.random.rand() < np.exp(-fitness_difference / self.temperature):
                current_solution = candidate_solution
                current_fitness = candidate_fitness
                # If it's the best solution so far, save it
                if candidate_fitness < self.solution.fitness:
                    self.solution.fitness = candidate_fitness
                    self.best_solution = candidate_solution
                    self.points.append(candidate_solution)
                else:
                    if len(self.points) > 3:
                        self.points[-2] = candidate_solution
                    else:
                        self.points.append(candidate_solution)

            # Cool down the temperature
            self.temperature *= self.cooling_rate

            # Visualize
            if visualize:
                self.plotter.plot_function(self.objective_function, self.points, pause_for=0.05)

        print(f"Best solution found: {self.best_solution} with fitness: {self.solution.fitness}\n")
        self.plotter.plot_function(self.objective_function, self.points, wait=True)
        return self.best_solution, self.solution.fitness
