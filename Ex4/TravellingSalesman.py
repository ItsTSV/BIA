import numpy as np
import matplotlib.pyplot as plt


class TravellingSalesman:
    def __init__(self, individuals, generations, city_count, lower_bound, upper_bound, mutation_rate=0.5, cut_offset=4):
        # Parameters
        self.individuals = individuals
        self.generations = generations
        self.city_count = city_count
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mutation_rate = mutation_rate
        self.cut_offset = cut_offset
        self.names = [chr(c) for c in range(ord('A'), ord('A') + city_count)]

        # Data
        self.cities = None
        self.population = None
        self.best_value = float('inf')
        self.best_solution = None

    def distance(self, index1, index2):
        return np.linalg.norm(self.cities[index1] - self.cities[index2])

    def visualize(self, wait=False):
        # Create plot and scatter the cities
        fig = plt.figure(0)
        plt.clf()
        ax = plt.gca()
        ax.scatter(self.cities[:, 0], self.cities[:, 1])

        # Add labels
        for i in range(len(self.cities)):
            ax.annotate(self.names[i], (self.cities[i, 0], self.cities[i, 1]))

        # Add paths
        for i in range(len(self.best_solution) - 1):
            ax.plot([self.cities[self.best_solution[i], 0], self.cities[self.best_solution[i + 1], 0]],
                    [self.cities[self.best_solution[i], 1], self.cities[self.best_solution[i + 1], 1]], color="black")

        # Add return path
        ax.plot([self.cities[self.best_solution[-1], 0], self.cities[self.best_solution[0], 0]],
                [self.cities[self.best_solution[-1], 1], self.cities[self.best_solution[0], 1]], color="black")

        if not wait:
            plt.pause(1)
        else:
            plt.show()

    def evaluate_solution(self, solution):
        path_len = sum([self.distance(solution[i], solution[i + 1]) for i in range(len(solution) - 1)])
        return_len = self.distance(solution[-1], solution[0])
        return path_len + return_len

    def genetic_tsp(self):
        # Initialize cities and population
        self.cities = np.random.randint(self.lower_bound, self.upper_bound, (self.city_count, 2))
        self.population = [np.random.permutation(self.city_count) for _ in range(self.individuals)]

        # Evaluate
        for individual in self.population:
            if self.evaluate_solution(individual) < self.best_value:
                self.best_value = self.evaluate_solution(individual)
                self.best_solution = individual

        # Plot
        self.visualize()
        print(f"Initial best solution: {self.best_solution} with length: {self.best_value}")

        # Loop
        for generation in range(self.generations):
            # Save population
            new_population = np.copy(self.population)

            # Do genetic stuff
            for individual in range(self.individuals):
                # Randomly choose two parents
                parent_a, parent_b = np.random.choice(self.individuals, 2, replace=False)

                # Create child and randomly choose cut point
                child = np.zeros(self.city_count, dtype=int)
                cut_point = np.random.randint(self.cut_offset, self.city_count - self.cut_offset)

                # Crossover
                child[:cut_point] = self.population[parent_a][:cut_point]
                child[cut_point:] = [city for city in reversed(self.population[parent_b])
                                     if city not in child[:cut_point]]

                # Mutate?
                if np.random.rand() < self.mutation_rate:
                    mutation = np.random.choice(self.city_count, 2, replace=False)
                    child[mutation[0]], child[mutation[1]] = child[mutation[1]], child[mutation[0]]

                # Evaluate, replace individual or best if this one is better
                value = self.evaluate_solution(child)

                if value < self.evaluate_solution(self.population[individual]):
                    new_population[individual] = child

                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = child
                    self.visualize()
                    print(
                        f"Generation {generation}: found new best solution {self.best_solution} with length {self.best_value}")

            # Update population
            self.population = new_population

        print(f"Final best solution: {self.best_solution} with length: {self.best_value}")
        self.visualize(wait=True)
