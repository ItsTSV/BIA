import numpy as np
import matplotlib.pyplot as plt


class AntColony:
    def __init__(self, individuals, generations, city_count, evaporation_rate=0.25,
                 pheromone_influence=1.0, visibility_influence=2.0, deposit_factor=100, lower_bound=0,
                 upper_bound=200):
        # Parameters -- tsp
        self.individuals = individuals
        self.generations = generations
        self.city_count = city_count
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.names = [chr(c) for c in range(ord('A'), ord('A') + city_count)]

        # Parameters -- aco
        self.evaporation_rate = evaporation_rate
        self.pheromone_influence = pheromone_influence
        self.visibility_influence = visibility_influence
        self.deposit_factor = deposit_factor

        # Data
        self.cities = np.random.randint(self.lower_bound, self.upper_bound, (self.city_count, 2))
        self.population = None
        self.best_value = float('inf')
        self.best_solution = None

        # Matrices
        self.pheromones = np.ones((city_count, city_count))
        self.visibility = np.zeros((city_count, city_count))
        for i in range(city_count):
            for j in range(city_count):
                if i != j:
                    self.visibility[i, j] = 1 / self.distance(i, j)

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

    def run_tsp(self):
        for generation in range(self.generations):
            # Initialize paths for all ants -- each ant starts at a random city
            ant_paths = [[np.random.randint(self.city_count)] for _ in range(self.individuals)]

            # Construct paths for all ants
            for ant in range(self.individuals):
                visited = set(ant_paths[ant])
                while len(visited) < self.city_count:
                    current_city = ant_paths[ant][-1]
                    probabilities = self._calculate_probabilities(current_city, visited)
                    next_city = np.random.choice(range(self.city_count), p=probabilities)
                    ant_paths[ant].append(next_city)
                    visited.add(next_city)

            # Evaluate paths, update the best solution and visualize
            for path in ant_paths:
                path_length = self._calculate_path_length(path)
                if path_length < self.best_value:
                    self.best_value = path_length
                    self.best_solution = path
                    self.visualize(wait=False)
                    print(f"Generation {generation} -- found new best solution with length {self.best_value}")

            # Update pheromones
            self.pheromones *= (1 - self.evaporation_rate)
            for path in ant_paths:
                path_length = self._calculate_path_length(path)
                pheromone_contribution = self.deposit_factor / path_length
                for i in range(len(path) - 1):
                    self.pheromones[path[i], path[i + 1]] += pheromone_contribution
                    self.pheromones[path[i + 1], path[i]] += pheromone_contribution
                # Add return to starting city
                self.pheromones[path[-1], path[0]] += pheromone_contribution
                self.pheromones[path[0], path[-1]] += pheromone_contribution

        # Final visualization of the best solution
        print(f"Final best solution with length {self.best_value}")
        self.visualize(wait=True)

    # Calculate probabilities for all cities that have not been visited yet
    def _calculate_probabilities(self, current_city, visited):
        probabilities = np.zeros(self.city_count)
        for city in range(self.city_count):
            if city not in visited:
                pheromone = self.pheromones[current_city, city] ** self.pheromone_influence
                visibility = self.visibility[current_city, city] ** self.visibility_influence
                probabilities[city] = pheromone * visibility
        probabilities /= probabilities.sum()  # Normalize probabilities
        return probabilities

    # Calculate length of all distances between cities + return to starting city
    def _calculate_path_length(self, path):
        path_length = sum(self.distance(path[i], path[i + 1]) for i in range(len(path) - 1))
        path_length += self.distance(path[-1], path[0])
        return path_length
