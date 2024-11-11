import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, solution, objective_function, plotter=None, population_size=15, migration_cycles=50,
                 learning_constant1=2, learning_constant2=2, inertia=0.5, velocity_min=-1, velocity_max=1):
        # Parameters
        self.solution = solution
        self.objective_function = objective_function
        self.plotter = plotter
        self.population_size = population_size
        self.migration_cycles = migration_cycles
        self.learning_constant1 = learning_constant1
        self.learning_constant2 = learning_constant2
        self.inertia = inertia
        self.velocity_min = velocity_min
        self.velocity_max = velocity_max

        # Data
        self.best_solution = None
        self.best_fitness = float("inf")
        self.points = []

    def search(self, visualize=False):
        # Generate initial population and velocities
        swarm = np.random.uniform(self.solution.lower_bound, self.solution.upper_bound,
                                  (self.population_size, self.solution.dimension))
        velocities = np.random.uniform(self.velocity_min, self.velocity_max,
                                       (self.population_size, self.solution.dimension))

        # Initialize personal best positions and fitnesses
        personal_best_positions = np.copy(swarm)
        personal_best_fitnesses = np.array([self.objective_function(individual) for individual in swarm])

        # Initialize global best position
        self.best_solution = swarm[np.argmin(personal_best_fitnesses)]
        self.best_fitness = min(personal_best_fitnesses)

        # Main search loop
        for cycle in range(self.migration_cycles):
            for i in range(self.population_size):
                # Random factors for cognitive and social components
                r1 = np.random.rand(self.solution.dimension)
                r2 = np.random.rand(self.solution.dimension)

                # Velocity update using personal best and global best
                cognitive_velocity = self.learning_constant1 * r1 * (personal_best_positions[i] - swarm[i])
                social_velocity = self.learning_constant2 * r2 * (self.best_solution - swarm[i])
                velocities[i] = self.inertia * velocities[i] + cognitive_velocity + social_velocity

                # Clip velocities
                velocities[i] = np.clip(velocities[i], self.velocity_min, self.velocity_max)

                # Update position
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.solution.lower_bound, self.solution.upper_bound)

                # Evaluate fitness and update personal and global bests
                fitness = self.objective_function(swarm[i])
                if fitness < personal_best_fitnesses[i]:
                    personal_best_fitnesses[i] = fitness
                    personal_best_positions[i] = swarm[i]

                    # Update global best if the current position is better
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_solution = swarm[i]
                        print(f"Generation {cycle} - Best individual {self.best_solution} with fitness {self.best_fitness}")

            # Visualize if plotter is provided
            if visualize and self.plotter:
                self.points = list(swarm)
                self.points.append(self.best_solution)
                self.plotter.plot_function(self.objective_function, self.points)

        print(f"Best solution: {self.best_solution} with fitness {self.best_fitness}")
        if self.plotter:
            self.plotter.plot_function(self.objective_function, self.points, wait=True)
        return self.best_solution, self.best_fitness
