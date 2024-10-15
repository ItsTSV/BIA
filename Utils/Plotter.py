import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, solution):
        self.solution = solution
        self.figure = plt.figure(0)

    def plot_function(self, func, points=None, wait=False, resolution=100):
        # If plot does not exist, create it
        if not plt.fignum_exists(1):
            self.figure = plt.figure(0)

        # Clear plot
        self.figure.clf()

        # 2D grid based on bounds and resolution
        x = np.linspace(self.solution.lower_bound, self.solution.upper_bound, resolution)
        y = np.linspace(self.solution.lower_bound, self.solution.upper_bound, resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluate the function over the grid
        Z = np.array([func(np.array([x_val, y_val])) for x_val, y_val in zip(np.ravel(X), np.ravel(Y))])
        Z = Z.reshape(X.shape)

        # Plot stuff go
        ax = self.figure.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.7)

        # If new point is provided (visualizing algorithm progress), plot it
        if points is not None:
            for point in points:
                Z_point = func(point)
                color = 'green' if Z_point == self.solution.fitness else 'red'

                # Add point to the chart
                ax.scatter(point[0], point[1], Z_point, color=color, s=100, edgecolor='black')

                # If this is the best solution, add line to it
                if color == "green":
                    ax.plot([point[0], point[0]], [point[1], point[1]], [Z_point, 0], color=color, linewidth=2)

        # The axis is inverted in the charts from presentation, so... reverse?
        ax.invert_xaxis()

        # Cosmetic stuff and show the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{func.__name__} function'.capitalize())
        if wait:
            plt.show()
        else:
            plt.pause(0.4)
