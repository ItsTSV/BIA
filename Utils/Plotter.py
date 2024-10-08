import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, solution):
        self.solution = solution

    def plot_function(self, func, point=None, resolution=100):
        # New figure
        figure = plt.figure()

        # 2D grid based on bounds and resolution
        x = np.linspace(self.solution.lower_bound, self.solution.upper_bound, resolution)
        y = np.linspace(self.solution.lower_bound, self.solution.upper_bound, resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluate the function over the grid
        Z = np.array([func(np.array([x_val, y_val])) for x_val, y_val in zip(np.ravel(X), np.ravel(Y))])
        Z = Z.reshape(X.shape)

        # Plot stuff go
        ax = figure.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.7)

        # If point is provided (visualizing algorithm progress), plot it
        if point is not None:
            Z_point = func(point)
            ax.plot([point[0], point[0]], [point[1], point[1]], [Z_point, 0], color='red', linewidth=2)

            # Add line to the bottom
            ax.scatter(point[0], point[1], Z_point, color='blue', s=100, edgecolor='black', label='Current Point')

        # The axis is inverted in the charts from presentation, so... reverse?
        ax.invert_xaxis()

        # Cosmetic stuff and show the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{func.__name__} function'.capitalize())
        plt.show()