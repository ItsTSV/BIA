from Utils import Functions as f
from Utils import Solution as s
from Utils import Plotter as p
import BlindSearch as bs

# Test -> Plot all functions
'''
solution = s.Solution(3,-10, 10)
plotter = p.Plotter(solution)
plotter.plot_function(f.sphere, wait=True)
plotter.plot_function(f.ackley, wait=True)
plotter.plot_function(f.rastrigin, wait=True)
plotter.plot_function(f.rosenbrock, wait=True)
plotter.plot_function(f.griewank, wait=True)
plotter.plot_function(f.schwefel, wait=True)
plotter.plot_function(f.levy, wait=True)
plotter.plot_function(f.michalewicz, wait=True)
plotter.plot_function(f.zakharov, wait=True)
'''

# Blind search on selected functions
solution_ack = s.Solution(3, -32.768, 32.768)
plotter = p.Plotter(solution_ack)
search = bs.BlindSearch(solution_ack, f.ackley, plotter, 10000)
search.search(visualize=True)

solution_sph = s.Solution(3, -5.12, 5.12)
plotter = p.Plotter(solution_sph)
search = bs.BlindSearch(solution_sph, f.sphere, plotter, 10000)
search.search(visualize=True)

solution_lev = s.Solution(3, -10, 10)
plotter = p.Plotter(solution_lev)
search = bs.BlindSearch(solution_lev, f.levy, plotter, 10000)
search.search(visualize=True)
