from Utils import Functions as f
from Utils import Solution as s
from Utils import Plotter as p
import SimulatedAnnealing as sa
import math

# Test on functions
solution_sph = s.Solution(2, -5.12, 5.12)
plotter = p.Plotter(solution_sph)
search = sa.SimulatedAnnealing(solution_sph, f.sphere, plotter, 100, 2, 0.95, 1)
search.search(visualize=True)

solution_ack = s.Solution(2, -32.768, 32.768)
plotter = p.Plotter(solution_ack)
search = sa.SimulatedAnnealing(solution_ack, f.ackley, plotter, 100, 0.1, 0.95, 5)
search.search(visualize=True)

solution_ras = s.Solution(2, -5.12, 5.12)
plotter = p.Plotter(solution_ras)
search = sa.SimulatedAnnealing(solution_ras, f.rastrigin, plotter, 100, 0.1, 0.95, 1)
search.search(visualize=True)

solution_ros = s.Solution(2, -5, 10)
plotter = p.Plotter(solution_ros)
search = sa.SimulatedAnnealing(solution_ros, f.rosenbrock, plotter, 100, 0.1, 0.95, 2)
search.search(visualize=True)

solution_gri = s.Solution(5, -600, 600)
plotter = p.Plotter(solution_gri)
search = sa.SimulatedAnnealing(solution_gri, f.griewank, plotter, 100, 0.1, 0.95, 120)
search.search(visualize=True)

solution_sch = s.Solution(2, -500, 500)
plotter = p.Plotter(solution_sch)
search = sa.SimulatedAnnealing(solution_sch, f.schwefel, plotter, 100, 0.1, 0.95, 100)
search.search(visualize=True)

solution_lev = s.Solution(2, -10, 10)
plotter = p.Plotter(solution_lev)
search = sa.SimulatedAnnealing(solution_lev, f.levy, plotter, 100, 0.1, 0.95, 2)
search.search(visualize=True)

solution_mic = s.Solution(2, 0, math.pi)
plotter = p.Plotter(solution_mic)
search = sa.SimulatedAnnealing(solution_mic, f.michalewicz, plotter, 100, 0.1, 0.95, 2)
search.search(visualize=True)

solution_zak = s.Solution(2, -5, 10)
plotter = p.Plotter(solution_zak)
search = sa.SimulatedAnnealing(solution_zak, f.zakharov, plotter, 100, 0.1, 0.95, 2)
search.search(visualize=True)