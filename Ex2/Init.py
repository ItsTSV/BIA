from Utils import Functions as f
from Utils import Solution as s
from Utils import Plotter as p
import HillClimbing as hc
import math

# Hill Climbing on all functions
solution_sph = s.Solution(2, -5.12, 5.12)
plotter = p.Plotter(solution_sph)
search = hc.HillClimbing(solution_sph, f.sphere, plotter, 1000, 1)
search.search(visualize=True)

solution_ack = s.Solution(2, -32.768, 32.768)
plotter = p.Plotter(solution_ack)
search = hc.HillClimbing(solution_ack, f.ackley, plotter, 1000, 5)
search.search(visualize=True)

solution_ras = s.Solution(2, -5.12, 5.12)
plotter = p.Plotter(solution_ras)
search = hc.HillClimbing(solution_ras, f.rastrigin, plotter, 1000, 1)
search.search(visualize=True)

solution_ros = s.Solution(2, -5, 10)
plotter = p.Plotter(solution_ros)
search = hc.HillClimbing(solution_ros, f.rosenbrock, plotter, 1000, 2)
search.search(visualize=True)

solution_gri = s.Solution(2, -600, 600)
plotter = p.Plotter(solution_gri)
search = hc.HillClimbing(solution_gri, f.griewank, plotter, 1000, 120)
search.search(visualize=True)

solution_sch = s.Solution(2, -500, 500)
plotter = p.Plotter(solution_sch)
search = hc.HillClimbing(solution_sch, f.schwefel, plotter, 1000, 100)
search.search(visualize=True)

solution_lev = s.Solution(2, -10, 10)
plotter = p.Plotter(solution_lev)
search = hc.HillClimbing(solution_lev, f.levy, plotter, 1000, 2)
search.search(visualize=True)

solution_mic = s.Solution(2, 0, math.pi)
plotter = p.Plotter(solution_mic)
search = hc.HillClimbing(solution_mic, f.michalewicz, plotter, 1000, 0.5)
search.search(visualize=True)

solution_zak = s.Solution(2, -5, 10)
plotter = p.Plotter(solution_zak)
search = hc.HillClimbing(solution_zak, f.zakharov, plotter, 1000, 2)
search.search(visualize=True)
