from Utils import Functions as f
from Utils import Solution as s
from Utils import Plotter as p
import SomaAllToOne as soma
import math

solution_sph = s.Solution(2, -5.12, 5.12)
plotter = p.Plotter(solution_sph)
search = soma.SomaAllToOne(solution_sph, f.sphere, plotter)
search.search(visualize=True)

solution_ack = s.Solution(2, -32.768, 32.768)
plotter = p.Plotter(solution_ack)
search = soma.SomaAllToOne(solution_ack, f.ackley, plotter)
search.search(visualize=True)

solution_ras = s.Solution(2, -5.12, 5.12)
plotter = p.Plotter(solution_ras)
search = soma.SomaAllToOne(solution_ras, f.rastrigin, plotter)
search.search(visualize=True)

solution_ros = s.Solution(3, -10, 10)
plotter = p.Plotter(solution_ros)
search = soma.SomaAllToOne(solution_ros, f.rosenbrock, plotter)
search.search(visualize=True)

solution_gri = s.Solution(2, -10, 10)
plotter = p.Plotter(solution_gri)
search = soma.SomaAllToOne(solution_gri, f.griewank, plotter)
search.search(visualize=True)

solution_sch = s.Solution(2, -400, 400)
plotter = p.Plotter(solution_sch)
search = soma.SomaAllToOne(solution_sch, f.schwefel, plotter)
search.search(visualize=True)

solution_lev = s.Solution(2, -10, 10)
plotter = p.Plotter(solution_lev)
search = soma.SomaAllToOne(solution_lev, f.levy, plotter)
search.search(visualize=True)

solution_mic = s.Solution(2, 0, math.pi)
plotter = p.Plotter(solution_mic)
search = soma.SomaAllToOne(solution_mic, f.michalewicz, plotter)
search.search(visualize=True)

solution_zak = s.Solution(2, -10, 10)
plotter = p.Plotter(solution_zak)
search = soma.SomaAllToOne(solution_zak, f.zakharov, plotter)
search.search(visualize=True)
