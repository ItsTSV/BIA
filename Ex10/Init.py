from Utils import Functions as f
from Utils import Solution as s
from Utils import Plotter as p
import TeachingLearningOptimization as tlo
import math

solution_sph = s.Solution(2, -5.12, 5.12)
plotter = p.Plotter(solution_sph)
search = tlo.TeachingLearningOptimization(solution_sph, f.sphere, plotter)
search.search(visualize=True)

solution_ack = s.Solution(2, -32.768, 32.768)
plotter = p.Plotter(solution_ack)
search = tlo.TeachingLearningOptimization(solution_ack, f.ackley, plotter)
search.search(visualize=True)

solution_ras = s.Solution(2, -5.12, 5.12)
plotter = p.Plotter(solution_ras)
search = tlo.TeachingLearningOptimization(solution_ras, f.rastrigin, plotter)
search.search(visualize=True)

solution_ros = s.Solution(3, -10, 10)
plotter = p.Plotter(solution_ros)
search = tlo.TeachingLearningOptimization(solution_ros, f.rosenbrock, plotter)
search.search(visualize=True)

solution_gri = s.Solution(2, -10, 10)
plotter = p.Plotter(solution_gri)
search = tlo.TeachingLearningOptimization(solution_gri, f.griewank, plotter)
search.search(visualize=True)

solution_sch = s.Solution(2, -400, 400)
plotter = p.Plotter(solution_sch)
search = tlo.TeachingLearningOptimization(solution_sch, f.schwefel, plotter)
search.search(visualize=True)

solution_lev = s.Solution(2, -10, 10)
plotter = p.Plotter(solution_lev)
search = tlo.TeachingLearningOptimization(solution_lev, f.levy, plotter)
search.search(visualize=True)

solution_mic = s.Solution(2, 0, math.pi)
plotter = p.Plotter(solution_mic)
search = tlo.TeachingLearningOptimization(solution_mic, f.michalewicz, plotter)
search.search(visualize=True)

solution_zak = s.Solution(2, -10, 10)
plotter = p.Plotter(solution_zak)
search = tlo.TeachingLearningOptimization(solution_zak, f.zakharov, plotter)
search.search(visualize=True)
