from Utils import Functions as f
from Utils import Solution as s
from Utils import Plotter as p
import ParticleSwarmOptimization as pso
import math

# Hill Climbing on all functions
solution_sph = s.Solution(2, -5.12, 5.12)
plotter = p.Plotter(solution_sph)
search = pso.ParticleSwarmOptimization(solution_sph, f.sphere, plotter, 20, 40)
search.search(visualize=True)

solution_ack = s.Solution(2, -32.768, 32.768)
plotter = p.Plotter(solution_ack)
search = pso.ParticleSwarmOptimization(solution_ack, f.ackley, plotter, 20, 40)
search.search(visualize=True)

solution_ras = s.Solution(2, -5.12, 5.12)
plotter = p.Plotter(solution_ras)
search = pso.ParticleSwarmOptimization(solution_ras, f.rastrigin, plotter, 20, 40)
search.search(visualize=True)

solution_ros = s.Solution(3, -10, 10)
plotter = p.Plotter(solution_ros)
search = pso.ParticleSwarmOptimization(solution_ros, f.rosenbrock, plotter, 20, 40)
search.search(visualize=True)

solution_gri = s.Solution(2, -10, 10)
plotter = p.Plotter(solution_gri)
search = pso.ParticleSwarmOptimization(solution_gri, f.griewank, plotter, 20, 40)
search.search(visualize=True)

solution_sch = s.Solution(2, -400, 400)
plotter = p.Plotter(solution_sch)
search = pso.ParticleSwarmOptimization(solution_sch, f.schwefel, plotter, 20, 40)
search.search(visualize=True)

solution_lev = s.Solution(2, -10, 10)
plotter = p.Plotter(solution_lev)
search = pso.ParticleSwarmOptimization(solution_lev, f.levy, plotter, 20, 40)
search.search(visualize=True)

solution_mic = s.Solution(2, 0, math.pi)
plotter = p.Plotter(solution_mic)
search = pso.ParticleSwarmOptimization(solution_mic, f.michalewicz, plotter, 20, 40)
search.search(visualize=True)

solution_zak = s.Solution(2, -10, 10)
plotter = p.Plotter(solution_zak)
search = pso.ParticleSwarmOptimization(solution_zak, f.zakharov, plotter, 20, 40)
search.search(visualize=True)
