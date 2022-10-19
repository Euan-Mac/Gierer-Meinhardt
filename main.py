import mesh_generator
from dolfinx import log, plot
from dolfinx.fem import Function, FunctionSpace, Constant
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, dot, FiniteElement, TestFunctions, split
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import time
import sys
import os

log.set_output_file("turing_patterns/log.txt")

# details of image to generate mesh from
in_dir='mesh_images/'
in_file='gecko2'

# generate mesh using MATLAB image analysis tools and gmsh
msh, cell_markers, facet_markers=mesh_generator.make_mesh(in_dir,in_file,False)

# define constants
r = 0.03
mu = 1
nu = 4
d = 0.001
Dv = 0.027
Du = d*Dv

dt=0.01
T=1000
t=0


# overall function space for the problem
P1 = FiniteElement("Lagrange", msh.ufl_cell(), 2)
ME = FunctionSpace(msh, P1 * P1)

# Trial and test functions of the space `ME` are now defined:
v_a, v_b = TestFunctions(ME)
u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step
# Split mixed functions
a, b = split(u)
a_n, b_n = split(u0)

# set initial conds
u.x.array[:] = 0.0 # zero array
# Interpolate initial conditions (random)
u_star=(-r/nu+np.sqrt((r/nu)**2 + 4*r*mu)) / (2*r)
v_star=r*(u_star**2)/nu
u.sub(0).interpolate(lambda x: (u_star + 0.1 * u_star * (0.5 - np.random.rand(x.shape[1])) ) )
u.sub(1).interpolate(lambda x: (v_star + 0.1 * v_star * (0.5 - np.random.rand(x.shape[1])) ) )
#u.sub(0).interpolate(lambda x: 2 *(( (x[0]-1)**2 + (x[1]-1)**2 ) < 0.2) + 1.5 * ~(( (x[0]-1)**2 + (x[1]-1)**2 ) < 0.2))
#u.sub(1).interpolate(lambda x: 1 *(( (x[0]-1)**2 + (x[1]-1)**2 ) < 0.2) + 0.8 * ~(( (x[0]-1)**2 + (x[1]-1)**2 ) < 0.2))
u.x.scatter_forward()


# all constants from the Gierer-Meinhardt model
r = Constant(msh, PETSc.ScalarType(r))
mu = Constant(msh, PETSc.ScalarType(mu))
nu = Constant(msh, PETSc.ScalarType(nu))
Dv = Constant(msh, PETSc.ScalarType(Dv))
Du = Constant(msh, PETSc.ScalarType(Du))
delt = Constant(msh, PETSc.ScalarType(dt))

# Weak statement of the equations - see one note
F0 = ((a - a_n) / delt) * v_a * dx - r * (1 + a**2 / b) * v_a * dx + mu* a * v_a * dx + Du * dot(grad(a),grad(v_a)) * dx
F1 = ((b - b_n) / delt) * v_b * dx - r * a**2 * v_b * dx + nu * b * v_b * dx + Dv * dot(grad(b),grad(v_b)) * dx
F = F0 + F1

# since problem is non inear we solve with Newton's method
problem = NonlinearProblem(F, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
# set parameters for non-linear solver
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# set-up ability to write to output files
file = XDMFFile(MPI.COMM_WORLD, "output_data/a_field.xdmf", "w") # make file
file2 = XDMFFile(MPI.COMM_WORLD, "output_data/b_field.xdmf", "w") # make file
file.write_mesh(msh) # write mesh
file2.write_mesh(msh) # write mesh
a = u.sub(0)
b = u.sub(1)
u0.x.array[:] = u.x.array

n=0
# loop in time to sole problem
while (t < T):
    t += dt
    startTime = time.time()
    r = solver.solve(u)
    executionTime = (time.time() - startTime)
    if n%10==0:
        print(f"Step {int(t/dt)}: num iterations: {r[0]}, time: %.2f"%(executionTime))
        sys.stdout.flush()
    u0.x.array[:] = u.x.array # current solution is now old soution
    file.write_function(a, t) # write out data
    file2.write_function(b, t)  # write out data
    n+=1

file.close()
file2.close()
