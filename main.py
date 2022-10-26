import PDE_solver
import numpy as np
import ufl

# details of image to generate mesh from
in_file= "mesh_images/gecko2.png"
white_background_flag=True
filter_rate=1
do_plot=True
mesh_file= "meshes/square.msh"

#mesh_maker=PDE_solver.ImageMesh(in_file,"./mesh_points/","./meshes/",white_background=white_background_flag,filter_rate=3)
#mesh_maker.get_boundary(do_plot=True)
#mesh_maker.make_mesh(do_plot=False)
dt=0.01
T=100
t=0

# define constants
r = 10
mu = 10
nu = 20
d = 0.01
Dv = 0.27
Du = d*Dv

data_files="./output_data/square_non_random"

eqn=PDE_solver.GiererMeinhardt(mesh_file, 2, r, mu, nu, Dv, Du, dt)
eqn.make_function_space("Lagrange",1)

u_star=(-r/nu+np.sqrt((r/nu)**2 + 4*r*mu)) / (2*r)
v_star=r*(u_star**2)/nu
eqn.set_initial_conds(lambda x: (u_star + 0.1 * v_star * (0.5 - np.random.rand(x.shape[1])) ), lambda x: (v_star + 0.1 * v_star * (0.5 - np.random.rand(x.shape[1])) ) )

x = ufl.SpatialCoordinate(eqn.mesh)
mu=mu*(1+0.5*ufl.cos(1.5*ufl.pi*x[0])*ufl.cos(1.5*ufl.pi*x[1]))
eqn.set_up_residual()

eqn.set_up_solver()

eqn.set_up_output(data_files+"_a.xdmf",data_files+"_b..xdmf")

eqn.evolve_time(T,do_print=True)
