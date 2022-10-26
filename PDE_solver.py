from mpi4py import MPI
try:
    import matlab.engine
except ModuleNotFoundError:
    # Error handling
    pass
import numpy as np
import gmsh
from os.path import join, basename, splitext
import re
from petsc4py import PETSc
from dolfinx.io import gmshio
from dolfinx.fem import Function, FunctionSpace, Constant
from dolfinx.io import XDMFFile
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, dot, FiniteElement, TestFunctions, split
from dolfinx.fem.petsc import NonlinearProblem
from time import time
import sys


# A class designed for taking image files (pngs, tifs, jpegs, etc) and converting them into meshes suitable for fenicsx.
class ImageMesh:

    def __init__(self, im_file, coords_dir, mesh_dir, white_background=False, filter_rate=1, threshold=0,
                 max_mesh_size=0.01, point_size=0.01):
        self.im_file = im_file  # path to image including file extension
        filename_path_exp = re.compile(r'^(\\\w\\)(.*)(\..*)$')  # reg rexp to extract file name from path
        file_path_matches = filename_path_exp.search(im_file)
        self.filename = splitext(basename(im_file))[0]
        self.coords_dir = coords_dir  # directory to save coordinates of image boundary too, include slash on end
        self.mesh_dir = mesh_dir  # directory to save mesh to, include slash on end

        self.x_bound = None  # eventually x coordinates of image boundasry get written here
        self.y_bound = None
        self.white_background = white_background  # need to add a True boolean variable if the image in question has
        # a white background for the analysis to work
        self.filter_rate = filter_rate  # how much to undersample image boundary by, IE if this equals 3 then we take
        # every 3rd pixel, this leads to faster code and sometimes a smoother image boundary
        self.threshold = threshold  # how white does a  grayscale pixel need to be before it counts as white in the
        # binary version, if left as zero MATLAB will estimate this
        self.max_mesh_size = max_mesh_size  # maximum possilee length of a mesh element as a fraction of the size of the
        # image, increasing this will lead to a lower resolution mesh
        self.point_size = point_size  # similar to previous variable, I don't understand the technical difference

    # function which calls the MATLAB image processing function to get the boundary of the image, see MATLAB comments
    # Note boolean input will determine whether to show the image with the estimated boundary
    def get_boundary(self, do_plot=True):
        eng = matlab.engine.start_matlab()
        if self.threshold == 0:
            c1, c2 = eng.image_processing(self.im_file, self.white_background, self.filter_rate, do_plot, nargout=2)
        else:
            c1, c2 = eng.image_processing(self.im_file, self.white_background, self.filter_rate, do_plot,
                                          self.threshold, nargout=2)
        xs = np.asarray(c2)
        ys = np.asarray(c1)
        self.y_bound = ys  # save boundary for use later
        self.x_bound = xs
        bounds = np.concatenate((xs.transpose(), ys.transpose()))
        np.savetxt(join(self.coords_dir, self.filename + '.txt'), bounds)  # save output to txt file

    # if we have previously found the boundary of an image we can load the txt file and use that instead of re-running MATLAB
    def load_boundary(self):
        bounds = np.loadtxt(join(self.coords_dir, self.filename + '.txt'))
        self.x_bound = bounds[0, :]
        self.y_bound = bounds[1, :]

    # Function which creates a mesh from the image boundary which has been detected
    def make_mesh(self, do_plot=True):
        gmsh.initialize()
        gmsh.model.add(self.filename)
        num_points = len(self.x_bound)

        # make a big outer sqaure that will contain the image
        c1 = gmsh.model.geo.addPoint(-1, -1, 0, self.point_size, num_points + 1)  # add 4 corner points
        c2 = gmsh.model.geo.addPoint(-1, 1, 0, self.point_size, num_points + 2)
        c3 = gmsh.model.geo.addPoint(1, 1, 0, self.point_size, num_points + 3)
        c4 = gmsh.model.geo.addPoint(1, -1, 0, self.point_size, num_points + 4)
        l1 = gmsh.model.geo.addLine(c1, c2)  # join with straight lines
        l2 = gmsh.model.geo.addLine(c2, c3)
        l3 = gmsh.model.geo.addLine(c3, c4)
        l4 = gmsh.model.geo.addLine(c4, c1)
        full_bound = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])  # form closed loop with the lines

        # control resolution of mesh
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.max_mesh_size)

        n = 0
        for x_n, y_n, in zip(self.x_bound, self.y_bound):  # iterate through all x and y points on image boundary
            n += 1
            gmsh.model.geo.addPoint(x_n, y_n, 0, self.point_size, n)  # add each point

        point_tags = [i for i in range(1, n + 1)]
        point_tags.append(1)
        image_curve = gmsh.model.geo.addBSpline(point_tags)  # join these points with a spline curve

        image_bound = gmsh.model.geo.addCurveLoop([image_curve])  # define spline curve to be closed
        image_surface = gmsh.model.geo.addPlaneSurface([image_bound])  # define surface from spline curve
        full_surface = gmsh.model.geo.addPlaneSurface(
            [image_bound, full_bound])  # add another surface for the outer square

        gmsh.model.geo.synchronize()  # update gmsh internally (something to do with gmsh python API)
        gmsh.model.addPhysicalGroup(2, [image_surface], name=self.filename)  # define the actual image as the only
        # "physical" part of the mesh, this will be the only part actually used by fenicsx
        mesh = gmsh.model.mesh.generate(2)  # make mesh
        gmsh.write(join(self.mesh_dir, self.filename + ".msh"))  # save output to file

        if do_plot:  # show mesh if asked to
            gmsh.fltk.run()

        # runs both main functions in one line

    def do__conversion(self, do_plots):
        self.get_boundary(do_plots)
        self.make_mesh(do_plots)


# a class desgined to make using fenicsx on gmsh meshes a slittle simpler
class FiniteElementSolver:
    def __init__(self, space_dim):
        self.space_dim = space_dim  # spatial dimension of image
        # ( right now the code is desgined for 2D systems anyway, not sure if it will generalise)

        self.facet_markers = None  # internal gmsh features used by fenicsx
        self.cell_markers = None
        self.mesh = None

        self.func_space = None  # space our functions will live in

    # loads a saved mesh made by gmsh in a way suitable to be run in parallel
    def load_gmsh(self, path_to_mesh):
        self.mesh, self.cell_markers, self.facet_markers = gmshio.read_from_msh(path_to_mesh, MPI.COMM_WORLD, 0, gdim=3)
        # gmsh.initialize()
        # gmsh.open(path_to_mesh)
        # gmsh.model.geo.synchronize()
        # gmsh_model_rank = 0
        # mesh_comm = MPI.COMM_WORLD
        # self.mesh, self.cell_markers, self.facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank,
        #                                                                         self.space_dim)
        # gmsh.finalize()


# a derived class for specifically nonlinear, time dependent PDEs with two coupled fields.
# We call one function a and the other one b, we assume to be solving with some sort of time discretization scheme that
# requires the current and previous values of the field. Dirichelt boundary conditions are assumed
class Coupled_Time_Dep_Solver(FiniteElementSolver):
    def __init__(self, space_dim, dt):
        super().__init__(space_dim)
        self.solver = None  # numerical solver from PETsC
        self.initialised = False  # boolean to check whether system has been initialised properly
        self.b_func_prev = None  # b function at prev time step
        self.a_func_prev = None
        self.b_func = None  # current b function
        self.a_func = None
        self.funcs = None  # tensor product of both functions
        self.funcs_prev = None
        self.test_b = None  # test function for b
        self.test_a = None
        self.dt = dt  # timestep
        self.output = False  # tells us that no output file has been set
        self.residual = None

    def make_function_space(self, basis_type, basis_order):
        if self.mesh is None:
            raise RuntimeError(
                "First thing you must do after initilisation is generate the mesh, as everything else is dependant on this!")

        # define space of function we search within at each node, eg make_function_space("Lagrange", 1)
        # would be the arguements for first order Lagrange polynomials
        element_type = FiniteElement(basis_type, self.mesh.ufl_cell(), basis_order)
        func_space = FunctionSpace(self.mesh, element_type * element_type)
        self.funcs = Function(func_space)  # current solution
        self.funcs_prev = Function(func_space)  # solution from previous converged step
        self.a_func, self.b_func = split(self.funcs)  # split tensor product into 2 coupled functions
        self.a_func_prev, self.b_func_prev = split(self.funcs_prev)
        self.test_a, self.test_b = TestFunctions(func_space)  # make corresponding test functions

    # Takes two functions as inputs and uses them as initial conditions for a and b
    def set_initial_conds(self, initial_conds_func_a, initial_conds_func_b):
        self.funcs.x.array[:] = 0.0  # zero array (everyone always does this in tutorials so I assume its required)
        self.funcs.sub(0).interpolate(initial_conds_func_a)
        self.funcs.sub(1).interpolate(initial_conds_func_b)
        self.funcs.x.scatter_forward()  # I think this actually updates the array associated with the fenicsx function
        self.initialised = True  # remember we have initialised the functions

    # set-up newton solver for non-linear problem all the default arguments seem to work well for the turing patterns
    # simulation, but these seem to have to be adjusted for different systems, so be careful
    def set_up_solver(self, convergence_criterion="incremental", rtol=1e-6, ksp_type="preonly", pc_type="lu",
                      pc_factor_mat_solver_type="mumps"):
        if self.initialised and ~(self.mesh is None) and ~(self.residual is None):  # check previous steps have been run
            problem = NonlinearProblem(self.residual, self.funcs)  # define problem we want to solve
            solver = NewtonSolver(MPI.COMM_WORLD, problem)  # define type of solver we want to use
            solver.convergence_criterion = convergence_criterion  # define how solver is said to be converged
            solver.rtol = rtol  # tolerance for convergence
            # set parameters for non-linear solver
            ksp = solver.krylov_solver  # set solver options
            opts = PETSc.Options()
            option_prefix = ksp.getOptionsPrefix()
            opts[f"{option_prefix}ksp_type"] = ksp_type
            opts[f"{option_prefix}pc_type"] = pc_type  # sets preconditioning of solver
            opts[f"{option_prefix}pc_factor_mat_solver_type"] = pc_factor_mat_solver_type
            ksp.setFromOptions()
            self.solver = solver
        else:
            raise RuntimeError(
                "Need to set intial condtion first using the set_initial_cods method \n and set the residual for the weak form!")

    # make XDMF files for paraview visualisation
    def set_up_output(self, out_file_a, out_file_b):
        if self.residual is None:
            raise RuntimeError("Need to set up residual before you create output files")
        # set-up ability to write to output files
        self.file_a = XDMFFile(MPI.COMM_WORLD, out_file_a, "w")  # make file
        self.file_b = XDMFFile(MPI.COMM_WORLD, out_file_b, "w")  # make file
        self.file_a.write_mesh(self.mesh)  # write mesh
        self.file_b.write_mesh(self.mesh)  # write mesh
        self.a_out = self.funcs.sub(0)  # get a function to write out to the files
        self.b_out = self.funcs.sub(1)
        self.output = True

    # evolve the full system in time for length T
    def evolve_time(self, T, do_print=True):
        if self.solver is None:  # check previous steps have been run
            raise RuntimeError("Can't evolve equation intime until numerical solver is setup!")
        world_comm = MPI.COMM_WORLD  # parallelization settings
        world_size = world_comm.Get_size()
        my_rank = world_comm.Get_rank()

        self.funcs_prev.x.array[:] = self.funcs.x.array  # initialise previous time step values
        for step in range(int(T / self.dt) + 1):  # (for lop better than while loop in parallel )
            t = step * self.dt
            startTime = time()
            r = self.solver.solve(self.funcs)  # get solution
            executionTime = (time() - startTime)

            if step % 10 == 0 and my_rank == 0 and do_print:  # print progress if this setting is on
                print("Simulation time %.2f: num iterations: %s, time: %.2f" % (t, step, executionTime))
                sys.stdout.flush()
            self.funcs_prev.x.array[:] = self.funcs.x.array  # current solution is now old soution

            if self.output:  # write out data if switched on
                self.file_a.write_function(self.a_out, t)  # write out data
                self.file_b.write_function(self.b_out, t)  # write out data

        if self.output:  # close files
            self.file_a.close()
            self.file_b.close()


class GiererMeinhardt(
    Coupled_Time_Dep_Solver):  # a derived class for the specific problem of Gierer-Meinhardt equations
    def __init__(self, path_to_mesh, space_dim, r, mu, nu, Dv, Du, dt):
        super().__init__(space_dim, dt)
        self.load_gmsh(path_to_mesh)  # load mesh from a gmsh file
        self.r = r  # parameters for the equations
        self.mu = mu
        self.nu = nu
        self.Dv = Dv
        self.Du = Du

    def set_up_residual(self):  # sets residual to minimise for the non-linear problem
        if self.funcs is None:  # check everything else has been set-up properly
            raise RuntimeWarning("Need to define function space before you create the weak form!")
        # all constants from the Gierer-Meinhardt model

        r=self.make_constant(self.r)
        mu=self.make_constant(self.mu)
        nu=self.make_constant(self.mu)
        Dv=self.make_constant(self.Dv)
        Du=self.make_constant(self.Du)
        delt=self.make_constant(self.dt)

        #  define weak form of both equations as F0 and F1
        F0 = (((self.a_func - self.a_func_prev) / delt) * self.test_a * dx
              - r * (1 + self.a_func ** 2 / self.b_func) * self.test_a * dx
              + mu * self.a_func * self.test_a * dx
              + Du * dot(grad(self.a_func), grad(self.test_a)) * dx
              )
        F1 = (((self.b_func - self.b_func_prev) / delt) * self.test_b * dx
              - r * self.a_func ** 2 * self.test_b * dx
              + nu * self.b_func * self.test_b * dx
              + Dv * dot(grad(self.b_func), grad(self.test_b)) * dx
              )
        self.residual = F0 + F1  # overall  residual is the sum

    def make_constant(self,param):
        if type(param)==int or type(param)==float:
            return Constant(self.mesh, PETSc.ScalarType(param))
        else:
            return param