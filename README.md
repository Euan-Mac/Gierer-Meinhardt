# Gierer-Meinhardt
Project desgined to simulate turing patterns formation from Gierer-Meinhardt equations using fenicsx package. 

The code is contained in two clas structures entitled ImageMesh and FiniteElementSolver. The former one is designed to call the  MATLAB function image_procressing.m to find the boundaries of a given image. These boundaries are then exported to python which uses the gmsh python API to construct a mesh based on the image. This is called at the start of main.py right now for a standard gecko image. The MATLAB script will probably need to be edited for different images.

The actual finite elemnt PDE solution is wrappedin the other class (FiniteElementSolver) which does things like define the function space, the weak from, the Newton solver, etc.

How to use both classes is demonstrated in main.py. 

The process is easily parrllelisable (can't spell that) by simply calling mpirun -np 4 main.py.

The neccesary environments can be set up in conda. For example

conda create fenicsx-env
cona activate fenicsx-env
conda install conda install -c conda-forge fenics-dolfinx mpich python-gmsh


You also need to set-up MATLAB and python to comminicate. You do this by

cd /Applications/MATLAB_R2022b.app/extern/engines/python # (your MATLAB root maybe different)
python setup.py install --prefix="installdir" # installdir should be the path to your conda environment
