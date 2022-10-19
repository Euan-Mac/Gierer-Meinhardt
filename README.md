# Gierer-Meinhardt
Project desgined to simulate turing patterns formation from Gierer-Meinhardt equations using fenicsx package. 

Right now most of the code to run a simulation is contained in the main.py. However, the code to make the mesh is
contained in the mesh_generator.py file, which calls the MATLAB script image_procressing.m to find the boundaries of a
given image. These boundaries are then exported to python which uses the gmsh python API to construct a mesh based on
the image. This is called at the start of main.py right now for a standard gecko image. The MATLAB script will probably
need to be edited for different images.

The process is easily parrllelisable (can't spell that) by simply calling mpirun -np 4 main.py.

The neccesary environments can be set up in conda. For example

conda create fenicsx-env
cona activate fenicsx-env
conda install conda install -c conda-forge fenics-dolfinx mpich python-gmsh


You also need to set-up MATLAB and python to comminicate. You do this by

cd /Applications/MATLAB_R2022b.app/extern/engines/python # (your MATLAB root maybe different)
python setup.py install --prefix="installdir" # installdir should be the path to your conda environment
