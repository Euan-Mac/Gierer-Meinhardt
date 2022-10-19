import matlab.engine
import gmsh
import sys
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI

def make_mesh(in_dir,in_file,show):
    eng = matlab.engine.start_matlab()
    c1,c2=eng.image_processing(in_dir,in_file,nargout=2)
    c1=np.array(c1)
    c2=np.array(c2)
    gmsh.initialize()
    gmsh.model.add(in_file)
    gmsh.model.addDiscreteEntity(1, 100)
    flat_pts = []
    for x,y in zip(c1,c2):
        flat_pts.append(float(x))
        flat_pts.append(float(y))
        flat_pts.append(0)

    gmsh.model.mesh.addNodes(1, 100, range(1, len(c1) + 1), flat_pts)
    n = [item for sublist in [[i, i + 1] for i in range(1, len(c1) + 1)] for item in sublist]
    n[-1] = 1
    gmsh.model.mesh.addElements(1, 100, [1], [range(1, len(c1) + 1)], [n])
    gmsh.model.geo.addCurveLoop([100], 101)
    gmsh.model.geo.addPlaneSurface([101], 102)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [102], 1)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.0005)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.0005)
    mesh=gmsh.model.mesh.generate(2)
    gmsh.write("./meshes/"+in_file+".msh")
    if show:
        gmsh.fltk.run()
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    domain, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, mesh_comm, gmsh_model_rank, 2)
    gmsh.finalize()
    return domain, cell_markers, facet_markers