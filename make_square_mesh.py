import gmsh
gmsh.initialize()
gmsh.model.add("square")
lc =0.02
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(2, 0, 0, lc, 2)
gmsh.model.geo.addPoint(2, 2, 0, lc, 3)
gmsh.model.geo.addPoint(0, 2, 0, lc, 4)
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)
gmsh.model.geo.addCurveLoop([1,2,3,4], 1)
surf=gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.synchronize()
gmsh.model.addPhysicalGroup(2,[1],1)
gmsh.model.setPhysicalName(2,surf, "square")
gmsh.model.mesh.generate(2)
gmsh.write("meshes/square.msh")
gmsh.fltk.run()



