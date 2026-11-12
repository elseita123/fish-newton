import math
from pathlib import Path
import gmsh

stl_path = str(Path(__file__).parent / "fish_tail_mesh.stl")
h = 3.0

gmsh.initialize()
gmsh.merge(stl_path)

gmsh.model.mesh.classifySurfaces(30 * math.pi / 180, True, False, 180 * math.pi / 180)
gmsh.model.mesh.createGeometry()

surfaces = gmsh.model.getEntities(2)
loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
gmsh.model.geo.addVolume([loop])
gmsh.model.geo.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMax", h)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3 * h)

gmsh.model.mesh.generate(3)

node_tags, _, _ = gmsh.model.mesh.getNodes()
elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
print("h [mm]:", h)
print("nodos:", len(node_tags))
print("tetraedros:", sum(len(t) for t in elem_tags))

gmsh.write(str(Path(__file__).parent / "fish_tail.msh"))
gmsh.finalize()