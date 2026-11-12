from pathlib import Path
import newton

msh_path = str(Path(__file__).parent / "fish_tail.msh")

tet_mesh = newton.TetMesh.create_from_file(msh_path)

print("vertices:", tet_mesh.vertices.shape)
print("tetraedros:", tet_mesh.tet_count)
print("x min/max [mm]:", tet_mesh.vertices[:, 0].min(), tet_mesh.vertices[:, 0].max())
print("y min/max [mm]:", tet_mesh.vertices[:, 1].min(), tet_mesh.vertices[:, 1].max())
print("z min/max [mm]:", tet_mesh.vertices[:, 2].min(), tet_mesh.vertices[:, 2].max())