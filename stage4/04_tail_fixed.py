import warp as wp
import newton
import newton.examples
from pathlib import Path

tail_path = str(Path(__file__).parent.parent / "tools" / "fish_tail.msh")
tail_mesh = newton.TetMesh.create_from_file(tail_path)

builder = newton.ModelBuilder()
builder.add_ground_plane()

start_particle = len(builder.particle_q)

builder.add_soft_mesh(
    pos = wp.vec3(0.0, 0.0, 1.0),
    rot = wp.quat_identity(),
    scale = 0.001,
    vel = wp.vec3(0.0, 0.0, 0.0),
    vertices = tail_mesh.vertices,
    indices = tail_mesh.tet_indices,
    density = 1.15e3,
    k_mu = 1.5e6,
    k_lambda = 12.95e6,
    k_damp = 1.0e2,
    validate_mesh = True,
)


# A particle with mass = 0.0 is fixed.
fixed = 0
for i, v in enumerate(tail_mesh.vertices):
    if v[0] > -0.5:
        builder.particle_mass[start_particle + i] = 0.0
        fixed += 1
print("fixed particles:", fixed)

builder.color()
model = builder.finalize()

print("particles:", model.particle_count)
print("tetrahedra:", model.tet_count)

q = model.particle_q.numpy()
print("min:", q.min(axis=0))
print("max:", q.max(axis=0))

print("total mass:", model.particle_mass.numpy().sum())

model.soft_contact_ke = 1.0e2
model.soft_contact_kd = 0.0
model.soft_contact_mu = 1.0

iterations = 10

solver = newton.solvers.SolverVBD(
    model = model,
    iterations = iterations,
    particle_enable_self_contact = False,
    particle_enable_tile_solve = False,
)

state_0 = model.state()
state_1 = model.state()
control = model.control()

collision_pipeline = newton.CollisionPipeline(model)
contacts = collision_pipeline.contacts()

sim_substeps = 10
sim_time = 0.0
fps = 60
frame_dt = 1.0 / fps
sim_dt = frame_dt / sim_substeps

parser = newton.examples.create_parser()
parser.add_argument(
    "--solver",
    help ="Type of solver",
    type = str,
    choices = ["vbd"],
    default = "vbd",
)



viewer, args = newton.examples.init(parser)
viewer.hide_loading_splash()
viewer.set_model(model)

running = True

while viewer.is_running():

    for _ in range(sim_substeps):
        state_0.clear_forces()

        viewer.apply_forces(state_0)

        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, sim_dt)

        state_0, state_1 = state_1, state_0

    viewer.begin_frame(sim_time)
    viewer.log_state(state_0)
    viewer.end_frame()
        
    sim_time += frame_dt









