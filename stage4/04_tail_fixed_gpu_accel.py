import time
import numpy as np
import warp as wp
import newton
import newton.examples
from pathlib import Path


def simulate():
    global state_0, state_1

    for _ in range(sim_substeps):
        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, sim_dt)
        state_0, state_1 = state_1, state_0


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

q0 = model.particle_q.numpy()
print("min:", q0.min(axis=0))
print("max:", q0.max(axis=0))
print("total mass:", model.particle_mass.numpy().sum())

i_tip = int(np.argmin(q0[:, 0]))

model.soft_contact_ke = 1.0e2
model.soft_contact_kd = 0.0
model.soft_contact_mu = 1.0

iterations = 10

solver = newton.solvers.SolverVBD(
    model = model,
    iterations = iterations,
    particle_enable_self_contact = False,
    particle_enable_tile_solve = True,
)

state_0 = model.state()
state_1 = model.state()
control = model.control()

collision_pipeline = newton.CollisionPipeline(model)
contacts = collision_pipeline.contacts()

sim_substeps = 10
assert sim_substeps % 2 == 0, "sim_substeps debe ser par para el grafo CUDA"
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

print("Warp device:", wp.get_device())
print("CUDA available:", wp.is_cuda_available())

viewer, args = newton.examples.init(parser)
viewer.set_model(model)
if args.viewer == "gl":
    viewer.hide_loading_splash()

viewer.set_camera(
    pos = wp.vec3(-0.08, -0.45, 1.0),
    pitch = 0.0,
    yaw = 90.0,
)

graph = None

if wp.get_device().is_cuda:
    with wp.ScopedCapture(device=model.device) as capture:
        simulate()

    graph = capture.graph
    print("Graph captured successfully")

frame = 0
t_start = time.perf_counter()

while viewer.is_running():
    if graph is not None:
        wp.capture_launch(graph)
    else:
        simulate()

    if frame % 30 == 0:
        p = state_0.particle_q.numpy()[i_tip]
        dz = (p[2] - q0[i_tip][2]) * 1000.0
        fps_real = (frame + 1) / (time.perf_counter() - t_start)
        print(f"t = {sim_time:.2f} s   dz = {dz:.2f} mm   fps = {fps_real:.2f}")
    frame += 1

    viewer.begin_frame(sim_time)
    viewer.log_state(state_0)
    viewer.end_frame()

    sim_time += frame_dt
