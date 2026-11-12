import math
import numpy as np
import warp as wp
import newton 
import newton.examples
from pathlib import Path

tail_path = str(Path(__file__).parent.parent / "tools" / "fish_tail.msh")
tail_mesh = newton.TetMesh.create_from_file(tail_path)



@wp.kernel
def apply_lateral_force(
    loaded: wp.array[int],
    particle_f: wp.array[wp.vec3],
    force_y: float,
):
    i = wp.tid()
    if loaded[i] == 1:
        particle_f[i] = particle_f[i] + wp.vec3(0.0, force_y, 0.0)


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

q0 = model.particle_q.numpy()
mask = (q0[:, 0] < -0.155).astype(np.int32)
n_loaded = int(mask.sum())
loaded = wp.array(mask, dtype=int, device=model.device)

i_tip = int(np.argmin(q0[:, 0]))

force_amp = 0.0
frequency = 1.0
force_per_particle = force_amp / n_loaded
print("loaded particles:", n_loaded)


print("particles:", model.particle_count)
print("tetrahedra:", model.tet_count)

q = model.particle_q.numpy()
print("min:", q.min(axis=0))
print("max:", q.max(axis=0))

print("total mass:", model.particle_mass.numpy().sum())

model.soft_contact_ke = 1.0e2
model.soft_contact_kd = 0.0
model.soft_contact_mu = 1.0

iterations = 100

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

sim_substeps = 40
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


viewer.set_camera(
    pos=wp.vec3(0.52, -0.62, 1.08),
    pitch=-5.9,
    yaw=127.4,
)

count = wp.zeros(1, dtype=int, device=model.device)

print("Selected particles:", count.numpy()[0])

frame = 0

while viewer.is_running():

    for substep in range(sim_substeps):
        state_0.clear_forces()
        viewer.apply_forces(state_0)

        t = sim_time + substep * sim_dt
        force_y = force_per_particle * math.sin(2.0 * math.pi * frequency * t)

        wp.launch(
            kernel = apply_lateral_force,
            dim = model.particle_count,
            inputs = [loaded, state_0.particle_f, force_y],
        )

        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, sim_dt)
        state_0, state_1 = state_1, state_0

    if frame % 30 == 0:
        p = state_0.particle_q.numpy()[i_tip]
        dy = (p[1] - q0[i_tip][1]) * 1000.0
        dz = (p[2] - q0[i_tip][2]) * 1000.0
        print(f"t = {sim_time:.2f} s   dy = {dy:.2f} mm   dz = {dz:.2f} mm")
    frame += 1

    viewer.begin_frame(sim_time)
    viewer.log_state(state_0)
    viewer.end_frame()

        
    sim_time += frame_dt









