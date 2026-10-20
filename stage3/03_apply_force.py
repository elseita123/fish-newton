import math
import warp as wp
import newton 
import newton.examples


@wp.kernel
def apply_lateral_force(
    particle_q: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    force_y: float,
):
    i = wp.tid()
    p = particle_q[i]

    if p[0] > 0.14 and wp.abs(p[1] - 0.03) < 0.006 and wp.abs(p[2] - 1.05) < 0.006:
        particle_f[i] = particle_f[i] + wp.vec3(0.0, force_y, 0.0)


builder = newton.ModelBuilder()
builder.add_ground_plane()

builder.add_soft_grid(
    pos = wp.vec3(0.0, 0.0, 1.0),
    rot = wp.quat_identity(),
    vel = wp.vec3(0.0, 0.0, 0.0),
    dim_x = 16,
    dim_y = 6,
    dim_z = 10,
    cell_x = 0.01,
    cell_y = 0.01,
    cell_z = 0.01,

    density = 1.15e3,
    k_mu = 2.64e5,
    k_lambda = 12.95e6, # v (poison) = 0.49
    k_damp = 1.0e2,
    fix_left = True,
)

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

    for substep in range(sim_substeps):
        state_0.clear_forces()
        viewer.apply_forces(state_0)

        force_amp = 0.1
        frequency = 0.5
        t = sim_time + substep * sim_dt
        force_y = force_amp * math.sin(2.0 * math.pi * frequency * t)

        wp.launch(
            kernel=apply_lateral_force,
            dim=model.particle_count,
            inputs=[state_0.particle_q, state_0.particle_f, force_y],
        )

        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, sim_dt)
        state_0, state_1 = state_1, state_0

    viewer.begin_frame(sim_time)
    viewer.log_state(state_0)
    viewer.end_frame()
        
    sim_time += frame_dt









