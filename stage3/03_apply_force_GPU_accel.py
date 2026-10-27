import math
import warp as wp
import newton 
import newton.examples


@wp.kernel
def apply_lateral_force(
    particle_q: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    sim_time_gpu: wp.array(dtype=float),
    force_amp: float,
    frequency: float,
):
    i = wp.tid()
    p = particle_q[i]

    t = sim_time_gpu[0]
    force_y = force_amp * wp.sin(2.0 * 3.141592653589793 * frequency * t)

    if (
        wp.abs(p[0] - 0.16) < 0.001
        and p[1] >= 0.0 and p[1] <= 0.06
        and p[2] >= 1.0 and p[2] <= 1.10
    ):
        particle_f[i] = particle_f[i] + wp.vec3(0.0, force_y, 0.0)


@wp.kernel
def advance_time(
    sim_time_gpu: wp.array(dtype=float),
    dt: float,
):
    sim_time_gpu[0] = sim_time_gpu[0] + dt



def simulate():
    global state_0, state_1

    for _ in range(sim_substeps):
        state_0.clear_forces()

        wp.launch(
            kernel=apply_lateral_force,
            dim=model.particle_count,
            inputs=[
                state_0.particle_q,
                state_0.particle_f,
                sim_time_gpu,
                1.0,
                2.0,
            ],
            device=model.device,
        )

        collision_pipeline.collide(state_0, contacts)

        solver.step(
            state_0,
            state_1,
            control,
            contacts,
            sim_dt,
        )

        wp.launch(
            kernel=advance_time,
            dim=1,
            inputs=[sim_time_gpu, sim_dt],
            device=model.device,
        )

        state_0, state_1 = state_1, state_0


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
    k_mu = 1.5e6,
    k_lambda = 12.95e6, # v (poison ratio) = 0.49
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

sim_time_gpu = wp.array(
    [sim_time],
    dtype=float,
    device=model.device,
)

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
viewer.hide_loading_splash()

viewer.set_camera(
    pos=wp.vec3(0.52, -0.62, 1.08),
    pitch=-5.9,
    yaw=127.4,
)

count = wp.zeros(1, dtype=int, device=model.device)

print("Selected particles:", count.numpy()[0])


graph = None

if wp.get_device().is_cuda:
    with wp.ScopedCapture(device=model.device) as capture:
        simulate()

    graph = capture.graph
    print("Graph captured successfully")


while viewer.is_running():
    if graph is not None:
        wp.capture_launch(graph)
    else:
        simulate()

    viewer.begin_frame(sim_time)
    viewer.log_state(state_0)
    viewer.end_frame()

    sim_time += frame_dt






