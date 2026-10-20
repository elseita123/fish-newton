import warp as wp
import newton
import newton.examples

builder = newton.ModelBuilder()
builder.add_ground_plane()

builder.add_soft_grid(
    pos = wp.vec3(0.0, 0.0, 1.0),
    rot = wp.quat_identity(),
    vel = wp.vec3(0.0, 0.0, 0.0),
    dim_x = 10,
    dim_y = 5,
    dim_z = 5,
    cell_x = 0.1,
    cell_y = 0.1,
    cell_z = 0.1,
    density = 1.0e3,
    k_mu = 1.0e5,
    k_lambda = 1.0e5,
    k_damp = 1.0e2,
    fix_left = True,
)

# Segundo bloque deformable
builder.add_soft_grid(
    pos=wp.vec3(0.5, 0.0, 3.0),
    rot=wp.quat_identity(),
    vel=wp.vec3(0.0, 0.0, 0.0),

    dim_x=5,
    dim_y=5,
    dim_z=5,

    cell_x=0.1,
    cell_y=0.1,
    cell_z=0.1,

    density=1.0e3,
    k_mu=1.0e5,
    k_lambda=1.0e5,
    k_damp=1.0e2,

    fix_left=False,
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
# print("Viewer:", type(viewer))
# print("Arguments:", args)
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









