import taichi as ti
import numpy as np
import cv2
import setups
import math

# Artery image parameters setups
filepath = 'imgs/artery5.png'
entrance = 'up'
exit = 'down'
img = cv2.imread(filepath, 0)
height, width, background, edge, angle, bc_back, entrance_min, entrance_max, entrance_mid, exit_mid = setups.get_param(
    img, entrance=entrance,
    exit=exit)
entrance_count = len(entrance_mid)
exit_count = len(exit_mid)
background = np.rot90(background, -1)
angle = np.rot90(angle, -1)
bc_back = np.rot90(bc_back, -1)

ti.init(arch=ti.gpu)  # Try to run on GPU

quality = width // 128  # Use a larger value for higher-res simulations
# n_particles = 9000 * quality **2
n_particles = 22000 * quality ** 2

# n_grid = 128 * quality
n_grid = width
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

y_max = entrance_max[0]
y_min = entrance_min[0]
# Entrance and Exit BC set
v_entrance = [0, 0]
v_exit = [0, 0]
# Entrance='left' or 'right'
# low_label = (width - y_max) * dx
# Entrance='up' or 'down'
low_label = y_min * dx

# ti.Matrix.var(n, m, dt, shape = None, offset = None)  n-rows m-columns
x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
color_flag = ti.field(dtype=int, shape=n_particles)  # color of particles
c_label = ti.field(dtype=int, shape=1)
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(2, dtype=float, shape=(width, height))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(width, height))  # grid node mass
grid_flag = ti.field(dtype=int, shape=(width, height))  # grid node flag for fluid(1)/edge(255)/wall(0)
grid_angle = ti.field(dtype=float, shape=(width, height))  # grid angle
grid_bc = ti.field(dtype=int, shape=(width, height))  # grid bc flag
gravity = ti.Vector.field(2, dtype=float, shape=())

# Attractor (Entrance) force setups
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=entrance_count)
# Attractor (Exit) force setups
attractor_strength2 = ti.field(dtype=float, shape=())
attractor_pos2 = ti.Vector.field(2, dtype=float, shape=exit_count)

for i in range(entrance_count):
    if entrance == 'up':
        attractor_pos[i] = [dx * entrance_mid[i], 1 - dx]
    if entrance == 'down':
        attractor_pos[i] = [dx * entrance_mid[i], dx]
    if entrance == 'left':
        attractor_pos[i] = [dx, dx * entrance_mid[i]]
    if entrance == 'right':
        attractor_pos[i] = [1 - dx, dx * entrance_mid[i]]

for i in range(exit_count):
    if exit == 'up':
        attractor_pos2[i] = [dx * exit_mid[i], 1 - dx]
    if exit == 'down':
        attractor_pos2[i] = [dx * exit_mid[i], dx]
    if exit == 'left':
        attractor_pos2[i] = [dx, dx * exit_mid[i]]
    if exit == 'right':
        attractor_pos2[i] = [1 - dx, dx * exit_mid[i]]
# if entrance == 'up':
#     attractor_pos[None] = [0.5 * dx * (y_max + y_min), 1 - dx]
# if entrance == 'down':
#     attractor_pos[None] = [0.5 * dx * (y_max + y_min), dx]
# if entrance == 'left':
#     attractor_pos[None] = [dx, 0.5 * dx * (y_max + y_min)]
# if entrance == 'right':
#     attractor_pos[None] = [1 - dx, 0.5 * dx * (y_max + y_min)]


# Taichi
@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)

        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]  # deformation gradient update
        h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p]))))  # Hardening coefficient: snow gets harder when compressed

        # if material[p] == 1:  # jelly, make it softer
        #     h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        # if material[p] == 0:  # liquid
        #     mu = 0.0
        if material[p] == 1:  # liquid
            mu = 0.0
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])  # 矩阵正交分解SVD
        J = 1.0

        # Snow set
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig

        # Jelly and liquid set
        if material[p] == 0 or 1:  # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 2:
            F[p] = U @ sig @ V.transpose()  # Reconstruct elastic deformation gradient after plasticity

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (
                J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]

        # Loop over 3x3 grid node neighborhood
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    # Grid flag setting for Wall area
    for i, j in grid_flag:
        if grid_flag[i, j] == 0:
            grid_m[i, j] = 0
            grid_v[i, j] = [0, 0]

    # Grid Operations(Grid velocity and Boundary Condition)
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]  # Momentum to velocity
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            dist = attractor_pos[0] - dx * ti.Vector([i, j])
            grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100

            # # Boundary conditions (初始图像边界条件)
            # if i < 2 and grid_v[i, j][0] < 0:
            #     grid_v[i, j][0] = 0
            # if i > width - 2 and grid_v[i, j][0] > 0:
            #     grid_v[i, j][0] = 0
            # if j < 2 and grid_v[i, j][1] < 0:
            #     grid_v[i, j][1] = 0
            # if j > height - 2 and grid_v[i, j][1] > 0:
            #     grid_v[i, j][1] = 0

            # Boundary conditions on Edge
            if grid_flag[i, j] == 255:
                beta = 0  # Velocity rebound value [0,1];  0：velocity on normal direction=0
                alpha = grid_angle[i, j]
                if grid_v[i, j][1] * ti.cos(alpha) - grid_v[i, j][0] * ti.sin(alpha) > 0:
                    v_x = grid_v[i, j][0]
                    v_y = grid_v[i, j][1]
                    grid_v[i, j][0] = v_x * ((ti.cos(alpha)) ** 2 + beta * (ti.sin(alpha)) ** 2) + (
                            1 - beta) * v_y * ti.sin(alpha) * ti.cos(alpha)
                    grid_v[i, j][1] = v_y * ((ti.sin(alpha)) ** 2 + beta * (ti.cos(alpha)) ** 2) + (
                            1 - beta) * v_x * ti.sin(alpha) * ti.cos(alpha)

    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection

        # Particles reset
        # Out of image reset
        if x[p][0] <= 0 or x[p][0] >= 1:
            x[p] = [ti.random() * (y_max - y_min) * dx + low_label, 1 - ti.random() * 5 * dx]
            v[p] = v_entrance
            F[p] = ti.Matrix([[1, 0], [0, 1]])
            Jp[p] = 1
            C[p] = ti.Matrix.zero(float, 2, 2)
            color_flag[p] = c_label[0]
        if x[p][1] <= 0 or x[p][1] >= 1:
            x[p] = [ti.random() * (y_max - y_min) * dx + low_label, 1 - ti.random() * 5 * dx]
            v[p] = v_entrance
            F[p] = ti.Matrix([[1, 0], [0, 1]])
            Jp[p] = 1
            C[p] = ti.Matrix.zero(float, 2, 2)
            color_flag[p] = c_label[0]

        # Exit reset
        if exit == 'up':
            if x[p][1] >= 1 - dx:
                # x[p] = [ti.random() * dx, ti.random()*(y_max-y_min) * dx + (0.5*(width + height) - y_max) * dx]
                x[p] = [ti.random() * (y_max - y_min) * dx + low_label, 1 - ti.random() * 5 * dx]
                v[p] = v_entrance
                F[p] = ti.Matrix([[1, 0], [0, 1]])
                Jp[p] = 1
                C[p] = ti.Matrix.zero(float, 2, 2)
                color_flag[p] = c_label[0]
        if exit == 'down':
            if x[p][1] <= dx:
                # x[p] = [ti.random() * dx, ti.random()*(y_max-y_min) * dx + (0.5*(width + height) - y_max) * dx]
                x[p] = [ti.random() * (y_max - y_min) * dx + low_label, 1 - ti.random() * 5 * dx]
                v[p] = v_entrance
                F[p] = ti.Matrix([[1, 0], [0, 1]])
                Jp[p] = 1
                C[p] = ti.Matrix.zero(float, 2, 2)
                color_flag[p] = c_label[0]
        if exit == 'left':
            if x[p][0] <= dx:
                # x[p] = [ti.random() * dx, ti.random()*(y_max-y_min) * dx + (0.5*(width + height) - y_max) * dx]
                x[p] = [ti.random() * (y_max - y_min) * dx + low_label, 1 - ti.random() * 5 * dx]
                v[p] = v_entrance
                F[p] = ti.Matrix([[1, 0], [0, 1]])
                Jp[p] = 1
                C[p] = ti.Matrix.zero(float, 2, 2)
                color_flag[p] = c_label[0]
        if exit == 'right':
            if x[p][0] >= 1 - dx:
                # x[p] = [ti.random() * dx, ti.random()*(y_max-y_min) * dx + (0.5*(width + height) - y_max) * dx]
                x[p] = [ti.random() * (y_max - y_min) * dx + low_label, 1 - ti.random() * 5 * dx]
                v[p] = v_entrance
                F[p] = ti.Matrix([[1, 0], [0, 1]])
                Jp[p] = 1
                C[p] = ti.Matrix.zero(float, 2, 2)
                color_flag[p] = c_label[0]


@ti.kernel
def reset():
    # group_size = n_particles // 3
    group_size = n_particles
    gravity[None] = [0, 0]
    c_label[0] = 0
    # flag setup:
    for i in range(n_particles):
        # Original position reset
        # x[i] = [ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size), ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size)]
        x[i] = [ti.random() * (y_max - y_min) * dx + low_label, 1 - ti.random() * 5 * dx]
        material[i] = i // group_size  # 0: fluid
        color_flag[i] = 0  # 0: fluid
        v[i] = [0, 0]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)


# Main
print(
    "[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse buttons to attract/repel. Press R to reset. Press C to change color")
gui = ti.GUI("ArterySimulation", res=width * 2, background_color=0xC0C0C0)  # background_color=0x112F41

grid_flag.from_numpy(background)
grid_angle.from_numpy(angle)
grid_bc.from_numpy(bc_back)
reset()

# Reset Area setups
# if entrance == 'left':
#     reset_x = ti.random() * 5 * dx
#     reset_y = ti.random() * (y_max - y_min) * dx + low_label
# if entrance == 'right':
#     reset_x = 1 - ti.random() * 5 * dx
#     reset_y = ti.random() * (y_max - y_min) * dx + low_label
# if entrance == 'up':
#     [ti.random() * (y_max - y_min) * dx + low_label, 1 - ti.random() * 5 * dx]
# if entrance == 'down':
#     reset_x = ti.random() * (y_max - y_min) * dx + low_label
#     reset_y = ti.random() * 5 * dx

for frame in range(20000):
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'r':
            reset()
            print('Reset Complete')
        elif gui.event.key == 'c':
            c_label[0] += 1
            if c_label[0] == 3:
                c_label[0] = 0
            print('Color Changed. C_label={}'.format(c_label[0]))
        elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            break

    attractor_strength[None] = -(3 + 2 * ti.sin(math.pi * 0.03 * frame))
    attractor_strength2[None] = 3 + 2 * ti.sin(math.pi * 0.03 * frame)

    # if gui.event is not None: gravity[None] = [0, 0]  # if had any event
    # if gui.is_pressed(ti.GUI.LEFT, 'a'): gravity[None][0] = -1
    # if gui.is_pressed(ti.GUI.RIGHT, 'd'): gravity[None][0] = 1
    # if gui.is_pressed(ti.GUI.UP, 'w'): gravity[None][1] = 1
    # if gui.is_pressed(ti.GUI.DOWN, 's'): gravity[None][1] = -1
    # Entrance BC by acceleration [F=ma]
    # mouse = gui.get_cursor_pos()
    # gui.circle((mouse[0], mouse[1]), color=0x336699, radius=10)
    # attractor_pos[None] = [mouse[0], mouse[1]]
    # attractor_strength[None] = 0
    # if gui.is_pressed(ti.GUI.LMB):
    #     attractor_strength[None] = 5
    # if gui.is_pressed(ti.GUI.RMB):
    #     attractor_strength[None] = -5
    # # Velocity condition for artery
    # v_acc = [0, 0]
    # # v_acc = [3 + 2 * ti.sin(math.pi * 0.03 * frame), 0]
    # # v_acc = [3 + 2 * ti.sin(math.pi * 0.03 * frame), 3 + 2 * ti.sin(math.pi * 0.03 * frame)]
    # gravity[None] = v_acc

    for s in range(int(2e-3 // dt)):
        substep()
    # colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
    colors = np.array([0xffffff, 0x000000, 0x00008b], dtype=np.uint32)
    gui.circles(x.to_numpy(), radius=0.8, color=colors[color_flag.to_numpy()])
    gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk
