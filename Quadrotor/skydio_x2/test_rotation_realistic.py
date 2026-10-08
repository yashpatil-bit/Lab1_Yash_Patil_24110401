import time
import math
import glfw
import mujoco
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "/home/yash/projects/skydio_x2/scene.xml"

# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

# x2.xml uses 0.01 s. A quadrotor attitude loop needs a faster step.
# 0.002 s = 500 Hz physics + 500 Hz control.
SIM_TIMESTEP = 0.002

# ------------------------------------------------------------
# Pilot commands (what the keys ask for)
# ------------------------------------------------------------

MOVE_SPEED = 2.0                    # m/s   W/S/A/D commanded speed
VERTICAL_SPEED = 0.6                # m/s   R/F commanded climb/descent
YAW_SPEED = math.radians(60)        # rad/s Q/E commanded yaw rate

# ------------------------------------------------------------
# Position / velocity controller (outer loop)
#
#   a_des = KP_POS * (p_target - p) + KD_POS * (v_cmd - v)
#
# order is [X, Y, Z] in the WORLD frame
# ------------------------------------------------------------

KP_POS = np.array([2.0, 2.0, 6.0])      # 1/s^2
KD_POS = np.array([2.5, 2.5, 5.0])      # 1/s

MAX_POS_LEAD = 0.6                  # m   how far the target may run ahead
MAX_TILT = math.radians(30)         # rad hard limit on roll/pitch command
MIN_ALTITUDE = 0.15                 # m   lowest allowed altitude target (COM)

# ------------------------------------------------------------
# Attitude controller (inner loop)
#
#   alpha_des = -KP_ATT * e_R - KD_ATT * omega
#
# order is [roll, pitch, yaw] in the BODY frame
# ------------------------------------------------------------

KP_ATT = np.array([160.0, 160.0, 25.0])  # 1/s^2
KD_ATT = np.array([22.0, 22.0, 9.0])     # 1/s

MAX_YAW_LEAD = 0.6                  # rad how far the yaw target may run ahead

# ------------------------------------------------------------
# Motor / aerodynamic realism
# ------------------------------------------------------------

MOTOR_TAU = 0.02                    # s    motor + prop spin-up lag
DRAG_LINEAR = 0.7                   # N*s/m  body drag (0 = no drag, no cruise tilt)

# ------------------------------------------------------------
# Reference frames
# ------------------------------------------------------------

FRAME_LENGTH = 0.35
FRAME_WIDTH = 0.015

# ------------------------------------------------------------
# Initial camera
# ------------------------------------------------------------

CAMERA_DISTANCE = 2.5
CAMERA_ELEVATION = -25.0
CAMERA_AZIMUTH = 135.0

# ------------------------------------------------------------
# Mouse
# ------------------------------------------------------------

MOUSE_SENSITIVITY = 0.35
ZOOM_SENSITIVITY = 0.15

GRAVITY = 9.81


# ============================================================
# LOAD MUJOCO MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

model.opt.timestep = SIM_TIMESTEP

# Start from hover keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)


# ============================================================
# FIND X2 BODY
# ============================================================

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "x2"
)

if body_id == -1:
    raise RuntimeError("Could not find body 'x2'.")


# ============================================================
# FLIGHT CONTROLLER
#
# Pilot input  ->  desired acceleration  ->  desired thrust vector
#              ->  desired attitude      ->  body torques
#              ->  4 motor thrusts       ->  MuJoCo physics
#
# To move forward the controller has to TILT the thrust vector
# forward, so the drone pitches nose-down exactly like a real one.
# ============================================================

class QuadFlightController:

    def __init__(self, model, data, body_id):

        self.model = model
        self.data = data
        self.body_id = body_id

        # ----------------------------------------------------
        # Mass, centre of mass and inertia (read from MuJoCo)
        # ----------------------------------------------------

        self.mass = float(model.body_mass[body_id])

        com_local = model.body_ipos[body_id].copy()

        # Principal inertia -> full 3x3 inertia matrix in body frame
        Ri = np.zeros(9)
        mujoco.mju_quat2Mat(Ri, model.body_iquat[body_id])
        Ri = Ri.reshape(3, 3)

        self.inertia = (
            Ri
            @ np.diag(model.body_inertia[body_id])
            @ Ri.T
        )

        # ----------------------------------------------------
        # Motor mixer
        #
        # [ T  ]   [  1    1    1    1  ] [F1]
        # [ tx ] = [ y1   y2   y3   y4  ] [F2]
        # [ ty ]   [-x1  -x2  -x3  -x4  ] [F3]
        # [ tz ]   [ g1   g2   g3   g4  ] [F4]
        #
        # x_i, y_i : rotor position relative to the COM
        # g_i      : yaw-torque gear of the motor in x2.xml
        # ----------------------------------------------------

        M = np.zeros((4, 4))

        for j in range(4):

            name = f"thrust{j + 1}"

            site_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                name
            )

            act_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name
            )

            r = model.site_pos[site_id] - com_local

            M[0, j] = 1.0
            M[1, j] = r[1]
            M[2, j] = -r[0]
            M[3, j] = model.actuator_gear[act_id, 5]

        self.mixer_inv = np.linalg.inv(M)

        self.max_motor = float(model.actuator_ctrlrange[0, 1])

        self.reset()

    # --------------------------------------------------------
    # Reset targets to the current state
    # --------------------------------------------------------

    def reset(self):

        R = self.data.xmat[self.body_id].reshape(3, 3)

        self.pos_target = self.data.xipos[self.body_id].copy()

        self.yaw_target = math.atan2(R[1, 0], R[0, 0])

        # Each motor starts at hover thrust
        self.motor_thrust = np.full(
            4,
            self.mass * GRAVITY / 4.0
        )

    # --------------------------------------------------------
    # COM position, COM velocity, rotation, body angular velocity
    # --------------------------------------------------------

    def com_state(self):

        d = self.data
        b = self.body_id

        R = d.xmat[b].reshape(3, 3).copy()

        # Free joint: qvel[3:6] is angular velocity in the BODY frame
        omega_body = d.qvel[3:6].copy()
        omega_world = R @ omega_body

        pos = d.xipos[b].copy()

        # qvel[0:3] is the velocity of the body ORIGIN, not the COM
        vel = d.qvel[0:3] + np.cross(
            omega_world,
            d.xipos[b] - d.xpos[b]
        )

        return pos, vel, R, omega_body

    # --------------------------------------------------------
    # One control step
    #
    # vel_cmd_body : [forward, left] m/s in the drone's heading frame
    # vert_cmd     : climb rate m/s
    # yaw_rate_cmd : rad/s
    # --------------------------------------------------------

    def step(self, vel_cmd_body, vert_cmd, yaw_rate_cmd, dt):

        pos, vel, R, omega = self.com_state()

        yaw = math.atan2(R[1, 0], R[0, 0])

        # ----------------------------------------------------
        # 1. Pilot velocity command -> world frame
        # ----------------------------------------------------

        c = math.cos(yaw)
        s = math.sin(yaw)

        v_cmd = np.array([
            c * vel_cmd_body[0] - s * vel_cmd_body[1],
            s * vel_cmd_body[0] + c * vel_cmd_body[1],
            vert_cmd
        ])

        # ----------------------------------------------------
        # 2. Position target moves with the command, so the
        #    drone holds position when the keys are released
        # ----------------------------------------------------

        self.pos_target += v_cmd * dt

        self.pos_target[2] = max(
            self.pos_target[2],
            MIN_ALTITUDE
        )

        err = self.pos_target - pos

        # Limit how far the target may run ahead of the drone
        n = np.linalg.norm(err[:2])

        if n > MAX_POS_LEAD:
            err[:2] *= MAX_POS_LEAD / n

        err[2] = np.clip(err[2], -MAX_POS_LEAD, MAX_POS_LEAD)

        self.pos_target = pos + err

        # ----------------------------------------------------
        # 3. Desired acceleration -> desired thrust vector
        # ----------------------------------------------------

        a_des = KP_POS * err + KD_POS * (v_cmd - vel)

        f_des = self.mass * (
            a_des + np.array([0.0, 0.0, GRAVITY])
        )

        # Never ask for (almost) zero lift
        f_des[2] = max(f_des[2], 0.2 * self.mass * GRAVITY)

        # Limit the tilt of the thrust vector
        f_h = np.linalg.norm(f_des[:2])
        f_h_max = f_des[2] * math.tan(MAX_TILT)

        if f_h > f_h_max:
            f_des[:2] *= f_h_max / f_h

        # ----------------------------------------------------
        # 4. Yaw target
        # ----------------------------------------------------

        self.yaw_target += yaw_rate_cmd * dt

        yaw_err = (
            self.yaw_target - yaw + math.pi
        ) % (2.0 * math.pi) - math.pi

        yaw_err = np.clip(yaw_err, -MAX_YAW_LEAD, MAX_YAW_LEAD)

        self.yaw_target = yaw + yaw_err

        # ----------------------------------------------------
        # 5. Desired attitude: Z_B along thrust vector,
        #    heading from the yaw target
        # ----------------------------------------------------

        z_des = f_des / np.linalg.norm(f_des)

        x_heading = np.array([
            math.cos(self.yaw_target),
            math.sin(self.yaw_target),
            0.0
        ])

        y_des = np.cross(z_des, x_heading)
        y_des /= np.linalg.norm(y_des)

        x_des = np.cross(y_des, z_des)

        R_des = np.column_stack((x_des, y_des, z_des))

        # ----------------------------------------------------
        # 6. Total thrust = thrust vector projected on Z_B
        # ----------------------------------------------------

        thrust = max(float(f_des @ R[:, 2]), 0.0)

        # ----------------------------------------------------
        # 7. Attitude error -> torque
        # ----------------------------------------------------

        E = R_des.T @ R - R.T @ R_des

        e_R = 0.5 * np.array([
            E[2, 1],
            E[0, 2],
            E[1, 0]
        ])

        alpha_des = -KP_ATT * e_R - KD_ATT * omega

        torque = (
            self.inertia @ alpha_des
            + np.cross(omega, self.inertia @ omega)
        )

        # ----------------------------------------------------
        # 8. Mixer: [T, tx, ty, tz] -> 4 motor thrusts
        # ----------------------------------------------------

        F_cmd = self.mixer_inv @ np.array([
            thrust,
            torque[0],
            torque[1],
            torque[2]
        ])

        F_cmd = np.clip(F_cmd, 0.0, self.max_motor)

        # ----------------------------------------------------
        # 9. Motor lag, then send to MuJoCo
        # ----------------------------------------------------

        k = min(dt / MOTOR_TAU, 1.0)

        self.motor_thrust += (F_cmd - self.motor_thrust) * k

        self.data.ctrl[:4] = self.motor_thrust


controller = QuadFlightController(model, data, body_id)


# ============================================================
# KEYBOARD STATE
# ============================================================

keys = {
    "w": False,
    "a": False,
    "s": False,
    "d": False,
    "q": False,
    "e": False,
    "r": False,
    "f": False,
}


# ============================================================
# MOUSE STATE
# ============================================================

mouse_left_pressed = False

last_mouse_x = 0.0
last_mouse_y = 0.0

camera_azimuth = CAMERA_AZIMUTH
camera_elevation = CAMERA_ELEVATION
camera_distance = CAMERA_DISTANCE


# ============================================================
# RESET
# ============================================================

def reset_simulation():

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    data.xfrc_applied[:] = 0.0

    controller.reset()

    for name in keys:
        keys[name] = False


# ============================================================
# KEYBOARD CALLBACK
# ============================================================

def key_callback(window, key, scancode, action, mods):

    pressed = action != glfw.RELEASE

    # --------------------------------------------------------
    # Movement
    # --------------------------------------------------------

    if key == glfw.KEY_W:
        keys["w"] = pressed

    elif key == glfw.KEY_S:
        keys["s"] = pressed

    elif key == glfw.KEY_A:
        keys["a"] = pressed

    elif key == glfw.KEY_D:
        keys["d"] = pressed

    # --------------------------------------------------------
    # Yaw
    # --------------------------------------------------------

    elif key == glfw.KEY_Q:
        keys["q"] = pressed

    elif key == glfw.KEY_E:
        keys["e"] = pressed

    # --------------------------------------------------------
    # Vertical
    # --------------------------------------------------------

    elif key == glfw.KEY_R:
        keys["r"] = pressed

    elif key == glfw.KEY_F:
        keys["f"] = pressed

    # --------------------------------------------------------
    # Stop: release all commands, drone brakes and hovers
    # --------------------------------------------------------

    elif key == glfw.KEY_SPACE and pressed:

        for name in keys:
            keys[name] = False

    # --------------------------------------------------------
    # Reset to the hover keyframe
    # --------------------------------------------------------

    elif key == glfw.KEY_ENTER and pressed:

        reset_simulation()


# ============================================================
# MOUSE BUTTON CALLBACK
# ============================================================

def mouse_button_callback(window, button, action, mods):

    global mouse_left_pressed
    global last_mouse_x
    global last_mouse_y

    if button == glfw.MOUSE_BUTTON_LEFT:

        if action == glfw.PRESS:

            mouse_left_pressed = True

            last_mouse_x, last_mouse_y = glfw.get_cursor_pos(
                window
            )

        elif action == glfw.RELEASE:

            mouse_left_pressed = False


# ============================================================
# MOUSE MOVEMENT CALLBACK
# ============================================================

def cursor_position_callback(window, xpos, ypos):

    global last_mouse_x
    global last_mouse_y
    global camera_azimuth
    global camera_elevation

    if not mouse_left_pressed:

        last_mouse_x = xpos
        last_mouse_y = ypos

        return

    # Mouse displacement
    dx = xpos - last_mouse_x
    dy = ypos - last_mouse_y

    last_mouse_x = xpos
    last_mouse_y = ypos

    # --------------------------------------------------------
    # Camera rotation
    # --------------------------------------------------------

    # Drag right -> rotate view right
    camera_azimuth -= dx * MOUSE_SENSITIVITY

    # Drag down -> move view down
    camera_elevation += dy * MOUSE_SENSITIVITY

    # Prevent flipping
    camera_elevation = max(
        -89.0,
        min(camera_elevation, 89.0)
    )


# ============================================================
# MOUSE SCROLL CALLBACK
# ============================================================

def scroll_callback(window, xoffset, yoffset):

    global camera_distance

    camera_distance *= math.exp(
        -yoffset * ZOOM_SENSITIVITY
    )

    camera_distance = max(
        0.5,
        min(camera_distance, 10.0)
    )


# ============================================================
# PHYSICS STEP
#
# Keys -> controller -> motor thrusts -> mj_step
# Nothing here writes to qpos. MuJoCo moves the drone.
# ============================================================

def update_control(dt):

    # --------------------------------------------------------
    # Forward / backward
    # --------------------------------------------------------

    forward = 0.0

    if keys["w"]:
        forward += MOVE_SPEED

    if keys["s"]:
        forward -= MOVE_SPEED

    # --------------------------------------------------------
    # Left / right
    # --------------------------------------------------------

    lateral = 0.0

    if keys["a"]:
        lateral += MOVE_SPEED

    if keys["d"]:
        lateral -= MOVE_SPEED

    # --------------------------------------------------------
    # Vertical
    # --------------------------------------------------------

    vertical = 0.0

    if keys["r"]:
        vertical += VERTICAL_SPEED

    if keys["f"]:
        vertical -= VERTICAL_SPEED

    # --------------------------------------------------------
    # Yaw
    # --------------------------------------------------------

    yaw_rate = 0.0

    if keys["q"]:
        yaw_rate += YAW_SPEED

    if keys["e"]:
        yaw_rate -= YAW_SPEED

    # --------------------------------------------------------
    # Run the flight controller (sets data.ctrl)
    # --------------------------------------------------------

    controller.step(
        np.array([forward, lateral]),
        vertical,
        yaw_rate,
        dt
    )

    # --------------------------------------------------------
    # Air drag on the body (world frame, applied at the COM).
    # This is what keeps the drone tilted during steady cruise.
    # --------------------------------------------------------

    _, vel, _, _ = controller.com_state()

    data.xfrc_applied[body_id, :3] = -DRAG_LINEAR * vel


def physics_step(dt):

    update_control(dt)

    mujoco.mj_step(model, data)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # INITIALIZE GLFW
    # ========================================================

    if not glfw.init():

        raise RuntimeError(
            "Could not initialize GLFW"
        )


    window = glfw.create_window(
        1300,
        850,
        "Skydio X2 - Body and Inertial Frames (flight dynamics)",
        None,
        None
    )

    if not window:

        glfw.terminate()

        raise RuntimeError(
            "Could not create GLFW window"
        )


    glfw.make_context_current(window)

    # Keyboard
    glfw.set_key_callback(
        window,
        key_callback
    )

    # Mouse button
    glfw.set_mouse_button_callback(
        window,
        mouse_button_callback
    )

    # Mouse movement
    glfw.set_cursor_pos_callback(
        window,
        cursor_position_callback
    )

    # Mouse wheel
    glfw.set_scroll_callback(
        window,
        scroll_callback
    )

    glfw.swap_interval(1)


    # ========================================================
    # MUJOCO VISUALIZATION
    # ========================================================

    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()

    scene = mujoco.MjvScene(
        model,
        maxgeom=10000
    )

    context = mujoco.MjrContext(
        model,
        mujoco.mjtFontScale.mjFONTSCALE_150
    )

    mujoco.mjv_defaultCamera(camera)
    mujoco.mjv_defaultOption(option)


    # ========================================================
    # WORLD / INERTIAL FRAME
    # ========================================================

    # Fixed at the initial X2 position.
    # This does NOT move with the X2.

    inertial_origin = data.xipos[body_id].copy()


    # ========================================================
    # FRAME COLORS
    # ========================================================

    # --------------------------------------------------------
    # Body frame
    # --------------------------------------------------------

    BODY_COLORS = [

        # X_B = red
        np.array([1.0, 0.0, 0.0, 1.0]),

        # Y_B = green
        np.array([0.0, 1.0, 0.0, 1.0]),

        # Z_B = blue
        np.array([0.0, 0.4, 1.0, 1.0]),
    ]


    # --------------------------------------------------------
    # Inertial/world frame
    # --------------------------------------------------------

    INERTIAL_COLORS = [

        # X_W
        np.array([1.0, 0.55, 0.55, 1.0]),

        # Y_W
        np.array([0.55, 1.0, 0.55, 1.0]),

        # Z_W
        np.array([0.55, 0.70, 1.0, 1.0]),
    ]


    # ========================================================
    # DRAW REFERENCE FRAMES
    # ========================================================

    def draw_reference_frames():

        # ----------------------------------------------------
        # Current X2 position
        # ----------------------------------------------------

        body_origin = data.xipos[body_id].copy()

        # ----------------------------------------------------
        # Body rotation matrix
        # ----------------------------------------------------

        R = data.xmat[body_id].reshape(
            3,
            3
        ).copy()

        # ----------------------------------------------------
        # World axes
        # ----------------------------------------------------

        world_axes = [

            np.array([1.0, 0.0, 0.0]),

            np.array([0.0, 1.0, 0.0]),

            np.array([0.0, 0.0, 1.0]),
        ]

        # ----------------------------------------------------
        # Reserve seven extra geometries
        #
        # 3 world axes
        # 3 body axes
        # 1 COM sphere
        # ----------------------------------------------------

        base_geom_count = scene.ngeom

        scene.ngeom = base_geom_count + 7

        identity = np.eye(3).flatten()


        # ====================================================
        # WORLD / INERTIAL FRAME
        # ====================================================

        for i in range(3):

            geom = scene.geoms[
                base_geom_count + i
            ]

            start = inertial_origin

            end = (
                inertial_origin
                + FRAME_LENGTH * world_axes[i]
            )

            mujoco.mjv_initGeom(

                geom,

                mujoco.mjtGeom.mjGEOM_ARROW,

                np.array([
                    FRAME_WIDTH,
                    FRAME_WIDTH,
                    FRAME_WIDTH
                ]),

                start,

                identity,

                INERTIAL_COLORS[i]
            )

            mujoco.mjv_connector(

                geom,

                mujoco.mjtGeom.mjGEOM_ARROW,

                FRAME_WIDTH,

                start,

                end
            )

            geom.rgba[:] = INERTIAL_COLORS[i]


        # ====================================================
        # BODY FRAME
        # ====================================================

        for i in range(3):

            geom = scene.geoms[
                base_geom_count + 3 + i
            ]

            start = body_origin

            # Body axis expressed in world coordinates
            end = (
                body_origin
                + FRAME_LENGTH * R[:, i]
            )

            mujoco.mjv_initGeom(

                geom,

                mujoco.mjtGeom.mjGEOM_ARROW,

                np.array([
                    FRAME_WIDTH,
                    FRAME_WIDTH,
                    FRAME_WIDTH
                ]),

                start,

                identity,

                BODY_COLORS[i]
            )

            mujoco.mjv_connector(

                geom,

                mujoco.mjtGeom.mjGEOM_ARROW,

                FRAME_WIDTH,

                start,

                end
            )

            geom.rgba[:] = BODY_COLORS[i]


        # ====================================================
        # X2 CENTER OF MASS
        # ====================================================

        com_geom = scene.geoms[
            base_geom_count + 6
        ]

        mujoco.mjv_initGeom(

            com_geom,

            mujoco.mjtGeom.mjGEOM_SPHERE,

            np.array([
                0.035,
                0.035,
                0.035
            ]),

            body_origin,

            identity,

            np.array([
                1.0,
                1.0,
                1.0,
                1.0
            ])
        )


    # ========================================================
    # DRAW INFORMATION
    # ========================================================

    def draw_information(viewport):

        # ----------------------------------------------------
        # Current body rotation matrix
        # ----------------------------------------------------

        R = data.xmat[body_id].reshape(
            3,
            3
        )

        # ----------------------------------------------------
        # X2 position and velocity (centre of mass)
        # ----------------------------------------------------

        pos, vel, _, _ = controller.com_state()

        speed = float(np.linalg.norm(vel[:2]))

        # ----------------------------------------------------
        # Body-frame axes in world coordinates
        # ----------------------------------------------------

        Xb = R[:, 0]
        Yb = R[:, 1]
        Zb = R[:, 2]

        # ----------------------------------------------------
        # Attitude angles
        #
        # pitch > 0 : nose DOWN   (rotation about +Y_B)
        # roll  > 0 : left side UP (rotation about +X_B)
        # ----------------------------------------------------

        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))

        tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))


        # ====================================================
        # CONTROL INFORMATION
        # ====================================================

        control_text = (
            "SKYDIO X2  (flight dynamics)\n"
            "\n"
            "W/S  Forward / Backward\n"
            "A/D  Left / Right\n"
            "Q/E  Yaw Left / Right\n"
            "R/F  Up / Down\n"
            "SPACE  Release keys (brake + hover)\n"
            "ENTER  Reset\n"
            "\n"
            "Mouse Left Drag  Camera\n"
            "Mouse Wheel  Zoom"
        )


        # ====================================================
        # FRAME INFORMATION
        # ====================================================

        frame_text = (
            "REFERENCE FRAMES\n"
            "\n"
            "BODY FRAME\n"
            "Red   = X_B\n"
            "Green = Y_B\n"
            "Blue  = Z_B\n"
            "\n"
            "WORLD / INERTIAL FRAME\n"
            "Light Red   = X_W\n"
            "Light Green = Y_W\n"
            "Light Blue  = Z_W"
        )


        # ====================================================
        # ROTATION MATRIX
        # ====================================================

        matrix_text = (
            "R(W,B)\n"
            "\n"
            f"[ {R[0,0]: .3f}  {R[0,1]: .3f}  {R[0,2]: .3f} ]\n"
            f"[ {R[1,0]: .3f}  {R[1,1]: .3f}  {R[1,2]: .3f} ]\n"
            f"[ {R[2,0]: .3f}  {R[2,1]: .3f}  {R[2,2]: .3f} ]"
        )


        # ====================================================
        # BODY AXES
        # ====================================================

        axes_text = (
            "BODY AXES IN WORLD\n"
            "\n"
            f"X_B = ({Xb[0]: .2f}, "
            f"{Xb[1]: .2f}, "
            f"{Xb[2]: .2f})\n"

            f"Y_B = ({Yb[0]: .2f}, "
            f"{Yb[1]: .2f}, "
            f"{Yb[2]: .2f})\n"

            f"Z_B = ({Zb[0]: .2f}, "
            f"{Zb[1]: .2f}, "
            f"{Zb[2]: .2f})"
        )


        # ====================================================
        # ATTITUDE
        # ====================================================

        attitude_text = (
            "ATTITUDE [deg]\n"
            f"Roll  (+ left up)  : {roll: 6.1f}\n"
            f"Pitch (+ nose down): {pitch: 6.1f}\n"
            f"Yaw                : {yaw: 6.1f}\n"
            f"Total tilt         : {tilt: 6.1f}"
        )


        # ====================================================
        # POSITION / VELOCITY
        # ====================================================

        position_text = (
            "X2 POSITION [m]\n"
            f"X: {pos[0]: .3f}\n"
            f"Y: {pos[1]: .3f}\n"
            f"Z: {pos[2]: .3f}\n"
            f"Ground speed: {speed: .2f} m/s"
        )


        # ====================================================
        # MOTORS
        # ====================================================

        f = controller.motor_thrust

        motor_text = (
            "MOTOR THRUST [N]\n"
            f"M3 {f[2]: 5.2f}   M4 {f[3]: 5.2f}   (front)\n"
            f"M2 {f[1]: 5.2f}   M1 {f[0]: 5.2f}   (rear)"
        )


        # ====================================================
        # RENDER TEXT
        # ====================================================

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            control_text,
            "",
            context
        )

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            viewport,
            frame_text,
            "",
            context
        )

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            viewport,
            matrix_text + "\n\n" + motor_text,
            "",
            context
        )

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            (
                axes_text
                + "\n\n"
                + attitude_text
                + "\n\n"
                + position_text
            ),
            "",
            context
        )


    # ========================================================
    # MAIN LOOP
    # ========================================================

    last_time = time.time()

    # Wall-clock time not yet simulated
    accumulator = 0.0


    while not glfw.window_should_close(window):

        current_time = time.time()

        frame_dt = current_time - last_time

        last_time = current_time

        # Prevent large time jumps
        frame_dt = min(frame_dt, 0.05)


        # ====================================================
        # PROCESS INPUT
        # ====================================================

        glfw.poll_events()


        # ====================================================
        # SIMULATE (real time)
        # ====================================================

        accumulator += frame_dt

        while accumulator >= model.opt.timestep:

            physics_step(model.opt.timestep)

            accumulator -= model.opt.timestep

        # Refresh kinematics for drawing
        mujoco.mj_forward(
            model,
            data
        )


        # ====================================================
        # CAMERA
        # ====================================================

        camera.lookat[:] = data.xipos[body_id]

        camera.distance = camera_distance

        camera.azimuth = camera_azimuth

        camera.elevation = camera_elevation


        # ====================================================
        # VIEWPORT
        # ====================================================

        width, height = glfw.get_framebuffer_size(
            window
        )

        viewport = mujoco.MjrRect(
            0,
            0,
            width,
            height
        )


        # ====================================================
        # UPDATE SCENE
        # ====================================================

        mujoco.mjv_updateScene(
            model,
            data,
            option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene
        )


        # ====================================================
        # DRAW REFERENCE FRAMES
        # ====================================================

        draw_reference_frames()


        # ====================================================
        # RENDER 3D
        # ====================================================

        mujoco.mjr_render(
            viewport,
            scene,
            context
        )


        # ====================================================
        # RENDER INFORMATION
        # ====================================================

        draw_information(
            viewport
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        glfw.swap_buffers(
            window
        )


        # ====================================================
        # FRAME LIMITER (in case vsync is unavailable)
        # ====================================================

        elapsed = time.time() - current_time

        if elapsed < 1.0 / 120.0:

            time.sleep(
                1.0 / 120.0 - elapsed
            )


    # ========================================================
    # CLEANUP
    # ========================================================

    glfw.destroy_window(
        window
    )

    glfw.terminate()
