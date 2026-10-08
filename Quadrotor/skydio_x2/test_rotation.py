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
# Movement
# ------------------------------------------------------------

MOVE_SPEED = 0.8
VERTICAL_SPEED = 0.6
YAW_SPEED = math.radians(60)

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


# ============================================================
# LOAD MUJOCO MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

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
    # Emergency stop
    # --------------------------------------------------------

    elif key == glfw.KEY_SPACE and pressed:

        for name in keys:
            keys[name] = False


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
# UPDATE X2 CONTROL
# ============================================================

def update_control(dt):

    # --------------------------------------------------------
    # Read current quaternion
    # --------------------------------------------------------

    qw = data.qpos[3]
    qx = data.qpos[4]
    qy = data.qpos[5]
    qz = data.qpos[6]

    # --------------------------------------------------------
    # Current yaw
    # --------------------------------------------------------

    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    )

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
    # Convert body movement to world movement
    # --------------------------------------------------------

    dx = (
        math.cos(yaw) * forward
        - math.sin(yaw) * lateral
    )

    dy = (
        math.sin(yaw) * forward
        + math.cos(yaw) * lateral
    )

    data.qpos[0] += dx * dt
    data.qpos[1] += dy * dt

    # --------------------------------------------------------
    # Vertical
    # --------------------------------------------------------

    if keys["r"]:
        data.qpos[2] += VERTICAL_SPEED * dt

    if keys["f"]:
        data.qpos[2] -= VERTICAL_SPEED * dt

    # Don't go through floor
    data.qpos[2] = max(
        data.qpos[2],
        0.12
    )

    # --------------------------------------------------------
    # Yaw
    # --------------------------------------------------------

    if keys["q"]:
        yaw += YAW_SPEED * dt

    if keys["e"]:
        yaw -= YAW_SPEED * dt

    # --------------------------------------------------------
    # Convert yaw back to quaternion
    # --------------------------------------------------------

    data.qpos[3] = math.cos(yaw / 2.0)
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = math.sin(yaw / 2.0)


# ============================================================
# INITIALIZE GLFW
# ============================================================

if not glfw.init():

    raise RuntimeError(
        "Could not initialize GLFW"
    )


window = glfw.create_window(
    1300,
    850,
    "Skydio X2 - Body and Inertial Frames",
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


# ============================================================
# MUJOCO VISUALIZATION
# ============================================================

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


# ============================================================
# WORLD / INERTIAL FRAME
# ============================================================

# Fixed at the initial X2 position.
# This does NOT move with the X2.

inertial_origin = data.xipos[body_id].copy()


# ============================================================
# FRAME COLORS
# ============================================================

# ------------------------------------------------------------
# Body frame
# ------------------------------------------------------------

BODY_COLORS = [

    # X_B = red
    np.array([1.0, 0.0, 0.0, 1.0]),

    # Y_B = green
    np.array([0.0, 1.0, 0.0, 1.0]),

    # Z_B = blue
    np.array([0.0, 0.4, 1.0, 1.0]),
]


# ------------------------------------------------------------
# Inertial/world frame
# ------------------------------------------------------------

INERTIAL_COLORS = [

    # X_W
    np.array([1.0, 0.55, 0.55, 1.0]),

    # Y_W
    np.array([0.55, 1.0, 0.55, 1.0]),

    # Z_W
    np.array([0.55, 0.70, 1.0, 1.0]),
]


# ============================================================
# DRAW REFERENCE FRAMES
# ============================================================

def draw_reference_frames():

    # --------------------------------------------------------
    # Current X2 position
    # --------------------------------------------------------

    body_origin = data.xipos[body_id].copy()

    # --------------------------------------------------------
    # Body rotation matrix
    # --------------------------------------------------------

    R = data.xmat[body_id].reshape(
        3,
        3
    ).copy()

    # --------------------------------------------------------
    # World axes
    # --------------------------------------------------------

    world_axes = [

        np.array([1.0, 0.0, 0.0]),

        np.array([0.0, 1.0, 0.0]),

        np.array([0.0, 0.0, 1.0]),
    ]

    # --------------------------------------------------------
    # Reserve seven extra geometries
    #
    # 3 world axes
    # 3 body axes
    # 1 COM sphere
    # --------------------------------------------------------

    base_geom_count = scene.ngeom

    scene.ngeom = base_geom_count + 7

    identity = np.eye(3).flatten()


    # ========================================================
    # WORLD / INERTIAL FRAME
    # ========================================================

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


    # ========================================================
    # BODY FRAME
    # ========================================================

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


    # ========================================================
    # X2 CENTER OF MASS
    # ========================================================

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


# ============================================================
# DRAW INFORMATION
# ============================================================

def draw_information(viewport):

    # --------------------------------------------------------
    # Current body rotation matrix
    # --------------------------------------------------------

    R = data.xmat[body_id].reshape(
        3,
        3
    )

    # --------------------------------------------------------
    # X2 position
    # --------------------------------------------------------

    pos = data.xipos[body_id]

    # --------------------------------------------------------
    # Body-frame axes in world coordinates
    # --------------------------------------------------------

    Xb = R[:, 0]
    Yb = R[:, 1]
    Zb = R[:, 2]


    # ========================================================
    # CONTROL INFORMATION
    # ========================================================

    control_text = (
        "SKYDIO X2\n"
        "\n"
        "W/S  Forward / Backward\n"
        "A/D  Left / Right\n"
        "Q/E  Yaw Left / Right\n"
        "R/F  Up / Down\n"
        "SPACE  Stop\n"
        "\n"
        "Mouse Left Drag  Camera\n"
        "Mouse Wheel  Zoom"
    )


    # ========================================================
    # FRAME INFORMATION
    # ========================================================

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


    # ========================================================
    # ROTATION MATRIX
    # ========================================================

    matrix_text = (
        "R(W,B)\n"
        "\n"
        f"[ {R[0,0]: .3f}  {R[0,1]: .3f}  {R[0,2]: .3f} ]\n"
        f"[ {R[1,0]: .3f}  {R[1,1]: .3f}  {R[1,2]: .3f} ]\n"
        f"[ {R[2,0]: .3f}  {R[2,1]: .3f}  {R[2,2]: .3f} ]"
    )


    # ========================================================
    # BODY AXES
    # ========================================================

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


    # ========================================================
    # POSITION
    # ========================================================

    position_text = (
        "X2 POSITION [m]\n"
        f"X: {pos[0]: .3f}\n"
        f"Y: {pos[1]: .3f}\n"
        f"Z: {pos[2]: .3f}"
    )


    # ========================================================
    # RENDER TEXT
    # ========================================================

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
        matrix_text,
        "",
        context
    )

    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        viewport,
        axes_text + "\n\n" + position_text,
        "",
        context
    )


# ============================================================
# MAIN LOOP
# ============================================================

last_time = time.time()


while not glfw.window_should_close(window):

    current_time = time.time()

    dt = current_time - last_time

    last_time = current_time

    # Prevent large time jumps
    dt = min(dt, 0.05)


    # ========================================================
    # PROCESS INPUT
    # ========================================================

    glfw.poll_events()


    # ========================================================
    # UPDATE X2
    # ========================================================

    update_control(dt)

    mujoco.mj_forward(
        model,
        data
    )


    # ========================================================
    # CAMERA
    # ========================================================

    camera.lookat[:] = data.xipos[body_id]

    camera.distance = camera_distance

    camera.azimuth = camera_azimuth

    camera.elevation = camera_elevation


    # ========================================================
    # VIEWPORT
    # ========================================================

    width, height = glfw.get_framebuffer_size(
        window
    )

    viewport = mujoco.MjrRect(
        0,
        0,
        width,
        height
    )


    # ========================================================
    # UPDATE SCENE
    # ========================================================

    mujoco.mjv_updateScene(
        model,
        data,
        option,
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene
    )


    # ========================================================
    # DRAW REFERENCE FRAMES
    # ========================================================

    draw_reference_frames()


    # ========================================================
    # RENDER 3D
    # ========================================================

    mujoco.mjr_render(
        viewport,
        scene,
        context
    )


    # ========================================================
    # RENDER INFORMATION
    # ========================================================

    draw_information(
        viewport
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    glfw.swap_buffers(
        window
    )


    # ========================================================
    # REAL-TIME TIMING
    # ========================================================

    elapsed = time.time() - current_time

    if elapsed < model.opt.timestep:

        time.sleep(
            model.opt.timestep - elapsed
        )


# ============================================================
# CLEANUP
# ============================================================

glfw.destroy_window(
    window
)

glfw.terminate()