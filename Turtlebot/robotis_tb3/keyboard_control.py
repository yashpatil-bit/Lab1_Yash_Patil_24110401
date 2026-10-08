import time
import glfw
import mujoco
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "scene_turtlebot3_waffle_pi.xml"

SPEED = 7.5
TURN_SPEED = 3.0

# Reference-frame dimensions
FRAME_LENGTH = 0.45
FRAME_WIDTH = 0.015

# Initial camera
CAMERA_DISTANCE = 2.4
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -28.0


# ============================================================
# LOAD MUJOCO MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ============================================================
# FIND WHEEL ACTUATORS
# ============================================================

left_motor = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "wheel_left"
)

right_motor = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "wheel_right"
)

print("Left actuator ID :", left_motor)
print("Right actuator ID:", right_motor)


# ============================================================
# FIND ROBOT BODY
# ============================================================

body_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "base"
)

if body_id == -1:
    raise RuntimeError("Could not find robot body 'base'.")

print("Robot body ID    :", body_id)


# ============================================================
# FIXED INERTIAL FRAME ORIGIN
# ============================================================

# Fixed reference point at the robot's initial COM.
# It does NOT move with the robot.

inertial_origin = data.xipos[body_id].copy()

print("Inertial origin  :", inertial_origin)


# ============================================================
# KEYBOARD STATE
# ============================================================

keys = {
    "up": False,
    "left": False,
    "down": False,
    "right": False,
}


# ============================================================
# MOUSE CAMERA STATE
# ============================================================

mouse_left = False
mouse_right = False

last_mouse_x = 0.0
last_mouse_y = 0.0

mouse_initialized = False

MOUSE_SENSITIVITY = 0.4
PAN_SENSITIVITY = 0.002
ZOOM_SENSITIVITY = 0.15


# ============================================================
# KEYBOARD CALLBACK
# ============================================================

def key_callback(window, key, scancode, action, mods):

    pressed = action != glfw.RELEASE

    # Forward
    if key == glfw.KEY_UP:
        keys["up"] = pressed

    # Left
    elif key == glfw.KEY_LEFT:
        keys["left"] = pressed

    # Backward
    elif key == glfw.KEY_DOWN:
        keys["down"] = pressed

    # Right
    elif key == glfw.KEY_RIGHT:
        keys["right"] = pressed

    # Stop
    elif key == glfw.KEY_SPACE and pressed:

        keys["up"] = False
        keys["left"] = False
        keys["down"] = False
        keys["right"] = False


# ============================================================
# MOUSE BUTTON CALLBACK
# ============================================================

def mouse_button_callback(window, button, action, mods):

    global mouse_left
    global mouse_right
    global last_mouse_x
    global last_mouse_y
    global mouse_initialized

    if button == glfw.MOUSE_BUTTON_LEFT:
        mouse_left = action == glfw.PRESS

    elif button == glfw.MOUSE_BUTTON_RIGHT:
        mouse_right = action == glfw.PRESS

    if action == glfw.PRESS:

        last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)

        mouse_initialized = True


# ============================================================
# MOUSE MOVEMENT CALLBACK
# ============================================================

def cursor_position_callback(window, xpos, ypos):

    global last_mouse_x
    global last_mouse_y
    global mouse_initialized

    if not mouse_initialized:

        last_mouse_x = xpos
        last_mouse_y = ypos

        mouse_initialized = True

        return

    dx = xpos - last_mouse_x
    dy = ypos - last_mouse_y

    last_mouse_x = xpos
    last_mouse_y = ypos

    # --------------------------------------------------------
    # LEFT MOUSE = ROTATE CAMERA
    # --------------------------------------------------------

    if mouse_left:

        camera.azimuth -= dx * MOUSE_SENSITIVITY

        camera.elevation += dy * MOUSE_SENSITIVITY

        # Prevent camera from flipping upside down
        camera.elevation = max(
            -89.0,
            min(89.0, camera.elevation)
        )

    # --------------------------------------------------------
    # RIGHT MOUSE = PAN CAMERA
    # --------------------------------------------------------

    elif mouse_right:

        camera.lookat[0] -= dx * PAN_SENSITIVITY
        camera.lookat[1] += dy * PAN_SENSITIVITY


# ============================================================
# MOUSE SCROLL CALLBACK
# ============================================================

def scroll_callback(window, xoffset, yoffset):

    camera.distance -= yoffset * ZOOM_SENSITIVITY

    camera.distance = max(
        0.5,
        min(10.0, camera.distance)
    )


# ============================================================
# ROBOT CONTROL
# ============================================================

def update_control():

    forward = 0.0
    turn = 0.0

    # --------------------------------------------------------
    # Forward / backward
    # --------------------------------------------------------

    if keys["up"]:
        forward += SPEED

    if keys["down"]:
        forward -= SPEED

    # --------------------------------------------------------
    # Turning
    #
    # LEFT  -> positive turn
    # RIGHT -> negative turn
    # --------------------------------------------------------

    if keys["left"]:
        turn += TURN_SPEED

    if keys["right"]:
        turn -= TURN_SPEED

    # --------------------------------------------------------
    # Differential drive
    # --------------------------------------------------------

    left = forward - turn
    right = forward + turn

    # --------------------------------------------------------
    # Respect actuator limits
    # --------------------------------------------------------

    left_min = model.actuator_ctrlrange[left_motor, 0]
    left_max = model.actuator_ctrlrange[left_motor, 1]

    right_min = model.actuator_ctrlrange[right_motor, 0]
    right_max = model.actuator_ctrlrange[right_motor, 1]

    left = max(left_min, min(left, left_max))
    right = max(right_min, min(right, right_max))

    data.ctrl[left_motor] = left
    data.ctrl[right_motor] = right


# ============================================================
# GLFW WINDOW
# ============================================================

if not glfw.init():
    raise RuntimeError("Could not initialize GLFW")

window = glfw.create_window(
    1300,
    850,
    "TurtleBot Waffle Pi - Keyboard Control",
    None,
    None
)

if not window:
    glfw.terminate()
    raise RuntimeError("Could not create GLFW window")

glfw.make_context_current(window)

# Keyboard
glfw.set_key_callback(
    window,
    key_callback
)

# Mouse buttons
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
# INITIAL CAMERA VIEW
# ============================================================

camera.lookat[:] = inertial_origin
camera.distance = CAMERA_DISTANCE
camera.azimuth = CAMERA_AZIMUTH
camera.elevation = CAMERA_ELEVATION


# ============================================================
# FRAME COLORS
# ============================================================

# BODY FRAME
# X = Red
# Y = Green
# Z = Blue

BODY_COLORS = [
    np.array([1.0, 0.0, 0.0, 1.0]),
    np.array([0.0, 1.0, 0.0, 1.0]),
    np.array([0.0, 0.4, 1.0, 1.0]),
]


# INERTIAL / WORLD FRAME
# Lighter shades

INERTIAL_COLORS = [
    np.array([1.0, 0.55, 0.55, 1.0]),
    np.array([0.55, 1.0, 0.55, 1.0]),
    np.array([0.55, 0.70, 1.0, 1.0]),
]


# ============================================================
# DRAW REFERENCE FRAMES
# ============================================================

def draw_reference_frames():

    # --------------------------------------------------------
    # Current robot COM
    # --------------------------------------------------------

    body_origin = data.xipos[body_id].copy()

    # Body rotation matrix relative to world
    R = data.xmat[body_id].reshape(3, 3).copy()

    # --------------------------------------------------------
    # Inertial-frame axes
    # --------------------------------------------------------

    inertial_axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    # --------------------------------------------------------
    # Extra geometries
    #
    # 3 inertial axes
    # 3 body axes
    # 1 COM sphere
    # --------------------------------------------------------

    base_geom_count = scene.ngeom

    scene.ngeom = base_geom_count + 7

    identity = np.eye(3).flatten()


    # ========================================================
    # INERTIAL FRAME
    # ========================================================

    for i in range(3):

        geom_index = base_geom_count + i
        geom = scene.geoms[geom_index]

        start = inertial_origin

        end = (
            inertial_origin
            + FRAME_LENGTH * inertial_axes[i]
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

        geom_index = base_geom_count + 3 + i
        geom = scene.geoms[geom_index]

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
    # CENTER OF MASS
    # ========================================================

    com_geom = scene.geoms[base_geom_count + 6]

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
# INFORMATION DISPLAY
# ============================================================

def draw_text(viewport):

    # Current robot state
    com = data.xipos[body_id]

    R = data.xmat[body_id].reshape(3, 3)

    # --------------------------------------------------------
    # LEFT PANEL
    # --------------------------------------------------------

    left_text = (
        "TURTLEBOT WAFFLE PI\n"
        "\n"
        "KEYBOARD\n"
        "UP/DOWN  Forward / Backward\n"
        "LEFT/RIGHT  Turn Left / Right\n"
        "UP + LEFT  Forward + Left\n"
        "UP + RIGHT  Forward + Right\n"
        "DOWN + LEFT  Backward + Left\n"
        "DOWN + RIGHT  Backward + Right\n"
        "SPACE  Stop"
    )


    # --------------------------------------------------------
    # UPPER RIGHT
    # --------------------------------------------------------

    right_text = (
        # "BODY FRAME\n"
        # "White = COM\n"
        # "Red = Body X\n"
        # "Green = Body Y\n"
        # "Blue = Body Z\n"
        # "\n"
        "Displacement[m]\n"
        f"X: {com[0]: .3f}\n"
        f"Y: {com[1]: .3f}\n"
        f"Z: {com[2]: .3f}"
    )


    # --------------------------------------------------------
    # BOTTOM RIGHT
    # --------------------------------------------------------

    matrix_text = (
        "R_body/world\n"
        f"[ {R[0,0]: .3f}  {R[0,1]: .3f}  {R[0,2]: .3f} ]\n"
        f"[ {R[1,0]: .3f}  {R[1,1]: .3f}  {R[1,2]: .3f} ]\n"
        f"[ {R[2,0]: .3f}  {R[2,1]: .3f}  {R[2,2]: .3f} ]"
    )


    # --------------------------------------------------------
    # TOP LEFT
    # --------------------------------------------------------

    # Keyboard information currently disabled to keep the
    # interface clean.


    # --------------------------------------------------------
    # TOP RIGHT
    # --------------------------------------------------------

    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        viewport,
        right_text,
        "",
        context
    )


    # --------------------------------------------------------
    # BOTTOM RIGHT
    # --------------------------------------------------------

    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
        viewport,
        matrix_text,
        "",
        context
    )


# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

while not glfw.window_should_close(window):

    start_time = time.time()

    # --------------------------------------------------------
    # Read keyboard + mouse
    # --------------------------------------------------------

    glfw.poll_events()

    # --------------------------------------------------------
    # Robot movement
    # --------------------------------------------------------

    update_control()

    # --------------------------------------------------------
    # Physics
    # --------------------------------------------------------

    mujoco.mj_step(
        model,
        data
    )

    # --------------------------------------------------------
    # Viewport
    # --------------------------------------------------------

    width, height = glfw.get_framebuffer_size(window)

    viewport = mujoco.MjrRect(
        0,
        0,
        width,
        height
    )

    # --------------------------------------------------------
    # Normal MuJoCo scene
    # --------------------------------------------------------

    mujoco.mjv_updateScene(
        model,
        data,
        option,
        None,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene
    )

    # --------------------------------------------------------
    # Reference frames
    # --------------------------------------------------------

    draw_reference_frames()

    # --------------------------------------------------------
    # Render 3D scene
    # --------------------------------------------------------

    mujoco.mjr_render(
        viewport,
        scene,
        context
    )

    # --------------------------------------------------------
    # Render information
    # --------------------------------------------------------

    draw_text(viewport)

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    glfw.swap_buffers(window)

    # --------------------------------------------------------
    # Real-time simulation
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    if elapsed < model.opt.timestep:

        time.sleep(
            model.opt.timestep - elapsed
        )


# ============================================================
# CLEANUP
# ============================================================

glfw.destroy_window(window)
glfw.terminate()