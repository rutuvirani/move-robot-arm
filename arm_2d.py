import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 1. SETTINGS AND CURRENT STATE

link1_length = 1.0
link2_length = 1.0

link1_initial_angle_deg = 30.0
link2_initial_angle_deg = 45.0

target_x = 1.0
target_y = 0.5

link2_bend_sign = 1 # decides whether the elbow bends up (1) or down (-1)

tip_x_history = []
tip_y_history = []

motion_timer = None

# CALCULATIONS

def link_displacement(length, angle_deg):
    """Return a link's horizontal and vertical displacement."""
    angle_rad = np.deg2rad(angle_deg)

    dx = length * np.cos(angle_rad) # dx = l * cos(theta)
    dy = length * np.sin(angle_rad) # dy = l * sin(theta)

    return dx, dy

def forward_kinematics(
    link1_length, link2_length, link1_angle_deg, link2_angle_deg
):
    """Calculate both link tips from the joint angles."""
    link1_tip_x, link1_tip_y = link_displacement(
        link1_length, link1_angle_deg
    )

    link2_dx, link2_dy = link_displacement(
        link2_length, link1_angle_deg + link2_angle_deg  # the second link's angle is relative to the first link's angle
    )

    link2_tip_x = link1_tip_x + link2_dx
    link2_tip_y = link1_tip_y + link2_dy

    return link1_tip_x, link1_tip_y, link2_tip_x, link2_tip_y

def inverse_kinematics(
    link1_length, link2_length, target_x, target_y, link2_bend_sign
):
    """Return joint angles in degrees, or None if unreachable."""
    cos_link2_angle = (
        target_x ** 2
        + target_y ** 2
        - link1_length ** 2
        - link2_length ** 2
    ) / (2 * link1_length * link2_length) # cos(theta2) = (x^2 + y^2 - l1^2 - l2^2) / (2 * l1 * l2) paythagorean theorem


    # Allow tiny floating-point errors at the workspace boundary.
    tolerance = 1e-10

    if cos_link2_angle < -1 - tolerance:
        return None

    if cos_link2_angle > 1 + tolerance:
        return None

    cos_link2_angle = np.clip(cos_link2_angle, -1.0, 1.0)

    # At the base with equal lengths, choose one valid folded pose.
    if target_x == 0 and target_y == 0 and link1_length == link2_length:
        return 0.0, link2_bend_sign * 180.0
    
    link2_angle_rad = (link2_bend_sign * np.arccos(cos_link2_angle)) # theta2 = sign * arccos(cos(theta2))

    # theta1 = atan2(y, x) - atan2(l2 * sin(theta2), l1 + l2 * cos(theta2))
    target_direction_rad = np.arctan2(target_y, target_x)
    offset_rad = np.arctan2(
        link2_length * np.sin(link2_angle_rad),
        link1_length + link2_length * np.cos(link2_angle_rad)
    )
    link1_angle_rad = target_direction_rad - offset_rad

    link1_angle_deg = np.rad2deg(link1_angle_rad)
    link2_angle_deg = np.rad2deg(link2_angle_rad)

    # Represent the link 1 angle within the slider's range.
    link1_angle_deg = (link1_angle_deg + 180) % 360 - 180

    return link1_angle_deg, link2_angle_deg

def interpolate_angle(start_angle_deg, end_angle_deg, progress):
    angle_deg = (
        start_angle_deg
        + progress * (end_angle_deg - start_angle_deg)
    )

    return angle_deg

# EVENT CALLBACK FUNCTIONS

def update_plot(value=None, record_trail=True):
    link1_angle_deg = link1_slider.val
    link2_angle_deg = link2_slider.val

    link1_tip_x, link1_tip_y, link2_tip_x, link2_tip_y = (
        forward_kinematics(
            link1_length,
            link2_length,
            link1_angle_deg,
            link2_angle_deg
        )
    )

    arm.set_data(
        [0, link1_tip_x, link2_tip_x],
        [0, link1_tip_y, link2_tip_y]
    )

    if record_trail:
        tip_x_history.append(link2_tip_x)
        tip_y_history.append(link2_tip_y)

    trail.set_data(tip_x_history, tip_y_history)

    error_x = target_x - link2_tip_x
    error_y = target_y - link2_tip_y
    distance = np.sqrt(error_x ** 2 + error_y ** 2)

    ax.set_title(f"Distance to target: {distance:.3f} m")
    fig.canvas.draw_idle()


def clear_trail(event):
    tip_x_history.clear()
    tip_y_history.clear()

    trail.set_data([], [])
    fig.canvas.draw_idle()

def set_joint_angles(link1_angle_deg, link2_angle_deg):
    link1_slider.eventson = False
    link2_slider.eventson = False

    link1_slider.set_val(link1_angle_deg)
    link2_slider.set_val(link2_angle_deg)

    link1_slider.eventson = True
    link2_slider.eventson = True

    update_plot()

def reach_target(event):
    global motion_timer

    # stop any previous movement
    if motion_timer is not None:
        motion_timer.stop()
        motion_timer = None
        
    solution = inverse_kinematics(
        link1_length,
        link2_length,
        target_x,
        target_y,
        link2_bend_sign
    )

    if solution is None:
        ax.set_title("Target is outside the arm's reach.")
        fig.canvas.draw_idle()
        return

    link1_end_deg, link2_end_deg = solution

    # Set both angles before updating the arm.
    link1_start_deg = link1_slider.val
    link2_start_deg = link2_slider.val

    progress_steps = iter(np.linspace(0, 1, 61))

    def animate_step():
        progress = next(progress_steps, None)

        if progress is None:
            motion_timer.stop()
            return
        
        link1_angle_deg = interpolate_angle(link1_start_deg,link1_end_deg, progress)
        link2_angle_deg = interpolate_angle(link2_start_deg,link2_end_deg, progress)

        set_joint_angles(link1_angle_deg, link2_angle_deg)
    
    motion_timer = fig.canvas.new_timer(interval=30)
    motion_timer.add_callback(animate_step)
    motion_timer.start()

    update_plot()

def flip_link2(event):
    global link2_bend_sign

    link2_bend_sign = -link2_bend_sign
    reach_target(event)


def choose_target(event):
    global target_x, target_y

    if event.inaxes != ax or event.button != 1:
        return

    target_x = event.xdata
    target_y = event.ydata

    target_marker.set_data([target_x], [target_y])

    # Choosing a target does not move the arm.
    update_plot(record_trail=False)

#   THE FIGURE AND PLOTTED OBJECTS

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.30)

arm, = ax.plot(
    [], [], "o-",
    linewidth=4,
    markersize=10
)

trail, = ax.plot(
    [], [], "--",
    color="orange",
    linewidth=1
)

target_marker, = ax.plot(
    target_x, target_y, "rx",
    markersize=12,
    markeredgewidth=3
)

maximum_reach = link1_length + link2_length

ax.set_xlim(-maximum_reach - 0.2, maximum_reach + 0.2)
ax.set_ylim(-maximum_reach - 0.2, maximum_reach + 0.2)
ax.set_aspect("equal")
ax.grid(True)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

# SLIDERS AND BUTTONS

link1_slider_area = fig.add_axes([0.25, 0.15, 0.60, 0.03])
link2_slider_area = fig.add_axes([0.25, 0.07, 0.60, 0.03])

link1_slider = Slider(
    ax=link1_slider_area,
    label="Link 1 angle",
    valmin=-180,
    valmax=180,
    valinit=link1_initial_angle_deg
)

link2_slider = Slider(
    ax=link2_slider_area,
    label="Link 2 angle",
    valmin=-180,
    valmax=180,
    valinit=link2_initial_angle_deg
)

reach_area = fig.add_axes([0.25, 0.01, 0.25, 0.04])
flip_area = fig.add_axes([0.53, 0.01, 0.22, 0.04])
clear_area = fig.add_axes([0.78, 0.01, 0.18, 0.04])

reach_button = Button(reach_area, "Reach target")
flip_button = Button(flip_area, "Flip link 2")
clear_button = Button(clear_area, "Clear trail")


# CONNECT EVENTS

link1_slider.on_changed(update_plot)
link2_slider.on_changed(update_plot)

reach_button.on_clicked(reach_target)
flip_button.on_clicked(flip_link2)
clear_button.on_clicked(clear_trail)

fig.canvas.mpl_connect("button_press_event", choose_target)


# DRAW THE INITIAL STATE AND SHOW PLOT

update_plot()
plt.show()