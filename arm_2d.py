import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from time import perf_counter
from matplotlib.patches import Circle

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
movement_duration_s = 6.0

obstacles = [
    {"center_x": 0.8, "center_y": 0.7, "radius": 0.2},
    {"center_x": -0.6, "center_y": 0.9, "radius": 0.15}
]

# CALCULATIONS
def straight_line_points(start_x, start_y, end_x, end_y, count=61):
    progress = np.linspace(0, 1, count)

    path_x = start_x + progress * (end_x - start_x)
    path_y = start_y + progress * (end_y - start_y)

    return path_x, path_y

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

def smooth_progress(progress): # we use a cubic polynomial to smooth the progress of the movement, so that it starts and ends slowly
    return 3 * progress ** 2 - 2 * progress ** 3 # 3s^2 - 2s^3

def segment_hits_circle(
    start_x, start_y, end_x, end_y, obstacle, clearance=0.06
):
    # Direction of the robot link from the start point to the end point.
    dx = end_x - start_x
    dy = end_y - start_y

    # Direction from the start of the link to the center of the obstacle.
    center_dx = obstacle["center_x"] - start_x
    center_dy = obstacle["center_y"] - start_y

    # Length of the link squared.We use squared length because it is enough for the projection calculation.
    length_squared = dx ** 2 + dy ** 2

    # Special case: if start and end are the same point, the link has zero length.
    if length_squared == 0:
        closest_x = start_x
        closest_y = start_y

    else:
        # Find where the obstacle center projects onto the link.
        # fraction = 0   -> start of the link, fraction = 0.5 -> middle of the link, fraction = 1   -> end of the link
        fraction = (
            center_dx * dx + center_dy * dy
        ) / length_squared

        # The mathematical projection can fall outside the actual link.
        fraction = np.clip(fraction, 0.0, 1.0)

        # Calculate the closest point on the link to the obstacle center.
        closest_x = start_x + fraction * dx
        closest_y = start_y + fraction * dy

    # Calculate the distance between: # obstacle center <-> closest point on the robot link
    distance = np.sqrt(
        (obstacle["center_x"] - closest_x) ** 2
        + (obstacle["center_y"] - closest_y) ** 2
    )

    # Collision happens when the distance is smaller than the obstacle radius plus a small safety clearance. 
    return distance <= obstacle["radius"] + clearance

def arm_hits_obstacles(
    link1_tip_x, link1_tip_y,
    link2_tip_x, link2_tip_y,
    obstacles
):
    for obstacle in obstacles:
        link1_collision = segment_hits_circle(
            0, 0,
            link1_tip_x, link1_tip_y,
            obstacle
        )

        link2_collision = segment_hits_circle(
            link1_tip_x, link1_tip_y,
            link2_tip_x, link2_tip_y,
            obstacle
        )

        if link1_collision or link2_collision:
            return True

    return False

def is_near_singularity(link2_angle_deg, margin_deg=5.0):
    angle_rad = np.deg2rad(link2_angle_deg)
    margin_rad = np.deg2rad(margin_deg)

    return abs(np.sin(angle_rad)) <= np.sin(margin_rad)

def is_path_valid(
    path_x, path_y, link1_length, link2_length, bend_sign, obstacles
    ):
    
    # check if the path is reachable by the arm
    for x,y in zip(path_x, path_y):
        solution = inverse_kinematics(
            link1_length, link2_length, x, y, bend_sign)
        
        if solution is None:
            return False
        
        # check if the path contains a singularity
        _, link2_angle_deg = solution

        if is_near_singularity(link2_angle_deg):
            return False

        # check if the path collides with the obstacle    
        link1_angle_deg, link2_angle_deg = solution

        link1_tip_x, link1_tip_y, link2_tip_x, link2_tip_y = (
        forward_kinematics(
            link1_length,
            link2_length,
            link1_angle_deg,
            link2_angle_deg
            )
        )
        
        if arm_hits_obstacles(
            link1_tip_x, link1_tip_y,
            link2_tip_x, link2_tip_y,
            obstacles
            ):
            return False
    
    return True

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

    collision = arm_hits_obstacles(
        link1_tip_x, link1_tip_y,
        link2_tip_x, link2_tip_y,
        obstacles
    )

    if collision:
        arm.set_color("red")
    else:
        arm.set_color("tab:blue")

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

    _, _, start_x, start_y = forward_kinematics(
    link1_length,
    link2_length,
    link1_start_deg,
    link2_start_deg
    )

    path_x, path_y = straight_line_points( start_x, start_y, target_x, target_y )

    path_is_valid = is_path_valid(
        path_x,
        path_y,
        link1_length,
        link2_length,
        link2_bend_sign,
        obstacles
    )

    if not path_is_valid:
        ax.set_title("Path rejected: unreachable or near singularity or collison.")
        fig.canvas.draw_idle()
        return

    planned_path.set_data(path_x, path_y)

    end_x = target_x
    end_y = target_y
    movement_bend_sign = link2_bend_sign

    # progress_steps = iter(np.linspace(0, 1, 61)) # uncomment in case we want to use a fixed number of steps instead of time-based animation
    start_time = perf_counter()

    def animate_step():
        # uncomment in case we want to use a fixed number of steps instead of time-based animation
        # progress = next(progress_steps, None) 
        # if progress is None: 
        #     motion_timer.stop()
        #     return

        elapsed_time = perf_counter() - start_time
        progress = min(elapsed_time / movement_duration_s, 1.0) # linear progress from 0 to 1 over the duration of the movement
        smoothed_progress = smooth_progress(progress)
        
        # link1_angle_deg = interpolate_angle(link1_start_deg,link1_end_deg, smoothed_progress)
        # link2_angle_deg = interpolate_angle(link2_start_deg,link2_end_deg, smoothed_progress)

        current_x = start_x + smoothed_progress * (end_x - start_x)
        current_y = start_y + smoothed_progress * (end_y - start_y)

        current_solution = inverse_kinematics(
            link1_length,
            link2_length,
            current_x,
            current_y,
            movement_bend_sign
        )

        if current_solution is None:
            motion_timer.stop()
            ax.set_title("Movement stopped: path contains an unreachable point.")
            fig.canvas.draw_idle()
            return

        link1_angle_deg, link2_angle_deg = current_solution

        set_joint_angles(link1_angle_deg, link2_angle_deg)
        if progress >= 1.0:
            motion_timer.stop()

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

for obstacle in obstacles:
    obstacle_patch = Circle(
        (obstacle["center_x"], obstacle["center_y"]),
        radius=obstacle["radius"],
        facecolor="gray",
        edgecolor="black",
        alpha=0.5
    )

    ax.add_patch(obstacle_patch)

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

planned_path, = ax.plot( [], [], ":", color="green", linewidth=2 )

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