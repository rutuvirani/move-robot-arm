import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

L = 1
L2 = 1
angles = [0,45,90]
first_link_initial_angle = 30
second_link_initial_angle = 45

tip_x_history = []
tip_y_history = []

target_x = 1.0
target_y = 0.5

elbow_direction = 1

def link_tip_position(L,theta):
    x = (L * np.cos(np.deg2rad(theta)))
    y = (L * np.sin(np.deg2rad(theta)))
    return x,y

def ee_position(L1, L2, theta1, theta2):
    x1, y1 = link_tip_position(L1, theta1)

    x2, y2 = link_tip_position(L2, theta1 + theta2)

    x_ee = x1 + x2
    y_ee = y1 + y2

    return x1, y1, x_ee, y_ee

x1, y1, x_ee, y_ee = ee_position(
    L, L2, first_link_initial_angle, second_link_initial_angle
)


# Using pythagorian equation

cos_theta2 = (
    target_x ** 2 + target_y ** 2 - L ** 2 - L2 ** 2
) / (2 * L * L2)
print("Cosine of second link angle:", cos_theta2)

if -1 <= cos_theta2 <= 1:
    theta2_rad = np.arccos(cos_theta2)
    theta2_deg = np.rad2deg(theta2_rad)
    print("Calculated second link angle:", theta2_deg)

    target_direction = np.arctan2(target_y, target_x)
    offset = np.arctan2(
        L2 * np.sin(theta2_rad),
        L + L2 * np.cos(theta2_rad)
    )

    theta1_rad = target_direction - offset
    theta1_deg = np.rad2deg(theta1_rad)
    print("Calculated shoulder angle:", theta1_deg)

else:
    print("Target is outside the arm's reach.")

if -1 <= cos_theta2 <= 1:
    _, _, check_x, check_y = ee_position(
        L, L2, theta1_deg, theta2_deg
    )

    print("Calculated tip:", check_x, check_y)
    print("Target:", target_x, target_y)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.30)

def update(value):
    first_link_angle = first_link_slider.val
    second_link_angle = second_link_slider.val

    x1, y1, x_ee, y_ee = ee_position(
        L, L2, first_link_angle, second_link_angle
    )

    error_x = target_x - x_ee
    error_y = target_y - y_ee

    distance = np.sqrt(error_x ** 2 + error_y ** 2)

    arm.set_data([0, x1, x_ee], [0, y1, y_ee])
    tip_x_history.append(x_ee)
    tip_y_history.append(y_ee)

    trail.set_data(tip_x_history, tip_y_history)

    ax.set_title(f"Distance to target: {distance:.3f} m")
    fig.canvas.draw_idle()

slider_area = fig.add_axes([0.25, 0.15, 0.60, 0.03])
first_link_slider = Slider(
    ax=slider_area,
    label="Link_1 angle (°)",
    valmin=-180,
    valmax=180,
    valinit=first_link_initial_angle)

second_slider_area = fig.add_axes([0.25, 0.07, 0.60, 0.03])

second_link_slider = Slider(
    ax=second_slider_area,
    label="Link_2 angle (°)",
    valmin=-180,
    valmax=180,
    valinit=second_link_initial_angle
)

clear_area = fig.add_axes([0.78, 0.01, 0.18, 0.04])
clear_button = Button(clear_area, "Clear trail")

def clear_trail(event):
    tip_x_history.clear()
    tip_y_history.clear()

    trail.set_data([], [])
    fig.canvas.draw_idle()

clear_button.on_clicked(clear_trail)

reach_area = fig.add_axes([0.25, 0.01, 0.25, 0.04])
reach_button = Button(reach_area, "Reach target")

def reach_target(event):
    cos_theta2 = (
        target_x ** 2 + target_y ** 2 - L ** 2 - L2 ** 2
    ) / (2 * L * L2)

    if not -1 <= cos_theta2 <= 1:
        print("Target is outside the arm's reach.")
        return

    theta2_rad = elbow_direction*np.arccos(cos_theta2)

    theta1_rad = np.arctan2(target_y, target_x) - np.arctan2(
        L2 * np.sin(theta2_rad),
        L + L2 * np.cos(theta2_rad)
    )

    theta1_deg = np.rad2deg(theta1_rad)
    theta1_deg = (theta1_deg + 180) % 360 - 180

    first_link_slider.set_val(theta1_deg)
    second_link_slider.set_val(np.rad2deg(theta2_rad))

reach_button.on_clicked(reach_target)

flip_area = fig.add_axes([0.53, 0.01, 0.22, 0.04])
flip_button = Button(flip_area, "Flip elbow")


def flip_elbow(event):
    global elbow_direction

    elbow_direction = -elbow_direction
    reach_target(event)


flip_button.on_clicked(flip_elbow)

def choose_target(event):
    global target_x, target_y

    if event.inaxes != ax or event.button != 1:
        return

    target_x = event.xdata
    target_y = event.ydata

    target_marker.set_data([target_x], [target_y])
    update(None)


fig.canvas.mpl_connect("button_press_event", choose_target)

arm, = ax.plot([0, x1,x_ee], [0, y1, y_ee], "o-", linewidth=4, markersize=10)
trail, = ax.plot([], [], "--", color="orange", linewidth=1)
target_marker, = ax.plot(target_x, target_y, "rx", markersize=12, markeredgewidth=3)

reach = L + L2
ax.set_xlim(-reach - 0.2, reach + 0.2)
ax.set_ylim(-reach - 0.2, reach + 0.2)
ax.set_aspect("equal")
ax.grid(True)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

first_link_slider.on_changed(update)
second_link_slider.on_changed(update)

update(None)
plt.show()