import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from time import perf_counter
from matplotlib.patches import Circle
from heapq import heappush, heappop

# 1. SETTINGS AND CURRENT STATE

link1_length = 1.0
link2_length = 1.0

link1_initial_angle_deg = -60.0
link2_initial_angle_deg = 75.0

target_x = 1.0
target_y = 0.5

link2_bend_sign = 1 # Selects the positive (1) or negative (-1) elbow-angle solution.

tip_x_history = []
tip_y_history = []

motion_timer = None
movement_duration_s = 6.0

# speed limits of the joints in degrees per second
link1_max_speed_deg_s = 45.0
link2_max_speed_deg_s = 45.0

obstacles = [
    {"center_x": 0.8, "center_y": 0.7, "radius": 0.2},
    {"center_x": -0.6, "center_y": 0.9, "radius": 0.15}
]

# CALCULATIONS
def straight_line_points(start_x, start_y, end_x, end_y, count=61):
    '''Generate evenly spaced points along a straight line.

    Args:
        start_x, start_y (float): Starting position in meters.
        end_x, end_y (float): Ending position in meters.
        count (int): Number of samples, including both endpoints when >= 2.

    Returns:
        tuple: NumPy arrays (path_x, path_y) containing the sampled coordinates.
    '''
    progress = np.linspace(0, 1, count)

    path_x = start_x + progress * (end_x - start_x)
    path_y = start_y + progress * (end_y - start_y)

    return path_x, path_y

def link_displacement(length, angle_deg):
    '''Calculate a link's horizontal and vertical displacement.

    Args:
        length (float): Link length in meters.
        angle_deg (float): Counterclockwise angle from the positive x-axis.

    Returns:
        tuple: Displacement (dx, dy) in meters.
    '''
    angle_rad = np.deg2rad(angle_deg)

    dx = length * np.cos(angle_rad)
    dy = length * np.sin(angle_rad)

    return dx, dy

def forward_kinematics(
    link1_length, link2_length, link1_angle_deg, link2_angle_deg
):
    '''Calculate the elbow and end-effector positions from joint angles.

    Args:
        link1_length, link2_length (float): Link lengths in meters.
        link1_angle_deg (float): Link 1 angle relative to the positive x-axis.
        link2_angle_deg (float): Link 2 angle relative to link 1.

    Returns:
        tuple: (link1_tip_x, link1_tip_y, link2_tip_x, link2_tip_y)
            in meters, with the arm's base at (0, 0).
    '''
    link1_tip_x, link1_tip_y = link_displacement(
        link1_length, link1_angle_deg
    )

    link2_dx, link2_dy = link_displacement(
        link2_length, link1_angle_deg + link2_angle_deg  # Convert link 2's relative angle to a world angle.
    )

    link2_tip_x = link1_tip_x + link2_dx
    link2_tip_y = link1_tip_y + link2_dy

    return link1_tip_x, link1_tip_y, link2_tip_x, link2_tip_y

def inverse_kinematics(
    link1_length, link2_length, target_x, target_y, link2_bend_sign
):
    '''Calculate one joint-angle solution for a target position.

    Args:
        link1_length, link2_length (float): Positive link lengths in meters.
        target_x, target_y (float): Target position in meters.
        link2_bend_sign (int): Elbow-angle branch: 1 or -1.

    Returns:
        tuple or None: (link1_angle_deg, link2_angle_deg), or None if
            unreachable. Link 1's angle is normalized to [-180, 180).

    Notes:
        Checks geometric reachability only, not collisions or singularities.
    '''
    # Use the law of cosines to solve for the elbow angle.
    cos_link2_angle = (
        target_x ** 2
        + target_y ** 2
        - link1_length ** 2
        - link2_length ** 2
    ) / (2 * link1_length * link2_length)


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
    
    link2_angle_rad = (link2_bend_sign * np.arccos(cos_link2_angle))

    # Subtract the elbow-induced offset from the target direction.
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
    '''Linearly interpolate between two angles without wrapping.

    Args:
        start_angle_deg (float): Starting angle in degrees.
        end_angle_deg (float): Ending angle in degrees.
        progress (float): Interpolation fraction, normally from 0 to 1.

    Returns:
        float: Interpolated angle in degrees.

    Notes:
        Uses the direct angle difference, which may not be the shortest turn.
    '''
    angle_deg = (
        start_angle_deg
        + progress * (end_angle_deg - start_angle_deg)
    )

    return angle_deg

def smooth_progress(progress):
    '''Apply cubic easing to a movement's progress.

    Args:
        progress (float): Linear progress from 0 to 1.

    Returns:
        float: Eased progress using 3s^2 - 2s^3, with zero slope at
            both endpoints. Input values are not clamped.
    '''
    return 3 * progress ** 2 - 2 * progress ** 3

def segment_hits_circle(
    start_x, start_y, end_x, end_y, obstacle, clearance=0.06
):
    '''Check whether a line segment touches an obstacle's clearance region.

    Args:
        start_x, start_y (float): Segment start in meters.
        end_x, end_y (float): Segment end in meters.
        obstacle (dict): Circle with center_x, center_y, and radius in meters.
        clearance (float): Extra distance added to the obstacle radius.

    Returns:
        bool: True if the segment touches or enters the expanded circle.
    '''
    # Segment direction.
    dx = end_x - start_x
    dy = end_y - start_y

    # Vector from the segment start to the circle center.
    center_dx = obstacle["center_x"] - start_x
    center_dy = obstacle["center_y"] - start_y

    length_squared = dx ** 2 + dy ** 2

    # Treat a zero-length segment as a point.
    if length_squared == 0:
        closest_x = start_x
        closest_y = start_y

    else:
        # Project the circle center onto the segment's line.
        fraction = (
            center_dx * dx + center_dy * dy
        ) / length_squared

        # Clamp the projection to the actual segment endpoints.
        fraction = np.clip(fraction, 0.0, 1.0)

        closest_x = start_x + fraction * dx
        closest_y = start_y + fraction * dy

    # Distance from the circle center to the closest segment point.
    distance = np.sqrt(
        (obstacle["center_x"] - closest_x) ** 2
        + (obstacle["center_y"] - closest_y) ** 2
    )

    return distance <= obstacle["radius"] + clearance

def arm_hits_obstacles(
    link1_tip_x, link1_tip_y,
    link2_tip_x, link2_tip_y,
    obstacles
):
    '''Check both robot links against all circular obstacles.

    Args:
        link1_tip_x, link1_tip_y (float): Elbow position in meters.
        link2_tip_x, link2_tip_y (float): End-effector position in meters.
        obstacles (list): Circle dictionaries with center_x, center_y,
            and radius in meters.

    Returns:
        bool: True if either link touches an obstacle's clearance region.

    Notes:
        Uses a base at (0, 0) and segment_hits_circle's default clearance.
    '''
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
    '''Check whether the arm is nearly straight or fully folded.

    Args:
        link2_angle_deg (float): Link 2 angle relative to link 1, in degrees.
        margin_deg (float): Angular tolerance; intended range is 0 to 90.

    Returns:
        bool: True if the elbow angle is within the margin of a
            multiple of 180 degrees.
    '''
    angle_rad = np.deg2rad(link2_angle_deg)
    margin_rad = np.deg2rad(margin_deg)

    return abs(np.sin(angle_rad)) <= np.sin(margin_rad)

def is_path_valid(
    path_x, path_y, link1_length, link2_length, bend_sign, obstacles
    ):
    '''Validate sampled Cartesian positions using one elbow-angle branch.

    Args:
        path_x, path_y (sequence): Matching path coordinates in meters.
        link1_length, link2_length (float): Link lengths in meters.
        bend_sign (int): Elbow-angle branch: 1 or -1.
        obstacles (list): Circular obstacles to check against both links.

    Returns:
        bool: True if every sampled pose is reachable, outside the
            singularity margin, and clear of obstacles.

    Notes:
        Only supplied samples are checked, not the motion between them.
    '''
    
    for x,y in zip(path_x, path_y):
        solution = inverse_kinematics(
            link1_length, link2_length, x, y, bend_sign)
        
        if solution is None:
            return False
        
        _, link2_angle_deg = solution

        if is_near_singularity(link2_angle_deg):
            return False

        # Check both links, not only the end-effector position.
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

def build_collision_grid(
    link1_length, link2_length, obstacles, angle_values
):
    '''Build an obstacle-collision map for pairs of joint angles.

    Args:
        link1_length, link2_length (float): Link lengths in meters.
        obstacles (list): Circular obstacles to check against both links.
        angle_values (numpy.ndarray): Joint-angle samples in degrees.

    Returns:
        numpy.ndarray: Boolean grid of shape (N, N), where N is the
            number of angles. Rows represent link 2; columns represent
            link 1. True means collision, including clearance.

    Notes:
        Singularities are not marked as blocked unless a collision occurs.
    '''
    count = len(angle_values)
    collision_grid = np.zeros((count, count), dtype=bool)
    
    for row, link2_angle_deg in enumerate(angle_values):
        for column, link1_angle_deg in enumerate(angle_values):

            x1,y1,x2,y2 = forward_kinematics(
                link1_length, link2_length, link1_angle_deg, link2_angle_deg
            )

            collision_grid[row, column] = arm_hits_obstacles(
                x1, y1, x2, y2, obstacles
            )
    
    return collision_grid

def get_neighbors(cell, grid_shape):
    '''Return the in-bounds horizontal and vertical neighbors of a cell.

    Args:
        cell (tuple): Current (row, column).
        grid_shape (tuple): Grid size as (row_count, column_count).

    Returns:
        list: Adjacent (row, column) cells without diagonals or edge wrapping.
            Occupancy is not checked.
    '''
    row, column = cell
    row_count, column_count = grid_shape

    neighbors = []

    for row_change, column_change in [
        (-1,-1),(0,-1),(1,-1),
        (-1,0),(1,0),
        (-1,1),(0,1),(1,1)
    ]:
        new_row = row + row_change
        new_column = column + column_change

        if (
            0 <= new_row < row_count
            and 0 <= new_column < column_count
        ):
            neighbors.append((new_row, new_column))

    return neighbors

def angles_to_cell(link1_angle_deg, link2_angle_deg, angle_values):
    '''Map joint angles to the nearest sampled grid cell.

    Args:
        link1_angle_deg, link2_angle_deg (float): Joint angles in degrees.
        angle_values (numpy.ndarray): Nonempty array of grid angles.

    Returns:
        tuple: (row, column), with link 2 as the row and link 1 as the column.

    Notes:
        Uses absolute angle differences without wrapping or collision checks.
    '''
    column = int(np.argmin(np.abs(angle_values - link1_angle_deg)))
    row = int(np.argmin(np.abs(angle_values - link2_angle_deg)))

    return row, column

def is_transition_clear(
    start_cell,
    end_cell,
    angle_values,
    link1_length,
    link2_length,
    obstacles
):
    '''Sample a joint-space transition and check it for obstacle collisions.

    Args:
        start_cell, end_cell (tuple): Grid cells as (row, column).
        angle_values (numpy.ndarray): Grid angles in degrees.
        link1_length, link2_length (float): Link lengths in meters.
        obstacles (list): Circular obstacles to check against both links.

    Returns:
        bool: True if all 21 interpolated poses, including endpoints,
            are clear of obstacles.

    Notes:
        Does not check singularities or guarantee clearance between samples.
    '''
    start_row, start_column = start_cell
    end_row, end_column = end_cell

    link1_start = angle_values[start_column]
    link2_start = angle_values[start_row]

    link1_end = angle_values[end_column]
    link2_end = angle_values[end_row]

    for progress in np.linspace(0, 1, 21):
        link1_angle = (
            link1_start + progress * (link1_end - link1_start)
        )
        link2_angle = (
            link2_start + progress * (link2_end - link2_start)
        )

        x1, y1, x2, y2 = forward_kinematics(
            link1_length,
            link2_length,
            link1_angle,
            link2_angle
        )

        if arm_hits_obstacles(x1, y1, x2, y2, obstacles):
            return False

    return True

def is_joint_motion_clear(
    start_angles, end_angles,
    link1_length, link2_length, obstacles
):
    '''
    Sample a joint-angle transition and check it for obstacle collisions.
    
    Args:
        start_angles, end_angles (tuple): Joint angles in degrees.
        link1_length, link2_length (float): Link lengths in meters.
        obstacles (list): Circular obstacles to check against both links.

    Returns:
        bool: True if all sampled poses, including endpoints, are clear of obstacles.
    '''
    # Each pose contains two joint angles in degrees.
    link1_start, link2_start = start_angles
    link1_end, link2_end = end_angles

    # Sample at intervals no larger than 0.5° per joint.
    largest_change = max(
        abs(link1_end - link1_start),
        abs(link2_end - link2_start)
    )
    sample_count = max(2, int(np.ceil(largest_change / 0.5)) + 1)

    for progress in np.linspace(0, 1, sample_count):
        # Interpolate both joint angles.
        link1_angle = link1_start + progress * (link1_end - link1_start)
        link2_angle = link2_start + progress * (link2_end - link2_start)

        x1, y1, x2, y2 = forward_kinematics(
            link1_length, link2_length,
            link1_angle, link2_angle
        )

        if arm_hits_obstacles(x1, y1, x2, y2, obstacles):
            return False

    return True

def find_path(
    start_cell,
    goal_cell,
    collision_grid,
    angle_values,
    link1_length,
    link2_length,
    obstacles
):
    '''Find a joint-grid route using A* search and sampled collision checks.

    Args:
        start_cell, goal_cell (tuple): Valid grid indices as (row, column).
        collision_grid (numpy.ndarray): Boolean grid; True marks blocked cells.
        angle_values (numpy.ndarray): Grid angles in degrees.
        link1_length, link2_length (float): Link lengths in meters.
        obstacles (list): Circular obstacles used to check transitions.

    Returns:
        list or None: Route cells from start to goal, including both
            endpoints, or None if an endpoint is blocked or no route exists.

    Notes:
        Uses four-way moves with unit cost and a octile heuristic.
        Does not check singularities or wrap across grid boundaries.
    '''
    if collision_grid[start_cell] or collision_grid[goal_cell]:
        return None
    
    def estimate(cell):
        '''Return the octile-distance estimate from cell to goal_cell.'''
        return max(
            abs(cell[0] - goal_cell[0])
            , abs(cell[1] - goal_cell[1])
        ) + np.sqrt(2) * min(abs(cell[0] - goal_cell[0]), abs(cell[1] - goal_cell[1]))
    
    frontier = []
    # Heap entries: (estimated total cost, cost so far, cell).
    heappush(frontier, (estimate(start_cell), 0, start_cell))

    came_from = {}
    best_cost = {start_cell: 0}

    while frontier:
        _, current_cost, current_cell = heappop(frontier)

        # Ignore an older, more expensive entry for this cell.
        if current_cost > best_cost[current_cell]:
            continue

        if current_cell == goal_cell:
            path = [current_cell]

            # Follow parents backward, then reverse into start-to-goal order.
            while current_cell != start_cell:
                current_cell = came_from[current_cell]
                path.append(current_cell)

            path.reverse()
            return path

        for neighbor in get_neighbors(
            current_cell,
            collision_grid.shape
        ):
        
            if collision_grid[neighbor]:
                continue
            
            # Use octile distance for the cost of moving to a neighbor.
            cell_difference = current_cell[0] - neighbor[0], current_cell[1] - neighbor[1] 

            new_cost = current_cost + np.hypot(cell_difference[0], cell_difference[1])

            # Unvisited cells have infinite cost; keep only better routes.
            if new_cost >= best_cost.get(neighbor, float("inf")):
                continue
            
            # Clear endpoints alone do not ensure a clear transition.
            if not is_transition_clear(
                current_cell,
                neighbor,
                angle_values,
                link1_length,
                link2_length,
                obstacles
            ):
                continue

            best_cost[neighbor] = new_cost
            came_from[neighbor] = current_cell

            # A* priority: actual cost so far + estimated remaining cost.
            priority = new_cost + estimate(neighbor)

            heappush( frontier, (priority, new_cost, neighbor) )

    return None

def simplify_route(
    route_link1, route_link2,
    link1_length, link2_length, obstacles
):
    '''
    Shorten a joint-angle route using collision-checked shortcuts.

    Args:
        route_link1: Link 1 waypoint angles in degrees.
        route_link2: Matching link 2 waypoint angles in degrees.
        link1_length: Length of link 1 in metres.
        link2_length: Length of link 2 in metres.
        obstacles: List of circular obstacle dictionaries.

    Returns:
        Two lists containing the simplified joint-angle route.

    Notes:
        Preserves the exact first and last poses.
        Uses sampled collision checks, not a continuous guarantee.
        Does not guarantee the globally shortest route.
    '''
    if len(route_link1) != len(route_link2):
        raise ValueError("Both angle lists must have equal lengths.")

    waypoints = list(zip(route_link1, route_link2))

    if len(waypoints) < 2:
        return list(route_link1), list(route_link2)

    simplified = [waypoints[0]]
    current_index = 0

    while current_index < len(waypoints) - 1:
        next_index = None

        # Try the farthest remaining waypoint first.
        for candidate_index in range(
            len(waypoints) - 1, current_index, -1
        ):
            clear = is_joint_motion_clear(
                waypoints[current_index],
                waypoints[candidate_index],
                link1_length,
                link2_length,
                obstacles
            )

            if clear:
                next_index = candidate_index
                break

        if next_index is None:
            raise ValueError("The input route has a blocked connection.")

        # Keep the farthest waypoint we can reach directly.
        simplified.append(waypoints[next_index])
        current_index = next_index

    simplified_link1 = []
    simplified_link2 = []

    for link1_angle, link2_angle in simplified:
        simplified_link1.append(link1_angle)
        simplified_link2.append(link2_angle)

    return simplified_link1, simplified_link2

def calculate_segment_duration(
    start_pose, end_pose,
    link1_max_speed_deg_s, link2_max_speed_deg_s
):
    '''
    Calculate a shared duration that limits both joints' peak speeds.

    Args:
        start_pose: Starting (link1, link2) angles in degrees.
        end_pose: Ending (link1, link2) angles in degrees.
        link1_max_speed_deg_s: Positive link 1 speed limit.
        link2_max_speed_deg_s: Positive link 2 speed limit.

    Returns:
        Segment duration in seconds, with a minimum of 0.1 seconds.

    Notes:
        The factor 1.5 applies to cubic smoothstep: 3s² - 2s³.
        Limits planned joint speed, not acceleration.
    '''

    # Cubic smoothstep starts and ends at rest, reaching its peak speed halfway.
    # Peak speed = 1.5 * abs(angle_change) / duration. so, duration >= 1.5 * abs(angle_change) / max_speed.
    # Calculate this minimum duration for each joint, then use the larger one
    # so both joints finish together without exceeding either speed limit.
    # Angle changes are in degrees and speed limits in degrees/second, so the calculated duration is in seconds.
    if link1_max_speed_deg_s <= 0 or link2_max_speed_deg_s <= 0:
        raise ValueError("Speed limits must be positive.")

    # Calculate how far each joint rotates.
    link1_change_deg = abs(end_pose[0] - start_pose[0])
    link2_change_deg = abs(end_pose[1] - start_pose[1])

    # Allow for smoothstep's peak speed.
    link1_duration_s = (
        1.5 * link1_change_deg / link1_max_speed_deg_s
    )
    link2_duration_s = (
        1.5 * link2_change_deg / link2_max_speed_deg_s
    )

    # Give both joints enough time.
    return max(link1_duration_s, link2_duration_s, 0.1)

# EVENT CALLBACK FUNCTIONS

def update_plot(value=None, record_trail=True):
    '''Redraw the arm, collision color, trail, and target distance.

    Args:
        value (float or None): Slider callback value; unused because both
            angles are read directly from the sliders.
        record_trail (bool): Whether to append the current tip position.

    Notes:
        Updates global plot objects and, when requested, trail histories.
        Collisions change the arm's color but do not prevent movement.
    '''
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

    if link1_angle_deg is not None and link2_angle_deg is not None:
        arm_current_pose_marker.set_data(
            [link1_angle_deg],
            [link2_angle_deg]
        )

    grid_fig.canvas.draw_idle()

    error_x = target_x - link2_tip_x
    error_y = target_y - link2_tip_y
    distance = np.sqrt(error_x ** 2 + error_y ** 2)

    ax.set_title(f"Distance to target: {distance:.3f} m")
    fig.canvas.draw_idle()

def refresh_grid(target_x, target_y, link1_angle_deg=None, link2_angle_deg=None):
    '''Update the collision-map markers for both target IK solutions.

    Args:
        target_x, target_y (float): Target position in meters.

    Notes:
        Hides unreachable solutions. Markers indicate IK solutions only;
        they are not checked for collisions or singularities here.
        The collision grid itself is not rebuilt.
    '''

    solution1 = inverse_kinematics(
        link1_length,
        link2_length,
        target_x,
        target_y,
        link2_bend_sign
    )

    solution2 = inverse_kinematics(
        link1_length,
        link2_length,
        target_x,
        target_y,
        -link2_bend_sign
    )

    if solution1 is not None:
        target_marker1_coll_map.set_data(
            [solution1[0]],
            [solution1[1]]
        )
    else:
        target_marker1_coll_map.set_data([], [])

    if solution2 is not None:
        target_marker2_coll_map.set_data(
            [solution2[0]],
            [solution2[1]]
        )
    else:
        target_marker2_coll_map.set_data([], [])

    if link1_angle_deg is not None and link2_angle_deg is not None:
        arm_current_pose_marker.set_data(
            [link1_angle_deg],
            [link2_angle_deg]
        )

    grid_fig.canvas.draw_idle()

def clear_trail(event):
    '''Clear the recorded end-effector positions and redraw the empty trail.

    Args:
        event (MouseEvent): Button callback event; unused.
    '''
    tip_x_history.clear()
    tip_y_history.clear()

    trail.set_data([], [])
    fig.canvas.draw_idle()

def set_joint_angles(link1_angle_deg, link2_angle_deg):
    '''Update both joint sliders and redraw the arm once.

    Args:
        link1_angle_deg, link2_angle_deg (float): Joint angles in degrees.

    Notes:
        Temporarily disables slider callbacks to avoid drawing a partial
        update. The final pose is added to the trail by update_plot.
    '''
    link1_slider.eventson = False
    link2_slider.eventson = False

    link1_slider.set_val(link1_angle_deg)
    link2_slider.set_val(link2_angle_deg)

    link1_slider.eventson = True
    link2_slider.eventson = True

    update_plot()

def animate_route(route_link1, route_link2):
    '''
    Animate a joint-space route using smooth interpolation.
    Args:
        route_link1: Link 1 waypoint angles in degrees.
        route_link2: Matching link 2 waypoint angles in degrees.

    Returns:
        None.

    Notes:
        Uses smooth interpolation along each checked joint-space edge.
        Both lists must have equal lengths. 
        Each segment's duration is calculated from the joint speed limits.
    '''
    global motion_timer

    # Stop an earlier animation.
    if motion_timer is not None:
        motion_timer.stop()

    if len(route_link1) != len(route_link2):
        raise ValueError("Both angle lists must have equal lengths.")

    # Pair the joint angles and remove consecutive duplicates.
    waypoints = []

    for pose in zip(route_link1, route_link2):
        if not waypoints or pose != waypoints[-1]:
            waypoints.append(pose)

    if len(waypoints) < 2:
        return

    # Recheck every connection before starting.
    for start_pose, end_pose in zip(waypoints[:-1], waypoints[1:]):
        if not is_joint_motion_clear(
                start_pose, end_pose,
            link1_length, link2_length, obstacles
        ):
            ax.set_title("Route rejected: a connection collides.")
            fig.canvas.draw_idle()
            return

    # Calculate one duration for each consecutive pair of waypoints.
    segment_durations = []

    for start_pose, end_pose in zip(waypoints[:-1], waypoints[1:]):
        duration_s = calculate_segment_duration(
            start_pose,
            end_pose,
            link1_max_speed_deg_s,
            link2_max_speed_deg_s
        )
        segment_durations.append(duration_s)

    print(f"Planned movement time: {sum(segment_durations):.2f} s")

    segment_index = 0 # Current segment between waypoints. ._._._._. '_ is the current segment'
    segment_start_time = perf_counter() # Time when the current segment began.
    timer = fig.canvas.new_timer(interval=30)

    def animate_step():
        '''
        Advance one animation frame using elapsed time and joint-space interpolation.

        Args:
            None.

        Returns:
            None.

        Notes:
            Advances one frame and switches segments at each waypoint.
        '''
        nonlocal segment_index, segment_start_time

        start_pose = waypoints[segment_index]
        end_pose = waypoints[segment_index + 1] # end pose is the next waypoint because we are moving from the current waypoint to the next one

        # Calculate progress for this segment.
        elapsed_s = perf_counter() - segment_start_time
        # Use the duration belonging to this segment.
        duration_s = segment_durations[segment_index]
        progress = min(elapsed_s / duration_s, 1.0)
        smooth = smooth_progress(progress)

        link1_angle_deg = interpolate_angle(
            start_pose[0], end_pose[0], smooth
        )
        link2_angle_deg = interpolate_angle(
            start_pose[1], end_pose[1], smooth
        )

        set_joint_angles(link1_angle_deg, link2_angle_deg)

        if progress >= 1.0:
            # Advance after drawing the exact waypoint.
            segment_index += 1

            if segment_index >= len(waypoints) - 1:
                timer.stop()
            else:
                segment_start_time = perf_counter()

    timer.add_callback(animate_step)
    motion_timer = timer
    timer.start()

def reach_target(event):
    '''Validate and animate a straight end-effector path to the target.

    Args:
        event (MouseEvent): Button callback event; unused.

    Notes:
        Stops any existing animation and checks sampled path poses using
        the selected elbow branch. If accepted, starts a timer with cubic
        easing over movement_duration_s. Rejections appear in the plot title.
        This callback does not follow the A* grid route.
    '''
    global motion_timer

    # Stop any previous movement before validating a new one.
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

    # Capture the current pose as the movement's starting point.
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

    # Keep this movement's target and elbow branch fixed during animation.
    end_x = target_x
    end_y = target_y
    movement_bend_sign = link2_bend_sign

    # progress_steps = iter(np.linspace(0, 1, 61)) # Optional fixed-step animation.
    start_time = perf_counter()

    def animate_step():
        '''Advance one animation frame using elapsed time and Cartesian IK.

        Uses the start position, target, and elbow branch captured by
        reach_target. Updates the joint sliders and stops the timer when
        complete or when a position is unreachable.
        '''
        # Optional fixed-step alternative to elapsed-time progress.
        # progress = next(progress_steps, None) 
        # if progress is None: 
        #     motion_timer.stop()
        #     return

        elapsed_time = perf_counter() - start_time
        progress = min(elapsed_time / movement_duration_s, 1.0)
        smoothed_progress = smooth_progress(progress)
        
        # link1_angle_deg = interpolate_angle(link1_start_deg,link1_end_deg, smoothed_progress)
        # link2_angle_deg = interpolate_angle(link2_start_deg,link2_end_deg, smoothed_progress)

        # Interpolate Cartesian position to keep the tip on a straight line.
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

        refresh_grid(target_x, target_y, link1_angle_deg, link2_angle_deg)
        if progress >= 1.0:
            motion_timer.stop()

    motion_timer = fig.canvas.new_timer(interval=30)
    motion_timer.add_callback(animate_step)
    motion_timer.start()

    update_plot()

def flip_link2(event):
    '''Toggle the elbow-angle branch and attempt to reach the current target.

    Args:
        event (MouseEvent): Button event forwarded to reach_target.

    Notes:
        Changes the global branch even if the subsequent movement is rejected.
    '''
    global link2_bend_sign

    link2_bend_sign = -link2_bend_sign
    reach_target(event)


def choose_target(event):
    '''Set a new target from a left click inside the arm plot.

    Args:
        event (MouseEvent): Mouse event containing axes, button, and position.

    Notes:
        Updates the global target and target markers without recording a
        trail point. Does not start movement or stop an existing animation.
    '''
    global target_x, target_y

    if event.inaxes != ax or event.button != 1:
        return

    target_x = event.xdata
    target_y = event.ydata

    target_marker.set_data([target_x], [target_y])

    refresh_grid(target_x, target_y)

    # Refresh the distance display without adding a trail point.
    update_plot(record_trail=False)

def manual_joint_change(value):
    '''
    Handle a slider change by stopping any automatic movement and redrawing.
    Args:
        value: New value supplied by the changed slider.

    Returns:
        None.

    Notes:
        Manual control cancels automatic movement.
    '''
    if motion_timer is not None:
        motion_timer.stop()

    update_plot(value)

def preview_route(event):
    '''Find and display an A* route on the joint-angle collision map.

    Args:
        event (MouseEvent): Button callback event; unused.

    Notes:
        Stops the current animation and maps the current and target angles
        to their nearest grid cells. Displays the route without executing it.
        Connections from exact poses to snapped grid cells are not checked.
    '''
    # Freeze the current pose before planning.
    if motion_timer is not None:
        motion_timer.stop()

    # Remove the previous route.
    grid_route.set_data([], [])
    grid_fig.canvas.draw_idle()

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

    link1_goal_deg, link2_goal_deg = solution

    # Snap current and target angles to the discrete planning grid.
    start_cell = angles_to_cell(
        link1_slider.val, link2_slider.val, angle_values
    )
    goal_cell = angles_to_cell(
        link1_goal_deg, link2_goal_deg, angle_values
    )

    path = find_path(
        start_cell,
        goal_cell,
        collision_grid,
        angle_values,
        link1_length,
        link2_length,
        obstacles
    )

    if path is None:
        ax.set_title("No route found between the selected grid cells.")
        fig.canvas.draw_idle()
        return
    
    # Preserve the exact current and target configurations.
    exact_start = (link1_slider.val, link2_slider.val)
    exact_goal = (link1_goal_deg, link2_goal_deg)

    # Convert the route's endpoint cells into joint angles.
    start_row, start_column = path[0]
    goal_row, goal_column = path[-1]

    grid_start = (
        angle_values[start_column],
        angle_values[start_row]
    )
    grid_goal = (
        angle_values[goal_column],
        angle_values[goal_row]
    )

    # Check the short movements onto and off the grid.
    start_clear = is_joint_motion_clear(
        exact_start, grid_start,
        link1_length, link2_length, obstacles
    )

    goal_clear = is_joint_motion_clear(
        grid_goal, exact_goal,
        link1_length, link2_length, obstacles
    )

    if not start_clear or not goal_clear:
        ax.set_title("Cannot connect the exact pose to this grid route.")
        fig.canvas.draw_idle()
        return

    # Begin with the exact starting angles.
    route_link1 = [exact_start[0]]
    route_link2 = [exact_start[1]]

    # Add the grid route.
    for row, column in path:
        route_link1.append(angle_values[column])
        route_link2.append(angle_values[row])

    # Finish with the exact target angles.
    route_link1.append(exact_goal[0])
    route_link2.append(exact_goal[1])

    # Count waypoints before removing unnecessary ones.
    original_count = len(route_link1)

    route_link1, route_link2 = simplify_route(
        route_link1,
        route_link2,
        link1_length,
        link2_length,
        obstacles
    )

    print(f"Waypoints: {original_count} → {len(route_link1)}")

    grid_route.set_data(route_link1, route_link2)
    grid_ax.set_title( f"Simplified route: {len(route_link1) - 1} segments" )
    grid_fig.canvas.draw_idle()

    ax.set_title("Grid route preview ready.")
    fig.canvas.draw_idle()

    # Follow the complete route, including exact endpoints.
    animate_route(route_link1, route_link2)

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
plan_area = fig.add_axes([0.02, 0.01, 0.19, 0.04])

reach_button = Button(reach_area, "Reach target")
flip_button = Button(flip_area, "Flip link 2")
clear_button = Button(clear_area, "Clear trail")
plan_button = Button(plan_area, "Plan & Move")


# CONNECT EVENTS

link1_slider.on_changed(manual_joint_change)
link2_slider.on_changed(manual_joint_change)

reach_button.on_clicked(reach_target)
flip_button.on_clicked(flip_link2)
clear_button.on_clicked(clear_trail)
plan_button.on_clicked(preview_route)

fig.canvas.mpl_connect("button_press_event", choose_target)

# DRAW THE INITIAL STATE AND SHOW PLOT

grid_step_deg = 10
angle_values = np.arange(-180, 181, grid_step_deg)

collision_grid = build_collision_grid(
    link1_length,
    link2_length,
    obstacles,
    angle_values
)

grid_fig, grid_ax = plt.subplots()
half_step = grid_step_deg / 2

# Offset image edges so each pixel center matches its sampled joint angles.
grid_ax.imshow(
    collision_grid,
    origin="lower",
    extent=[
        angle_values[0] - half_step,
        angle_values[-1] + half_step,
        angle_values[0] - half_step,
        angle_values[-1] + half_step
    ],
    cmap="gray_r",
    vmin=0,
    vmax=1,
    interpolation="nearest"
)

grid_ax.set_xlabel("Link 1 angle ()")
grid_ax.set_ylabel("Link 2 angle ()")
grid_ax.set_title("Collision map: black = blocked, white = clear")

target_marker1_coll_map, = grid_ax.plot(
    [],[], "rx",
    markersize=12,
    markeredgewidth=3
)

target_marker2_coll_map, = grid_ax.plot(
    [],[], "rx",
    markersize=12,
    markeredgewidth=3
)

arm_current_pose_marker, = grid_ax.plot(
    [],[], "bo",
    markersize=8,
    markeredgewidth=2
)

# Reuse this line when displaying each new planned route.
grid_route, = grid_ax.plot(
    [], [], "o-",
    color="lime",
    markersize=3
)
update_plot()
plt.show()