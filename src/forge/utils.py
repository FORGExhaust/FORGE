"""Collection of various utilities that do not nicely sit elsewhere.

Copyright 2025-2026 Chris Marsden

This file is part of FORGE.

FORGE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FORGE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FORGE.  If not, see <http://www.gnu.org/licenses/>.
"""

import math
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as mPolygon
from matplotlib.path import Path
from matplotlib.widgets import Button
from shapely.geometry import LineString, Point


def filter_distant_points(x_coords, y_coords, reference_point, critical_distance):
    """Filters points that are farther than a given critical distance from a reference point.

    Parameters
    ----------
    x_coords : np.ndarray
        Array of x coordinates.
    y_coords : np.ndarray
        Array of y coordinates.
    reference_point : tuple
        The reference point (x0, y0).
    critical_distance : float
        The minimum distance threshold.

    Returns
    -------
    filtered_x : np.ndarray
        x coordinates of filtered points.
    filtered_y : np.ndarray
        y coordinates of filtered points.
    """
    x_coords = np.asarray(x_coords)
    y_coords = np.asarray(y_coords)
    reference_point = np.asarray(reference_point)

    distances = np.sqrt((x_coords - reference_point[0])**2 + (y_coords - reference_point[1])**2)
    mask = distances > critical_distance

    return x_coords[mask], y_coords[mask]

def magnitude_scale_factors(x):
    """Returns the order of mangitude scale factor for a list of numbers.

    Parameters
    ----------
    x : list
        List of numbers.

    Returns
    -------
    List of magnitude scale factors : list

    Examples
    --------
    print(magnitude_floor(5374))           # Output: 1000
    print(magnitude_floor([5374, 0.0056])) # Output: [1000, 0.001]
    """

    def single_value(val):
        return 10 ** math.floor(math.log10(abs(val))) if val != 0 else 0

    if isinstance(x, (int, float)):
        return single_value(x)
    elif isinstance(x, list):
        return [single_value(val) for val in x]
    elif isinstance(x, np.ndarray):
        return np.asarray([single_value(val) for val in x])
    else:
        raise TypeError("Input must be a number or a list of numbers.")

def reflect_and_join_shape(points):
    """Reflects a shape about y=0 and creates a newly joined up shape.

    Constructs a new shape by:
    - Keeping only the points below y = 0
    - Reflecting those points across y = 0
    - Joining the original and reflected points in a continuous loop

    Parameters
    ----------
    points : list
        List of (x,y) points defining the shape [(x1,y1),....].

    Returns
    -------
    Reflected joined up shape:
        List of (x,y) points defining the new shape.
    """
    x_coords = [x for (x,y) in points]
    y_coords = [y for (x,y) in points]

    # Extract points below y = 0
    original_x = []
    original_y = []
    for x, y in zip(x_coords, y_coords):
        if y < 0:
            original_x.append(x)
            original_y.append(y)

    # Reflect those points across y = 0, reversing order for continuity
    reflected_x = original_x[::-1]
    reflected_y = [-y for y in original_y[::-1]]

    # Join original and reflected points
    joined_x = original_x + reflected_x
    joined_y = original_y + reflected_y

    return list(zip(joined_x,joined_y))

def interactive_shape_editor(
        x_coords,
        y_coords,
        x_min = None,
        x_max = None,
        y_min = None,
        y_max = None,
        ax = None,
        ):
    """Interactively edit a shape.

    Parameters
    ----------
    x_coords : list
        List of the x coordinates of the shape.
    y_coords : list
        List of the y coordinates of the shape.
    x_min : float
        Minimum x coordinate for plotting the shape.
    x_max : float
        Maximum x coordinate for plotting the shape.
    y_min : float
        Minimum y coordinate for plotting the shape.
    y_max : float
        Maximum y coordinate for plotting the shape.
    ax : matplotlib axes
            Axes to plot onto. If None, these will be created.
    
    Returns
    -------
    x_list : list
        List of the x coordinate of the new shape.
    y_list : list
        List of the y coordinate of the new shape.
    """

    # Remove duplicate endpoint if shape is already closed
    if len(x_coords) > 2 and x_coords[0] == x_coords[-1] and y_coords[0] == y_coords[-1]:
        x_coords = x_coords[:-1]
        y_coords = y_coords[:-1]

    points = list(zip(x_coords, y_coords))
    selected_index = None
    tolerance = 0.05
    mode = 'move'
    segment_selected = None
    history = []

    if ax is None:
        fig, ax = plt.subplots()

    fig = ax.get_figure()

    plt.subplots_adjust(bottom=0.2)
    scatter = ax.scatter(x_coords, y_coords, color='blue', picker=True)
    line_segments = []

    def draw_segments():
        nonlocal line_segments
        for seg in line_segments:
            seg.remove()
        line_segments = []
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            seg, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
            line_segments.append(seg)

    draw_segments()

    def update_plot():
        if len(points) < 2:
            return
        xs, ys = zip(*points)
        scatter.set_offsets(points)
        draw_segments()
        fig.canvas.draw_idle()

    def save_history():
        history.append(points.copy())

    def on_press(event):
        nonlocal selected_index, segment_selected, mode
        if event.inaxes != ax or len(points) < 2:
            return
        mouse_point = np.array([event.xdata, event.ydata])

        if mode == 'move':
            for i, pt in enumerate(points):
                if np.linalg.norm(mouse_point - pt) < tolerance:
                    selected_index = i
                    break
        elif mode == 'remove':
            for i, pt in enumerate(points):
                if np.linalg.norm(mouse_point - pt) < tolerance:
                    save_history()
                    del points[i]
                    update_plot()
                    mode = 'move'
                    break
        elif mode == 'add':
            if segment_selected is None:
                for i in range(len(points)):
                    p1 = np.array(points[i])
                    p2 = np.array(points[(i + 1) % len(points)])
                    line_vec = p2 - p1
                    point_vec = mouse_point - p1
                    proj = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
                    closest = p1 + proj * line_vec
                    if 0 <= proj <= 1 and np.linalg.norm(mouse_point - closest) < tolerance:
                        segment_selected = i
                        line_segments[i].set_color('r')
                        fig.canvas.draw_idle()
                        break
            else:
                save_history()
                points.insert(segment_selected + 1, (event.xdata, event.ydata))
                segment_selected = None
                update_plot()
                mode = 'move'

    def on_release(event):
        nonlocal selected_index
        selected_index = None

    def on_motion(event):
        if selected_index is None or event.inaxes != ax or mode != 'move':
            return
        save_history()
        points[selected_index] = (event.xdata, event.ydata)
        update_plot()

    def set_mode_add(event):
        nonlocal mode, segment_selected
        mode = 'add'
        segment_selected = None
        update_plot()
        print("Mode: Add Point")

    def set_mode_remove(event):
        nonlocal mode
        mode = 'remove'
        update_plot()
        print("Mode: Remove Point")

    def set_mode_move(event):
        nonlocal mode, segment_selected
        mode = 'move'
        segment_selected = None
        update_plot()
        print("Mode: Move Point")

    def print_coordinates(event):
        print("Updated coordinates:")
        closed_points = points[:]
        if closed_points[0] != closed_points[-1]:
            closed_points.append(closed_points[0])
        x_list, y_list = zip(*closed_points)
        print("x_coords =", list(x_list))
        print("y_coords =", list(y_list))

    def undo_action(event):
        nonlocal points
        if history:
            points = history.pop()
            update_plot()
            print("Undo performed.")

    def restore_symmetry(event):
        nonlocal points
        below_y_zero = [pt for pt in points if pt[1] < 0]
        if not below_y_zero:
            print("No points below y=0. Symmetry not applied.")
            return
        save_history()
        #reflected = [(x, -y) for x, y in below_y_zero]
        #points = below_y_zero + reflected
        points = reflect_and_join_shape(points)
        update_plot()
        print("Symmetry restored about y=0.")

    # Button layout
    # x (bottom left), y (bottom left), dx, dy
    N = 6
    ax_add = plt.axes([0.0, 0.01, 1.0 / N, 0.075])
    ax_remove = plt.axes([1.0 / N, 0.01, 1.0 / N, 0.075])
    ax_move = plt.axes([2.0 / N, 0.01, 1.0 / N, 0.075])
    ax_print = plt.axes([3.0 / N, 0.01, 1.0 / N, 0.075])
    ax_undo = plt.axes([4.0 / N, 0.01, 1.0 / N, 0.075])
    ax_symmetry = plt.axes([5.0 / N, 0.01, 1.0 / N, 0.075])

    btn_add = Button(ax_add, 'Add Point')
    btn_remove = Button(ax_remove, 'Remove Point')
    btn_move = Button(ax_move, 'Move Point')
    btn_print = Button(ax_print, 'Print Shape')
    btn_undo = Button(ax_undo, 'Undo')
    btn_symmetry = Button(ax_symmetry, 'Symmetry')

    btn_add.on_clicked(set_mode_add)
    btn_remove.on_clicked(set_mode_remove)
    btn_move.on_clicked(set_mode_move)
    btn_print.on_clicked(print_coordinates)
    btn_undo.on_clicked(undo_action)
    btn_symmetry.on_clicked(restore_symmetry)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min,x_max)

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min,y_max)

    ax.set_title("Interactive Shape Editor")
    ax.set_aspect('equal')
    plt.show()

    closed_points = points[:]
    if closed_points[0] != closed_points[-1]:
        closed_points.append(closed_points[0])

    x_list = [x for (x,y) in closed_points]
    y_list = [y for (x,y) in closed_points]

    return x_list, y_list


def plot_currents_comparison_bar_chart(
        x_labels,
        initial_data,
        optimised_data,
        scale_kA = True
        ):
    """Bar chart comparing coil current.

    Plots a bar chart with absolute values of initial and optimised per-turn
    coil currents, and includes a title showing the sum of squares of each
    dataset.
    
    Parameters
    ----------
    x_labels : list 
        List of coil names.
    initial_data : list
        List of initial per-turn coil current values (A).
    optimised_data : list
        List of optimised per-turn coil current values (A).
    scale_kA : bool, optional
        If True (default), display currents in kA; otherwise in MA.
    """

    if scale_kA:
        current_scale_factor = 1.0e-03
    else:
        current_scale_factor = 1.0e-06

    # Convert data to absolute values and scale
    initial_data_abs = [abs(val) * current_scale_factor for val in initial_data]
    optimised_data_abs = [abs(val) * current_scale_factor for val in optimised_data]

    # Calculate sum of squares for each dataset
    initial_sum_squares = sum(val**2 for val in initial_data_abs)
    optimised_sum_squares = sum(val**2 for val in optimised_data_abs)
    improvement_factor = initial_sum_squares / optimised_sum_squares

    # Set the positions and width for the bars
    x = np.arange(len(x_labels))
    width = 0.35

    # Create the bar chart
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, initial_data_abs, width, label='Initial', color='orange', edgecolor='black')
    ax.bar(x + width / 2, optimised_data_abs, width, label='Optimised', color='blue', edgecolor='black')

    # Add labels and legend
    if scale_kA:
        ax.set_ylabel('|I| (kA)')
    else:
        ax.set_ylabel('|I| (MA)')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    # Add title with sum of squares
    title_str = f'Sum of Squares - Initial: {initial_sum_squares:.2f}, Optimised: {optimised_sum_squares:.2f}'
    title_str += ' - Improvement factor: ' + str(np.round(improvement_factor,2))
    ax.set_title(title_str)

    plt.tight_layout()
    plt.show()

def interactive_buffered_polygon(
        x_points,
        y_points,
        ax = None,
        ):
    """Interactively create buffers around parts of a shape.

    Allows the user to interactively select edges constituting a shape, around
    which a buffer of a user-specifid thickness is created. Multiple edges
    can, in turn, have buffers created around them. The buffer thickness is
    entered by the user in the terminal each time an edge is selected.

    Parameters
    ----------
    x_points : list
        List of points constituting the x coordinates of the shape.
    y_points : list
        List of points constituting the y coordinates of the shape.

    Returns
    -------
    buffers : list
        List of shapely.geometry.LineString objects representing the buffers.
    buffer_data : list
        List of dictionaries recording each buffer definition. Each dictionary
        contains the keys ``"R"`` and ``"Z"`` (the two endpoints of the wall
        segment) and ``"distance"`` (the buffer distance). This data can be
        saved to a JSON file and later passed to the ``Optimiser`` to
        reproduce the same buffers non-interactively.
    """

    # Create initial polygon
    points = list(zip(x_points, y_points))

    # Extract line segments
    segments = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    buffers = []
    buffer_data = []

    # Interactive plot setup
    if ax is None:
        fig, ax = plt.subplots()

    fig = ax.get_figure()

    ax.set_title("Click on a line segment to select it")
    ax.plot(x_points, y_points, color='black')
    ax.set_aspect('equal')
    ax.scatter(x_points,y_points,marker='x',color='b',zorder=3)

    # Draw segments individually for interaction
    lines = []
    for seg in segments:
        line, = ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color='black', picker=5)
        lines.append(line)

    # Function to draw buffer outline during interaction
    def draw_buffer_outline(buffer_geom):
        if buffer_geom.geom_type == 'Polygon':
            bx, by = buffer_geom.exterior.xy
            ax.plot(bx, by, color='blue', linestyle='--')
        elif buffer_geom.geom_type == 'MultiPolygon':
            for poly in buffer_geom:
                bx, by = poly.exterior.xy
                ax.plot(bx, by, color='blue', linestyle='--')

    # Event handler for picking a line segment
    def on_pick(event):
        thisline = event.artist
        ind = lines.index(thisline)
        seg = segments[ind]

        # Highlight immediately
        thisline.set_color('red')
        fig.canvas.draw_idle()

        # Prompt for buffer
        buffer_distance = float(input(f"Enter buffer distance for segment {seg}: "))
        line_geom = LineString(seg)
        buffer_geom = line_geom.buffer(buffer_distance)

        # It is more convenient for us to work with LineStrings later on, than
        # it is to work with Polygons/MultiPolygons. As such, we will append
        # a LineString of the outline of the buffer_geom. This is equiavalent
        # to the boundary of the Polygon
        buffers.append(buffer_geom.boundary)

        # Record the definition data for potential saving
        buffer_data.append({
            "R": [seg[0][0], seg[1][0]],
            "Z": [seg[0][1], seg[1][1]],
            "distance": buffer_distance,
        })

        draw_buffer_outline(buffer_geom)
        fig.canvas.draw_idle()
        print(f"Buffered segment {seg} with distance {buffer_distance}")

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

    return buffers, buffer_data

def draw_shape(
    ax = None,
    remove_pixel_radius=10,
    return_closed=True,
):
    """Interactive creation of a closed polygon with Add, Remove, Move, and Finish buttons.

    Modes:
      - Add (default): click in the axes to add points (blue markers, black edges).
      - Remove       : click near a point (within remove_pixel_radius pixels) to delete it.
      - Move         : click near a point to grab it, drag to reposition, release to drop.
      - Finish       : close the window and return the points (closed lists if return_closed=True).

    If return_closed=True, the first point is repeated at the end (with sensible handling for 0/1/2 points).

    Display:
      - Solid black line connects points, always drawn closed if ≥2 points.
      - Filled gray polygon shown if ≥3 points.
      - Zoom/Pan toolbar interactions are ignored (do not add/remove/move points).

    Parameters
    ----------
    remove_pixel_radius : float
        Pixel distance threshold for selecting points (remove/move).
    return_closed : bool
        If True, returned xs/ys repeat the first vertex at the end.

    Returns
    -------
    xs : list
        List of the x coordinates of the new shape.
    ys : list
        List of the y coordinates of the new shape.
    """
    # --- Setup figure and axes layout ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
    else:
        fig = ax.figure

    plt.subplots_adjust(bottom=0.24)  # room for 4 buttons

    # --- State ---
    xs, ys = [], []
    mode = {"value": "add"}            # 'add' | 'remove' | 'move'
    dragging = {"active": False, "index": None}

    # --- Artists ---
    points_artist = ax.scatter(
        [], [], s=64, facecolors="blue", edgecolors="black", linewidths=1.0, zorder=3
    )
    line_artist, = ax.plot([], [], color="black", linewidth=1.8, zorder=2, alpha=0.3)
    poly_patch = mPolygon(np.empty((0, 2)), closed=True,
                         facecolor=(0.6, 0.6, 0.6, 0.5), edgecolor="none", zorder=1, alpha=0.3)
    ax.add_patch(poly_patch)
    poly_patch.set_visible(False)

    # --- Helpers ---
    def toolbar_mode_is_active():
        """Return True if a Matplotlib toolbar tool (Zoom/Pan) is active."""
        manager = getattr(fig.canvas, "manager", None)
        toolbar = getattr(manager, "toolbar", None)
        if toolbar is None:
            return False
        return bool(getattr(toolbar, "mode", ""))

    def update_title():
        titles = {
            "add":    "Mode: Add — Click to add points · (Zoom/Pan clicks are ignored)",
            "remove": "Mode: Remove — Click near a point to delete · (Zoom/Pan clicks are ignored)",
            "move":   "Mode: Move — Click near a point and drag to reposition · (Zoom/Pan clicks are ignored)",
        }
        ax.set_title(titles.get(mode["value"], ""))
        fig.canvas.draw_idle()

    def update_artists():
        """Refresh scatter, closed line, and filled polygon."""
        if len(xs) == 0:
            points_artist.set_offsets(np.empty((0, 2)))
            line_artist.set_data([], [])
            poly_patch.set_xy(np.empty((0, 2)))
            poly_patch.set_visible(False)
        else:
            pts = np.column_stack([xs, ys])
            points_artist.set_offsets(pts)

            # closed line
            if len(xs) >= 2:
                line_artist.set_data(xs + [xs[0]], ys + [ys[0]])
            else:
                line_artist.set_data(xs, ys)

            # fill polygon if ≥3 points
            poly_patch.set_xy(pts)
            poly_patch.set_visible(len(xs) >= 3)

        fig.canvas.draw_idle()

    def find_nearest_point_index(event):
        """
        Find index of the nearest existing point to the click (in display/pixel coords).
        Returns index if within remove_pixel_radius, else None.
        """
        if len(xs) == 0 or event.x is None or event.y is None:
            return None
        data_pts = np.column_stack([xs, ys])            # (N, 2) data
        disp_pts = ax.transData.transform(data_pts)     # (N, 2) pixels
        click_pt = np.array([event.x, event.y])         # (2,) pixels
        dists = np.linalg.norm(disp_pts - click_pt, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] <= remove_pixel_radius:
            return idx
        return None

    # --- Event handlers ---
    def on_press(event):
        # Only inside axes, left button, and toolbar not active
        if event.inaxes != ax or event.button != 1 or toolbar_mode_is_active():
            return

        if mode["value"] == "add":
            if event.xdata is None or event.ydata is None:
                return
            xs.append(float(event.xdata))
            ys.append(float(event.ydata))
            update_artists()

        elif mode["value"] == "remove":
            idx = find_nearest_point_index(event)
            if idx is not None:
                xs.pop(idx); ys.pop(idx)
                update_artists()

        elif mode["value"] == "move":
            idx = find_nearest_point_index(event)
            if idx is not None:
                dragging["active"] = True
                dragging["index"] = idx
            # If not near any point, do nothing (no drag)

    def on_motion(event):
        # Move only when dragging is active, inside axes, and toolbar not active
        if not dragging["active"]:
            return
        if event.inaxes != ax or toolbar_mode_is_active():
            return
        if event.xdata is None or event.ydata is None:
            return
        idx = dragging["index"]
        if idx is None or idx < 0 or idx >= len(xs):
            return
        xs[idx] = float(event.xdata)
        ys[idx] = float(event.ydata)
        update_artists()

    def on_release(event):
        if dragging["active"]:
            dragging["active"] = False
            dragging["index"] = None

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    # --- Buttons ---
    # Layout: Add | Remove | Move | Finish (left -> right)
    btn_ax_add    = plt.axes([0.06, 0.06, 0.18, 0.09])
    btn_ax_remove = plt.axes([0.30, 0.06, 0.18, 0.09])
    btn_ax_move   = plt.axes([0.54, 0.06, 0.18, 0.09])
    btn_ax_finish = plt.axes([0.78, 0.06, 0.16, 0.09])

    btn_add = Button(btn_ax_add,"Add",color="#d0e6ff", hovercolor="#b7d6ff")
    btn_remove = Button(btn_ax_remove,"Remove",color="#ffd9d9", hovercolor="#ffc0c0")
    btn_move = Button(btn_ax_move,"Move",color="#fff3c4", hovercolor="#ffe89a")
    btn_finish = Button(btn_ax_finish,"Finish",color="#d9ffd9", hovercolor="#c0ffc0")

    def set_add_mode(event):
        mode["value"] = "add"; update_title()

    def set_remove_mode(event):
        mode["value"] = "remove"; update_title()

    def set_move_mode(event):
        mode["value"] = "move"; update_title()

    def on_finish(event):
        plt.close(fig)

    btn_add.on_clicked(set_add_mode)
    btn_remove.on_clicked(set_remove_mode)
    btn_move.on_clicked(set_move_mode)
    btn_finish.on_clicked(on_finish)

    # Default mode: add
    set_add_mode(None)
    update_artists()

    # --- Start interactive session ---
    plt.show()

    # After window closes, prepare return (closed if requested)
    if not return_closed:
        return xs, ys
    if len(xs) == 0:
        return xs, ys
    return xs + [xs[0]], ys + [ys[0]]

def densify_closed_shape(x, y, max_dist, return_closed=True, rtol=1e-12, atol=1e-12):
    """Adds points along the boundary of a shape.

    Densify a closed polygon by inserting points along each edge so that the
    spacing between consecutive boundary points is <= max_dist.

    Parameters
    ----------
    x, y : sequence of floats
        Coordinates of the polygon vertices, ordered along the boundary.
        The input may be explicitly closed (last point equals first) or open;
        the function treats it as a closed shape in either case.
    max_dist : float
        Maximum allowed distance between consecutive points along each edge.
        Must be > 0.
    return_closed : bool, default True
        If True, the output is explicitly closed (last point equals first).
        If False, the output is open (no duplicate endpoint); wrapping is implied.
    rtol, atol : float
        Tolerances to detect a duplicate last point equal to the first.

    Returns
    -------
    Xd, Yd : np.ndarray
        Densified boundary coordinates.

    Notes
    -----
    - Original vertices are preserved.
    - For an edge of length L, we split it into N = ceil(L / max_dist) segments
      with equal spacing L/N <= max_dist.
    - Complexity: O(N_edges + N_output_points).
    """
    if max_dist <= 0:
        raise ValueError("max_dist must be positive.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of the same length.")
    if x.size < 3:
        raise ValueError("At least 3 points are required for a closed shape.")

    # Drop duplicate endpoint if input is explicitly closed
    is_closed_input = np.isclose(x[0], x[-1], rtol=rtol, atol=atol) and np.isclose(y[0], y[-1], rtol=rtol, atol=atol)
    if is_closed_input:
        x = x[:-1]
        y = y[:-1]

    n = x.size
    Xd_list = []
    Yd_list = []

    for i in range(n):
        j = (i + 1) % n
        x0, y0 = x[i], y[i]
        x1, y1 = x[j], y[j]

        dx = x1 - x0
        dy = y1 - y0
        L = np.hypot(dx, dy)

        if L == 0:
            # Degenerate edge; keep the start point only
            ts = np.array([0.0])
        else:
            # Number of equal segments so that each step <= max_dist
            segs = int(np.ceil(L / max_dist))
            segs = max(segs, 1)
            # Sample including the start (t=0) but excluding the end (t=1),
            # because the next edge will start at the end.
            ts = np.linspace(0.0, 1.0, segs, endpoint=False)

        Xd_list.append(x0 + ts * dx)
        Yd_list.append(y0 + ts * dy)

    Xd = np.concatenate(Xd_list)
    Yd = np.concatenate(Yd_list)

    if return_closed:
        # Append the first point to explicitly close the shape
        Xd = np.r_[Xd, Xd[0]]
        Yd = np.r_[Yd, Yd[0]]

    return Xd, Yd

def calc_winding_number(Br,Bz):
    """Calclates the winding number of the poloidal magnetic flux around a closed boundary.

    Takes the radial (Br) and vertical (Bz) components of the poloidal
    magnetic field at points along an arbitary closed boundary and
    calculates the resultant winding number (Poincare index). The
    data provided is assumed to correspond to points on a closed boundary
    that are packed sufficiently dense such that the maximum change in
    poloidal angle between points does not exceed pi.

    Parameters
    ----------
    Br : list
        List of radial magnetic field values at points around the boundary.
    Bz : list
        List of vertical magnetic field values at points around the boundary.

    Returns
    -------
    winding_number : float
        The resultant winding number (Poincare index).
    """

    Br2 = np.roll(Br, -1); Bz2 = np.roll(Bz, -1)
    det = Br * Bz2 - Bz * Br2
    dot = Br * Br2 + Bz * Bz2
    dtheta = np.arctan2(det, dot)
    winding_number = int((np.sum(dtheta) / (2 * np.pi)))

    return winding_number

def estimate_xpoint_location(R,Z,Br,Bz,jacobians,delta_R,delta_Z):
    """Estimates the (R,Z) location of an X-point using the derivatives of the poloidal magnetic field at nearby points.

    Takes the R,Z location of points to search for nearby X-points, along with the
    poloidal field components Br, Bz and the 2x2 matrices for the poloidal field Jacobians.
    The grid point spacings in (R,Z) are also given.

    R : list
        List of the R coordinates of points to search for nearby X-points.
    Z : list
        List of the Z coordinates of points to search for nearby X-points.
    Br : list
        List of the values of the radial magnetic field at the (R,Z) points.
    Bz : list
        List of the values of the vertical magnetic field at the (R,Z) points.
    jacobians : list
        List of the 2x2 poloidal magnetic field Jacobian matrices at the (R,Z) points.
    delta_R : float
        Spacing of grid points in R of the grid on which the equilibrium is defined.
    delta_Z : float
        Spacing of grid points in Z of the grid on which the equilibrium is defined.

    Returns
    -------
    Rx : float
        R coordinate of the located X-point. If no X-point is present, this will be NaN.
    Zx : float
        Z coordinate of the located X-point. If no X-point is present, this will be NaN.
    xpoint_present : bool
        Flag for whether or not an X-point was present.
    """

    # Initialise the X-point location and location status
    Rx = np.nan
    Zx = np.nan
    xpoint_present = False

    # Iterate over the provided points
    for R_point, Z_point, Br_point, Bz_point, jacobian in zip(R,Z,Br,Bz,jacobians):

        # Check the determinant of the Jacobian. If it is small, X-point searching
        # is unreliable
        det = np.linalg.det(jacobian)

        if np.abs(det) > 1.0e-03:

            # Poloidal field at the point
            bp = np.array([Br_point,Bz_point])

            # Use a one-shot Newton approach to estimate the X-point location
            # Calculate the displacement in R,Z between the point and the candidate X-point
            disp = -np.linalg.solve(jacobian,bp)

            dR = disp[0]
            dZ = disp[1]

            # Check if the candidate X-point location lies close to the point
            if (np.abs(dR) < delta_R) and (np.abs(dZ) < delta_Z):

                Rx = R_point + dR
                Zx = Z_point + dZ

                xpoint_present = True

                break

    return Rx, Zx, xpoint_present

def grid_points_inside_linestring(X, Y, ls: LineString, include_boundary=True):
    """Return lists of x and y coordinates for grid points inside/on a closed Shapely LineString.

    Parameters
    ----------
    X, Y : 2D numpy arrays
        Meshgrid arrays of the grid coordinates. Must have the same shape.
    ls : shapely.geometry.LineString
        Boundary curve of the region. If not closed, it will be closed automatically.
    include_boundary : bool
        If True, points on the boundary are counted as inside.

    Returns
    -------
    xs_in, ys_in : list of floats
        1D lists containing the x and y coordinates of the grid points that lie
        inside (and optionally on) the shape.
    """
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape.")

    # Ensure the linestring is closed (first == last)
    coords = list(ls.coords)
    if coords[0] != coords[-1]:
        coords.append(coords[0])

    # Build a Matplotlib Path from the closed boundary
    poly_path = Path(np.asarray(coords, dtype=float))

    # Flatten grid points to an (N, 2) array of (x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])

    # Vectorized point-in-polygon test
    # A tiny positive radius counts boundary points as inside if include_boundary=True
    radius = 1e-12 if include_boundary else 0.0
    inside = poly_path.contains_points(pts, radius=radius)  # boolean mask, shape (N,)

    # Extract the selected coordinates and return as lists
    xs_in = pts[inside, 0].tolist()
    ys_in = pts[inside, 1].tolist()
    return xs_in, ys_in

def closest_point_along_shape(x_coords,y_coords,px,py):
    """Finds the closest point along a shape to a point.

    Finds the closest point along the boundary of the supplied
    shape to a single (x,y) point provided by the user.
    """

    # Create a LineString representing the boundary
    boundary = LineString(zip(x_coords, y_coords))

    # Supplied point
    point = Point(px, py)

    # Find the closest point along the boundary
    # 1. project() gives the distance along the line to the closest point
    distance_along_boundary = boundary.project(point)

    # 2. interpolate() gives the actual point at that distance
    closest_point = boundary.interpolate(distance_along_boundary)

    closest_point_x = closest_point.x
    closest_point_y = closest_point.y

    return closest_point_x, closest_point_y

def update_figure(fig, pause=0.001):
    """Updates a matplotlib figure in such a way that visualisation in both scripts and notebooks works."""

    backend = (matplotlib.get_backend() or "").lower()

    # <-- ensure opacity every frame
    force_opaque_figure(fig, facecolour="white", text_colour="black")

    if 'inline' not in backend:
        fig.canvas.draw_idle()
        try:
            fig.canvas.flush_events()
        except Exception:
            pass
        plt.pause(pause if pause is not None else 0.001)
        return

    # Inline fallback with persistent display_id
    try:
        from IPython.display import DisplayHandle, display
        fig.canvas.draw()
        handle = globals().get("__FIG_DISPLAY_REGISTRY", {}).get(id(fig))
        if handle is None or not isinstance(handle, DisplayHandle):
            handle = display(fig, display_id=True)
            globals().setdefault("__FIG_DISPLAY_REGISTRY", {})[id(fig)] = handle
        else:
            handle.update(fig)
    except Exception:
        pass

def force_opaque_figure(fig, facecolour="white", text_colour="black"):
    """Forces a matplotlib figure to have an opaque background.
    
    This is important when updating a figure in a notebook that is in dark mode.

    Parameters
    ----------
    fig : matplotlib figure
        The matplotlib figure whose background will be opaque.
    facecolour : str
        The background colour of the figure.
    text_colour : str
        The colour of the text in the figure.
    """

    # Figure (controls the gutters/background between axes)
    fig.patch.set_facecolor(facecolour)
    fig.patch.set_alpha(1.0)

    # Axes (optional but recommended if themes/styles are changing colors)
    for ax in fig.get_axes():
        ax.set_facecolor(facecolour)
        ax.tick_params(colors=text_colour)
        ax.xaxis.label.set_color(text_colour)
        ax.yaxis.label.set_color(text_colour)
        for spine in ax.spines.values():
            spine.set_color(text_colour)


def orthogonalised_convex_hull_from_rects(
    xc: List[float],
    yc: List[float],
    dx: List[float],
    dy: List[float],
    *,
    closed: bool = True,
    elbow_rule: str = "left",
) -> Tuple[List[float], List[float]]:
    """Draws a polygon around a set of filaments of finite width.

    Filaments are given by a set of axes-aligned rectangles with
    widths/heights provided.

    Single-function version:

    1. Build convex hull of all rectangle corners (may include diagonals).
    2. Replace each diagonal hull edge with a vertical-horizontal or
       horizontal-vertical pair.  The *left* elbow rule selects the elbow
       that lies to the left of the hull edge (for CCW orientation),
       bulging outward and avoiding self-intersections.  ``'vh'`` forces
       the elbow at ``(x0, y1)``; ``'hv'`` forces it at ``(x1, y0)``.
    3. Close the polygon and remove redundant collinear points.

    Parameters
    ----------
    xc, yc, dx, dy : lists of float
        Centers and sizes of axis-aligned rectangles (no rotation).
    closed : bool
        If True (default), repeat the first point at the end.
    elbow_rule : {'left', 'vh', 'hv'}
        Strategy to choose the elbow for diagonal hull edges.
        'left' is recommended.

    Returns
    -------
    xs, ys : lists of float
        Coordinates of the resulting orthogonal polygon.
        If closed=True, the first point is repeated at the end.
    """

    # ---- Helpers nested for a single-function interface ----

    # Monotone chain convex hull: returns open ring (no duplicate endpoint)
    def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        pts = sorted(set(points))
        if len(pts) <= 1:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    # Signed area (positive if CCW); works for open ring too (wrap-around)
    def signed_area(poly: List[Tuple[float, float]]) -> float:
        if not poly:
            return 0.0
        area = 0.0
        for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
            area += x0 * y1 - x1 * y0
        return 0.5 * area

    # Remove redundant colinear vertices in a closed, axis-aligned polygon
    def simplify_colinear_closed(poly: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(poly) <= 3:
            return poly[:]
        # poly is assumed closed: first == last
        out = [poly[0]]
        for i in range(1, len(poly) - 1):
            x0, y0 = out[-1]
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]
            colinear = (x0 == x1 == x2) or (y0 == y1 == y2)
            if colinear:
                # drop middle point
                continue
            out.append((x1, y1))
        # Close
        if out[0] != out[-1]:
            out.append(out[0])
        return out

    # ---- Input checks ----
    if not (len(xc) == len(yc) == len(dx) == len(dy)):
        raise ValueError("xc, yc, dx, dy must have the same length")

    # ---- Collect rectangle corners ----
    corners: List[Tuple[float, float]] = []
    for x, y, w, h in zip(xc, yc, dx, dy):
        if w <= 0 or h <= 0:
            continue
        hx, hy = 0.5 * w, 0.5 * h
        corners.append((x - hx, y - hy))
        corners.append((x + hx, y - hy))
        corners.append((x + hx, y + hy))
        corners.append((x - hx, y + hy))

    if not corners:
        return [], []

    # ---- Step 1: convex hull (open ring) ----
    hull = convex_hull(corners)
    if len(hull) == 1:
        xs = [hull[0][0]]
        ys = [hull[0][1]]
        if closed:
            xs.append(xs[0]); ys.append(ys[0])
        return xs, ys
    if len(hull) == 2:
        # Two points hull: either axis-aligned (simple) or diagonal (create an L)
        (x0, y0), (x1, y1) = hull
        ortho = [(x0, y0)]
        if x0 == x1 or y0 == y1:
            ortho.append((x1, y1))
        else:
            e_vh = (x0, y1)
            e_hv = (x1, y0)
            if elbow_rule == "vh":
                elbow = e_vh
            elif elbow_rule == "hv":
                elbow = e_hv
            else:
                vx, vy = (x1 - x0, y1 - y0)
                cross_vh = vx * (e_vh[1] - y0) - vy * (e_vh[0] - x0)
                cross_hv = vx * (e_hv[1] - y0) - vy * (e_hv[0] - x0)
                # Prefer strictly left (>0). If both same sign, pick the larger.
                if cross_vh > 0 and cross_hv <= 0:
                    elbow = e_vh
                elif cross_hv > 0 and cross_vh <= 0:
                    elbow = e_hv
                else:
                    elbow = e_vh if cross_vh >= cross_hv else e_hv
            if ortho[-1] != elbow:
                ortho.append(elbow)
            if ortho[-1] != (x1, y1):
                ortho.append((x1, y1))
        if closed and ortho[0] != ortho[-1]:
            ortho.append(ortho[0])
        if closed:
            ortho = simplify_colinear_closed(ortho)
        xs = [p[0] for p in ortho]
        ys = [p[1] for p in ortho]
        return xs, ys

    # Ensure CCW orientation for the "left" elbow rule
    if signed_area(hull) < 0:
        hull = list(reversed(hull))

    # ---- Step 2: replace diagonals with L-shaped segments ----
    ortho: List[Tuple[float, float]] = []
    n = len(hull)
    for i in range(n):
        x0, y0 = hull[i]
        x1, y1 = hull[(i + 1) % n]

        if i == 0:
            ortho.append((x0, y0))

        if x0 == x1 or y0 == y1:
            # Already axis-aligned
            if ortho[-1] != (x1, y1):
                ortho.append((x1, y1))
            continue

        # Candidate elbows
        e_vh = (x0, y1)  # vertical then horizontal
        e_hv = (x1, y0)  # horizontal then vertical

        if elbow_rule == "vh":
            elbow = e_vh
        elif elbow_rule == "hv":
            elbow = e_hv
        else:
            # Choose the elbow strictly to the LEFT of the edge p0->p1 (for CCW hull).
            vx, vy = (x1 - x0, y1 - y0)
            cross_vh = vx * (e_vh[1] - y0) - vy * (e_vh[0] - x0)
            cross_hv = vx * (e_hv[1] - y0) - vy * (e_hv[0] - x0)
            if cross_vh > 0 and cross_hv <= 0:
                elbow = e_vh
            elif cross_hv > 0 and cross_vh <= 0:
                elbow = e_hv
            else:
                # Degenerate/colinear numeric cases: pick the one with larger cross
                elbow = e_vh if cross_vh >= cross_hv else e_hv

        if ortho[-1] != elbow:
            ortho.append(elbow)
        if ortho[-1] != (x1, y1):
            ortho.append((x1, y1))

    # ---- Step 3: close & simplify ----
    if closed and ortho[0] != ortho[-1]:
        ortho.append(ortho[0])
    if closed:
        ortho = simplify_colinear_closed(ortho)

    xs = [p[0] for p in ortho]
    ys = [p[1] for p in ortho]
    return xs, ys
