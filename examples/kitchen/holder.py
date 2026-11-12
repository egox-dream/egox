"""Kitchen-Cleanup placement check: is the spreader released blade-first into the holder?

Evaluated every physics step in the holder frame (origin at the exterior bottom, opening
+Z) from the blade and handle vertices, holder tilt, tool velocity and the right hand's
measured aperture and contacts (pure Python, no simulator imports). Ported unchanged from
the simulator of the paper's benchmark; the holder geometry and thresholds of each trial
are frozen in its trial.json.
"""

from functools import lru_cache
import math


def footprint_below_rim(points, rim):
    """Points below `rim` plus every segment crossing it, clipped at the rim."""
    below = [p for p in points if p[2] <= rim]
    above = [p for p in points if p[2] > rim]
    footprint = list(below)
    for low in below:
        for high in above:
            fraction = (rim - low[2]) / (high[2] - low[2])
            footprint.append(tuple(low[i] + fraction * (high[i] - low[i]) for i in range(3)))
    return below, footprint


def opening_signed_distance(point, half_size, corner_radius):
    """Negative inside the (rounded-rectangle) cavity, positive inside a wall."""
    if corner_radius == 0.:
        return max(abs(point[i]) - half_size[i] for i in range(2))
    q = [abs(point[i]) - (half_size[i] - corner_radius) for i in range(2)]
    return math.hypot(max(q[0], 0.), max(q[1], 0.)) + min(max(q), 0.) - corner_radius


def convex_hull_xy(points):
    ordered = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(ordered) <= 2:
        return ordered

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    lower, upper = [], []
    for sequence, hull in ((ordered, lower), (reversed(ordered), upper)):
        for point in sequence:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 0.:
                hull.pop()
            hull.append(point)
    return lower[:-1] + upper[:-1]


@lru_cache(maxsize=16)
def outer_wall_polygon(half_size, radius):
    """Conservative 32-sided polygon circumscribing the rounded exterior wall."""
    if radius == 0.:
        return tuple((sx * half_size[0], sy * half_size[1]) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)))
    normals = [(math.cos(i * math.pi / 16), math.sin(i * math.pi / 16)) for i in range(32)]
    support = [(half_size[0] - radius) * abs(x) + (half_size[1] - radius) * abs(y) + radius for x, y in normals]
    result = []
    for i, (x, y) in enumerate(normals):
        u, v = normals[(i + 1) % len(normals)]
        determinant = x * v - y * u
        result.append(((support[i] * v - y * support[(i + 1) % len(normals)]) / determinant,
                       (x * support[(i + 1) % len(normals)] - support[i] * u) / determinant))
    return tuple(result)


def outside_wall_envelope(points, outer_size, corner_radius, tolerance):
    """True when the whole clipped convex footprint clears the exterior wall (separating axes)."""
    if not points:
        return True
    footprint = convex_hull_xy(points)
    wall = outer_wall_polygon(tuple(value / 2 for value in outer_size), corner_radius)
    for polygon in (wall, footprint):
        for i, point in enumerate(polygon):
            following = polygon[(i + 1) % len(polygon)]
            axis = (following[1] - point[1], point[0] - following[0])
            norm = math.hypot(*axis)
            if norm < 1e-15:
                continue
            axis = (axis[0] / norm, axis[1] / norm)
            a = [p[0] * axis[0] + p[1] * axis[1] for p in footprint]
            b = [p[0] * axis[0] + p[1] * axis[1] for p in wall]
            if min(a) >= max(b) - tolerance or min(b) >= max(a) - tolerance:
                return True
    return False


class HolderEvaluator:
    """Placement conditions and latched grasp/lift/success history of one episode.

    `config` holds the holder's cavity (`holder_inner_size_xy_m`, `holder_height_m`,
    `holder_bottom_thickness_m`, corner radius, physical wall envelope) and the thresholds.
    """

    def __init__(self, config):
        self.config = dict(config)
        self.reset()

    def reset(self):
        self.stable_duration_s = 0.
        self.success = self.ever_success = self.ever_grasped = self.ever_lifted = False
        self._retained_points = None
        self.state = {'success': False, 'ever_success': False, 'stable_duration_s': 0.,
                      'ever_grasped': False, 'ever_lifted': False, 'conditions': {}}

    def update(self, dt, *, blade_points_holder, handle_center_holder, holder_tilt_rad, linear_velocity_world,
               angular_velocity_world, right_grasp_command, released, hand_withdrawn, support_contact, grasped,
               lifted, handle_points_holder):
        c = self.config
        points = [tuple(map(float, p)) for p in blade_points_holder]
        handle_points = [tuple(map(float, p)) for p in handle_points_holder]
        handle = tuple(map(float, handle_center_holder))
        rim, floor, tolerance = c['holder_height_m'], c['holder_bottom_thickness_m'], c['penetration_tolerance_m']
        half = [v / 2 for v in c['holder_inner_size_xy_m']]
        radius = c['holder_inner_corner_radius_m']

        def wall_distance(point):
            return opening_signed_distance(point, half, radius)
        below, footprint = footprint_below_rim(points, rim)
        inside = bool(footprint) and all(wall_distance(p) <= tolerance for p in footprint)
        wall_top = c['holder_wall_top_z_m']
        # Resting contact can put a handle corner micrometres below the rim: apply the tolerance.
        handle_clip_z = rim if wall_top is None else wall_top - tolerance
        handle_below, handle_footprint = footprint_below_rim(handle_points, handle_clip_z)
        handle_inside = all(wall_distance(p) <= tolerance for p in handle_footprint)
        exterior_overhang = (wall_top is not None and not handle_inside
                             and outside_wall_envelope(handle_footprint, c['holder_wall_outer_size_xy_m'],
                                                       c['holder_wall_outer_corner_radius_m'], tolerance))
        handle_clear = (handle_inside or exterior_overhang) and all(p[2] >= floor - tolerance for p in handle_points)
        handle_exposed_height = max((p[2] - rim for p in handle_points), default=handle[2] - rim)
        handle_exposed = (handle_exposed_height >= c['minimum_exposed_handle_m'] if handle_points
                          else handle[2] >= rim - tolerance)
        lowest = min(p[2] for p in points)
        blade_center_z = sum(p[2] for p in points) / len(points)
        insertion_depth = max(0., rim - lowest)
        floor_gap = lowest - floor
        # Wall witnesses must lie below the rim; exterior handle overhang is not one.
        side_gaps = [-wall_distance(p) for p in (*below, *handle_below)
                     if p[2] < (rim if wall_top is None else wall_top) - tolerance]
        if wall_top is not None:
            side_gaps = [gap for gap in side_gaps if gap >= -tolerance]
        side_gap = min(side_gaps, default=math.inf)
        rim_supported = False
        if wall_top is not None and handle_clear and handle_points:
            _, near_rim = footprint_below_rim(handle_points, wall_top + c['support_proximity_m'])
            near_rim = [p for p in near_rim if p[2] >= wall_top - tolerance]
            rim_supported = (bool(near_rim) and any(wall_distance(p) >= -tolerance for p in near_rim)
                             and not outside_wall_envelope(near_rim, c['holder_wall_outer_size_xy_m'],
                                                           c['holder_wall_outer_corner_radius_m'], -tolerance))
        supported = bool(support_contact or -tolerance <= floor_gap <= c['support_proximity_m']
                         or -tolerance <= side_gap <= c['support_proximity_m'] or rim_supported)
        linear_speed = math.sqrt(sum(v * v for v in linear_velocity_world))
        angular_speed = math.sqrt(sum(v * v for v in angular_velocity_world))
        self.ever_grasped |= grasped
        self.ever_lifted |= lifted and self.ever_grasped
        conditions = {
            'grasp_observed': self.ever_grasped,
            'lift_observed': self.ever_lifted,
            'inserted': insertion_depth >= c['minimum_insertion_depth_m'],
            'inside_opening': inside,
            'handle_clear': handle_clear,
            'no_bottom_penetration': lowest >= floor - tolerance,
            'blade_first': handle_exposed and handle[2] - blade_center_z >= c['minimum_handle_above_blade_m'],
            'holder_upright': abs(holder_tilt_rad) <= c['maximum_holder_tilt_rad'],
            'supported': supported,
            'slow': linear_speed <= c['maximum_linear_speed_m_s'] and angular_speed <= c['maximum_angular_speed_rad_s'],
            'open_command': right_grasp_command >= c['minimum_open_command'],
            'released': released and not grasped,
            'hand_withdrawn': hand_withdrawn,
        }
        valid = all(conditions.values())
        drift = 0.
        if valid and c['stable_time_s'] > 0 and self._retained_points is not None:
            if len(self._retained_points) == len(points):
                drift = max(math.dist(p, initial) for p, initial in zip(points, self._retained_points))
            else:
                valid = False
            if drift > c['maximum_retained_point_drift_m']:
                valid = False
        conditions['retained'] = valid
        if valid:
            if self._retained_points is None:
                self._retained_points = points
                self.stable_duration_s = 0.
            else:
                self.stable_duration_s += dt
        else:
            self._retained_points = None
            self.stable_duration_s = 0.
        self.success = valid and self.stable_duration_s + 1e-9 >= c['stable_time_s']
        self.ever_success |= self.success
        self.state = {
            'success': self.success, 'ever_success': self.ever_success,
            'stable_duration_s': self.stable_duration_s, 'required_stable_duration_s': c['stable_time_s'],
            'ever_grasped': self.ever_grasped, 'ever_lifted': self.ever_lifted,
            'conditions': conditions, 'insertion_depth_m': insertion_depth,
            'blade_bottom_z_m': lowest, 'floor_gap_m': floor_gap,
            'handle_exposed_height_m': handle_exposed_height,
            'side_gap_m': side_gap if math.isfinite(side_gap) else None,
            'linear_speed_m_s': linear_speed, 'angular_speed_rad_s': angular_speed,
            'retained_point_drift_m': drift,
        }
        return dict(self.state)
