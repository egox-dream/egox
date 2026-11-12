"""Kitchen-Bread coating: condensed milk loaded onto the spreader blade in the can and
transferred to the bread by real blade/bread contacts (NumPy only, no simulator imports).

A visual transfer model, not a fluid: it applies no forces. Thickness grids live in the
rigid-body frames of the blade ([face (+Z, -Z), local Y, local X]) and the bread top
([local Y, local X]). `bread_coverage` gives the task's success measure. Ported unchanged
from the simulator of the paper's benchmark; the parameters of each trial are frozen in
its trial.json.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CoatingConfig:
    blade_resolution: tuple = (32, 96)
    bread_resolution: tuple = (128, 128)
    can_inner_radius_m: float = .03966
    can_mouth_z_m: float = .100
    can_fill_z_m: float = .070
    can_bottom_z_m: float = .00375
    immersion_dwell_s: float = .100
    loading_rate_m_s: float = .010
    blade_max_thickness_m: float = .0005
    bread_max_thickness_m: float = .001
    transfer_rate_s: float = 6.
    surface_tolerance_m: float = .002
    contact_brush_radius_m: float = .010
    min_downward_face_cos: float = .25
    max_step_distance_m: float = .050
    max_sweep_distance_m: float = .030
    max_sweep_samples: int = 16

    def __post_init__(self):
        object.__setattr__(self, 'blade_resolution', tuple(self.blade_resolution))
        object.__setattr__(self, 'bread_resolution', tuple(self.bread_resolution))


def blade_surface_samples(xy, bounds, profile=None):
    """Positions and unit normals of the +Z/-Z blade faces, shape (2, ..., 3).

    The distal tip is local -X; an optional underside bevel slopes the -Z face there.
    """
    xy = np.asarray(xy, dtype=np.float64)
    points = np.empty((2, *xy.shape[:-1], 3))
    points[..., :2] = xy
    points[0, ..., 2], points[1, ..., 2] = bounds[1, 2], bounds[0, 2]
    normals = np.zeros_like(points)
    normals[0, ..., 2], normals[1, ..., 2] = 1., -1.
    if profile is not None:
        rise = bounds[1, 2] - bounds[0, 2] - profile['tip_thickness_m']
        points[1, ..., 2] += rise * np.clip(1. - (xy[..., 0] - bounds[0, 0]) / profile['bevel_length_m'], 0., 1.)
        slope = -rise / profile['bevel_length_m']
        beveled = xy[..., 0] < bounds[0, 0] + profile['bevel_length_m']
        normals[1, ..., 0] = np.where(beveled, slope / math.sqrt(1 + slope * slope), 0.)
        normals[1, ..., 2] = np.where(beveled, -1 / math.sqrt(1 + slope * slope), -1.)
    return points, normals


def inside_polygon(points, polygon, include_boundary=True, boundary_tolerance=0.):
    inside = np.zeros(len(points), dtype=bool)
    boundary = np.zeros(len(points), dtype=bool)
    if len(polygon) < 3:
        return inside
    x, y = points[:, 0], points[:, 1]
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if include_boundary:
            edge, tolerance = b - a, 1e-12 + boundary_tolerance
            boundary |= ((np.abs((x - a[0]) * edge[1] - (y - a[1]) * edge[0]) <= tolerance * max(np.linalg.norm(edge), 1e-12))
                         & (x >= min(a[0], b[0]) - tolerance) & (x <= max(a[0], b[0]) + tolerance)
                         & (y >= min(a[1], b[1]) - tolerance) & (y <= max(a[1], b[1]) + tolerance))
        if abs(b[1] - a[1]) > 1e-15:
            inside ^= ((a[1] > y) != (b[1] > y)) & (x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0])
    return inside | boundary


def pose_matrix(pose):
    """XYZ + XYZW quaternion -> (translation, rotation matrix)."""
    pose = np.asarray(pose, dtype=np.float64)
    x, y, z, w = pose[3:]
    return pose[:3], np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                               [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                               [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class CoatingModel:
    """Blade and bread coating grids with a conservative, contact-gated transfer."""

    def __init__(self, config, blade_bounds_xyz, bread_outline_xy, bread_top_z, bread_contact_outline_xy,
                 blade_profile=None):
        self.config = config if isinstance(config, CoatingConfig) else CoatingConfig(**config)
        cfg = self.config
        self.blade_bounds_xyz = np.array(blade_bounds_xyz, dtype=np.float64)
        self.blade_profile = None if blade_profile is None else dict(blade_profile)
        self.bread_outline_xy = outline = np.array(bread_outline_xy, dtype=np.float64)
        self.bread_contact_outline_xy = np.array(bread_contact_outline_xy, dtype=np.float64)
        self.bread_bounds_xy = np.array([outline.min(0), outline.max(0)])
        self.blade_uv_bounds_xy = self.blade_bounds_xyz[:, :2].copy()
        self.bread_top_z = float(bread_top_z)
        by, bx = cfg.blade_resolution
        ry, rx = cfg.bread_resolution
        self.blade_cell_size_xy = (self.blade_uv_bounds_xy[1] - self.blade_uv_bounds_xy[0]) / [bx, by]
        self.bread_cell_size_xy = (self.bread_bounds_xy[1] - self.bread_bounds_xy[0]) / [rx, ry]
        self.blade_cell_area_m2 = float(np.prod(self.blade_cell_size_xy))
        self.bread_cell_area_m2 = float(np.prod(self.bread_cell_size_xy))
        self._blade_center = self.blade_bounds_xyz.mean(0)
        self._blade_radius = float(np.linalg.norm(self.blade_bounds_xyz[1] - self.blade_bounds_xyz[0]) / 2)

        def centers(bounds, shape):
            ny, nx = shape
            xx = bounds[0, 0] + (np.arange(nx) + .5) * (bounds[1, 0] - bounds[0, 0]) / nx
            yy = bounds[0, 1] + (np.arange(ny) + .5) * (bounds[1, 1] - bounds[0, 1]) / ny
            return np.stack(np.meshgrid(xx, yy), axis=-1)
        points, normals = blade_surface_samples(centers(self.blade_uv_bounds_xy, cfg.blade_resolution),
                                                self.blade_bounds_xyz, self.blade_profile)
        self._blade_points, self._blade_normals = points.reshape(-1, 3), normals.reshape(-1, 3)
        # Thickness is normal to each face: integrate true surface area, including the bevel.
        areas = np.full((2, by, bx), self.blade_cell_area_m2)
        if self.blade_profile is not None:
            bevel = self.blade_profile['bevel_length_m']
            rise = np.diff(self.blade_bounds_xyz[:, 2])[0] - self.blade_profile['tip_thickness_m']
            overlap = np.clip((bevel - np.arange(bx) * self.blade_cell_size_xy[0]) / self.blade_cell_size_xy[0], 0., 1.)
            areas[1] *= 1. + overlap * (math.sqrt(1. + (rise / bevel) ** 2) - 1.)
        self.blade_cell_areas_m2 = areas
        self.bread_mask = inside_polygon(centers(self.bread_bounds_xy, cfg.bread_resolution).reshape(-1, 2),
                                         outline).reshape(ry, rx)
        # A soft tent brush from two box blurs per axis (prefix sums).
        self._brush_radii_xy = np.maximum(1, np.ceil(cfg.contact_brush_radius_m / (2 * self.bread_cell_size_xy)).astype(int))
        self._brush_normalizer = self._blur(self.bread_mask.astype(float))
        self.blade_thickness_m = np.zeros((2, by, bx))
        self.bread_thickness_m = np.zeros((ry, rx))
        self.version = 0
        self.reset()

    def reset(self):
        self.blade_thickness_m.fill(0.)
        self.bread_thickness_m.fill(0.)
        self._entered = np.zeros(len(self._blade_points), dtype=bool)
        self._dwell = np.zeros(len(self._blade_points))
        self._previous_can_points = self._previous_bread_points = None
        self._dip_counted = False
        self._loaded_m3 = self._transferred_m3 = self._last_transfer_m3 = 0.
        self._dip_count = 0
        self.version += 1

    def _load(self, dt, points):
        """Load blade cells immersed below the fill line after entering through the mouth."""
        cfg = self.config
        inside = ((np.linalg.norm(points[:, :2], axis=1) < cfg.can_inner_radius_m)
                  & (points[:, 2] < cfg.can_mouth_z_m) & (points[:, 2] >= cfg.can_bottom_z_m))
        previous = self._previous_can_points
        self._entered &= inside
        if previous is not None:
            crossing = ((previous[:, 2] >= cfg.can_mouth_z_m) & (points[:, 2] < cfg.can_mouth_z_m)
                        & (np.linalg.norm(points - previous, axis=1) <= cfg.max_step_distance_m))
            ids = np.flatnonzero(crossing)
            if len(ids):
                alpha = (previous[ids, 2] - cfg.can_mouth_z_m) / (previous[ids, 2] - points[ids, 2])
                rim_xy = previous[ids, :2] + alpha[:, None] * (points[ids, :2] - previous[ids, :2])
                self._entered[ids] = np.linalg.norm(rim_xy, axis=1) < cfg.can_inner_radius_m
                self._entered &= inside
        if not self._entered.any():
            self._dip_counted = False
        submerged = self._entered & (points[:, 2] <= cfg.can_fill_z_m)
        self._dwell = np.where(submerged, self._dwell + dt, 0.)
        eligible = submerged & (self._dwell >= cfg.immersion_dwell_s - 1e-12)
        if eligible.any() and not self._dip_counted:
            self._dip_count += 1
            self._dip_counted = True
        flat = self.blade_thickness_m.reshape(-1)
        increments = np.minimum(np.maximum(cfg.blade_max_thickness_m - flat, 0.), cfg.loading_rate_m_s * dt)
        increments[~eligible] = 0.
        flat += increments
        volume = (float(increments.sum() * self.blade_cell_area_m2) if self.blade_profile is None
                  else float(increments @ self.blade_cell_areas_m2.reshape(-1)))
        self._loaded_m3 += volume
        self._previous_can_points = points.copy()
        return volume

    def _blur(self, values):
        result = np.asarray(values, dtype=float)
        for axis, radius in ((1, self._brush_radii_xy[0]), (0, self._brush_radii_xy[1])):
            width = 2 * int(radius) + 1
            for _ in range(2):
                padding = [(0, 0), (0, 0)]
                padding[axis] = (int(radius), int(radius))
                padded = np.pad(result, padding)
                zero_shape = list(padded.shape)
                zero_shape[axis] = 1
                prefix = np.concatenate((np.zeros(zero_shape), np.cumsum(padded, axis=axis)), axis=axis)
                start, end = [slice(None)] * 2, [slice(None)] * 2
                start[axis], end[axis] = slice(width, None), slice(None, -width)
                result = (prefix[tuple(start)] - prefix[tuple(end)]) / width
        return np.maximum(result, 0.)

    def _deposit(self, source_ids, xy, fraction):
        """Soft normalized footprint; bread capacity limits are fed back to the blade cells."""
        if not len(source_ids):
            return 0.
        shape = self.bread_thickness_m.shape
        pixel = (xy - self.bread_bounds_xy[0]) / self.bread_cell_size_xy - .5
        lower = np.floor(pixel).astype(np.int64)
        weight = pixel - lower
        targets, sources, requests = [], [], []
        flat = self.blade_thickness_m.reshape(-1)
        area = (self.blade_cell_area_m2 if self.blade_profile is None
                else self.blade_cell_areas_m2.reshape(-1)[source_ids])
        available = flat[source_ids] * area * fraction
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            x, y = lower[:, 0] + dx, lower[:, 1] + dy
            ids = np.flatnonzero((x >= 0) & (y >= 0) & (x < shape[1]) & (y < shape[0]))
            if not len(ids):
                continue
            ids = ids[self._brush_normalizer[y[ids], x[ids]] > 1e-15]
            ww = (weight[ids, 0] if dx else 1 - weight[ids, 0]) * (weight[ids, 1] if dy else 1 - weight[ids, 1])
            targets.append(y[ids] * shape[1] + x[ids])
            sources.append(source_ids[ids])
            requests.append(available[ids] * ww)
        if not targets:
            return 0.
        targets, sources, requests = np.concatenate(targets), np.concatenate(sources), np.concatenate(requests)
        normalizer = self._brush_normalizer.reshape(-1)
        anchors = np.bincount(targets, weights=requests / normalizer[targets], minlength=self.bread_thickness_m.size)
        demand = (self._blur(anchors.reshape(shape)) * self.bread_mask).reshape(-1)
        capacity = (np.maximum(self.config.bread_max_thickness_m - self.bread_thickness_m.reshape(-1), 0.)
                    * self.bread_cell_area_m2)
        scale = np.minimum(1., np.divide(capacity, demand, out=np.ones_like(demand), where=demand > 0))
        added = demand * scale
        if np.all(scale == 1.):
            accepted = requests
        else:
            feedback = self._blur(scale.reshape(shape) * self.bread_mask).reshape(-1)
            accepted = requests * np.clip(feedback[targets] / normalizer[targets], 0., 1.)
        removed = np.bincount(sources, weights=accepted, minlength=flat.size)
        self.bread_thickness_m.reshape(-1)[:] += added / self.bread_cell_area_m2
        area = self.blade_cell_area_m2 if self.blade_profile is None else self.blade_cell_areas_m2.reshape(-1)
        flat[:] = np.maximum(flat - removed / area, 0.)
        return float(accepted.sum())

    def _transfer(self, dt, points, normals_z, contacts, contacts_blade):
        cfg = self.config
        if contacts is None or not len(contacts):
            self._previous_bread_points = self._previous_contacts = None
            return 0.
        # Only top-surface contacts within the blade's extent, inside the bread's contact outline.
        valid = np.abs(contacts[:, 2] - self.bread_top_z) <= cfg.surface_tolerance_m
        outside = (np.maximum(self.blade_bounds_xyz[0] - contacts_blade, 0.)
                   + np.maximum(contacts_blade - self.blade_bounds_xyz[1], 0.))
        valid &= np.linalg.norm(outside, axis=1) <= cfg.surface_tolerance_m
        if self.blade_profile is not None:
            surfaces, normals = blade_surface_samples(contacts_blade[:, :2], self.blade_bounds_xyz, self.blade_profile)
            valid &= (surfaces[1, :, 2] - contacts_blade[:, 2]) * (-normals[1, :, 2]) <= cfg.surface_tolerance_m
        valid &= inside_polygon(contacts[:, :2], self.bread_contact_outline_xy, boundary_tolerance=1e-6)
        contacts, contacts_blade = contacts[valid], contacts_blade[valid]
        if not len(contacts):
            self._previous_bread_points = self._previous_contacts = None
            return 0.
        previous, steps = self._previous_bread_points, 1
        if previous is not None:
            distance = float(np.linalg.norm(points - previous, axis=1).max())
            if distance <= cfg.max_sweep_distance_m:
                spacing = max(min(self.bread_cell_size_xy) * .5, cfg.contact_brush_radius_m * .5)
                steps = min(cfg.max_sweep_samples, max(1, math.ceil(distance / spacing)))
            else:
                previous = None  # never draw a stroke across a teleport
        fraction = -math.expm1(-cfg.transfer_rate_s * dt / steps)
        if self.blade_profile is not None:
            surfaces, normals = blade_surface_samples(contacts_blade[:, :2], self.blade_bounds_xyz, self.blade_profile)
            distances = np.abs((surfaces[..., 2] - contacts_blade[:, 2]) * normals[..., 2])
            thickness = surfaces[0, :, 2] - surfaces[1, :, 2]
            tie = np.minimum(cfg.surface_tolerance_m * .15, thickness * .1)
            nearest = np.any(distances <= distances.min(axis=0) + tie, axis=1)
            near_face = np.repeat(nearest, np.prod(cfg.blade_resolution))
            permitted = ((normals_z <= -cfg.min_downward_face_cos)
                         | ((abs(normals_z) < cfg.min_downward_face_cos) & near_face))
        else:
            if abs(normals_z[0]) >= cfg.min_downward_face_cos:
                faces = normals_z < 0.
            else:  # near-vertical blade: the broad face nearest the actual contact
                face_z = self.blade_bounds_xyz[::-1, 2]
                distance = np.min(abs(contacts_blade[:, 2, None] - face_z), axis=0)
                tie = min(cfg.surface_tolerance_m * .15, np.diff(self.blade_bounds_xyz[:, 2])[0] * .1)
                faces = distance <= distance.min() + tie
            permitted = np.repeat(faces, np.prod(cfg.blade_resolution))
        request_ids, request_xy, request_fractions = [], [], []
        remaining = np.ones(self.blade_thickness_m.size)
        for sample in range(1, steps + 1):
            alpha = sample / steps
            brush = points if previous is None else previous * (1 - alpha) + points * alpha
            valid = (permitted & (self.blade_thickness_m.reshape(-1) > 0.)
                     & (np.abs(brush[:, 2] - self.bread_top_z) <= cfg.surface_tolerance_m))
            valid &= ((brush[:, 0] >= self.bread_bounds_xy[0, 0]) & (brush[:, 0] <= self.bread_bounds_xy[1, 0])
                      & (brush[:, 1] >= self.bread_bounds_xy[0, 1]) & (brush[:, 1] <= self.bread_bounds_xy[1, 1]))
            ids = np.flatnonzero(valid)
            if not len(ids):
                continue
            ids = ids[inside_polygon(brush[ids, :2], self.bread_outline_xy, include_boundary=False)]
            request_ids.append(ids)
            request_xy.append(brush[ids, :2])
            request_fractions.append(remaining[ids] * fraction)
            remaining[ids] *= 1 - fraction
        total = (self._deposit(np.concatenate(request_ids), np.concatenate(request_xy),
                               np.concatenate(request_fractions)) if request_ids else 0.)
        self._previous_bread_points, self._previous_contacts = points.copy(), contacts.copy()
        return total

    def update(self, dt, blade_pose_xyzw, can_pose_xyzw, bread_pose_xyzw, contacts_world=None):
        """Advance one physics step from native body poses and blade/bread contact points."""
        bt, br = pose_matrix(blade_pose_xyzw)
        ct, cr = pose_matrix(can_pose_xyzw)
        rt, rr = pose_matrix(bread_pose_xyzw)
        contacts = contacts_blade = None
        if contacts_world is not None:
            contacts = np.asarray(contacts_world, dtype=np.float64).reshape(-1, 3)
            contacts_blade = (contacts - bt) @ br
            contacts = (contacts - rt) @ rr
        cfg = self.config
        center_can = (br @ self._blade_center + bt - ct) @ cr
        near_can = (np.linalg.norm(center_can[:2]) <= cfg.can_inner_radius_m + self._blade_radius
                    and cfg.can_bottom_z_m - self._blade_radius <= center_can[2]
                    <= cfg.can_mouth_z_m + self._blade_radius + cfg.max_step_distance_m)
        has_contacts = contacts is not None and len(contacts) > 0
        world = self._blade_points @ br.T + bt if near_can or has_contacts else None
        loaded = transferred = 0.
        if near_can:
            loaded = self._load(float(dt), (world - ct) @ cr)
        else:
            self._entered.fill(False)
            self._dwell.fill(0.)
            self._previous_can_points = None
            self._dip_counted = False
        if has_contacts:
            if self.blade_profile is None:
                face_normal = rr.T @ br[:, 2]
                normals_z = np.array([face_normal[2], -face_normal[2]])
            else:
                normals_z = (self._blade_normals @ br.T @ rr)[:, 2]
            transferred = self._transfer(float(dt), (world - rt) @ rr, normals_z, contacts, contacts_blade)
        else:
            self._previous_bread_points = self._previous_contacts = None
        self._transferred_m3 += transferred
        self._last_transfer_m3 = transferred
        if loaded > 0 or transferred > 0:
            self.version += 1
        return self.summary()

    def summary(self):
        blade = (float(self.blade_thickness_m.sum() * self.blade_cell_area_m2) if self.blade_profile is None
                 else float(np.sum(self.blade_thickness_m * self.blade_cell_areas_m2)))
        bread = float(self.bread_thickness_m.sum() * self.bread_cell_area_m2)
        return {'blade_ml': blade * 1e6, 'bread_ml': bread * 1e6,
                'blade_coverage': float(np.mean(self.blade_thickness_m > 1e-6)),
                'bread_coverage': float(np.count_nonzero(self.bread_thickness_m > 1e-6) / self.bread_mask.sum()),
                'loaded_ml': self._loaded_m3 * 1e6, 'transferred_ml': self._transferred_m3 * 1e6,
                'last_transfer_ml': self._last_transfer_m3 * 1e6, 'dip_count': self._dip_count,
                'volume_error_ml': (self._loaded_m3 - blade - bread) * 1e6, 'version': self.version}
