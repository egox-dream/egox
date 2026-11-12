"""Kitchen scene choices, frozen trials and collection-only object resets."""
import json
from itertools import product
from pathlib import Path

import numpy as np

from egox.robots.g1_sim_dex3.description import USD, camera_rig
from egox.sim.scene import RobotSetup

EXAMPLE = Path(__file__).resolve().parent
KITCHEN = EXAMPLE / 'scenes/bread_supported'
CAMERAS = ('Overview', 'Worktop', 'Refrigerator')


def load_trial(path):
    """Load the scene, overlay and recorded robot/controller state of one trial."""
    directory = Path(path).resolve()
    trial = json.loads((directory / 'trial.json').read_text())
    for key in ('scene_usd', 'overlay', 'state'):
        trial[key] = (directory / trial[key]).resolve()
        if not trial[key].is_file():
            raise FileNotFoundError(trial[key])
    with np.load(trial['state'], allow_pickle=False) as state:
        trial['state'] = dict(state)
    return trial


def robot_at(task):
    """Robot setup from the first trial; viewers use the default camera rig."""
    trial = load_trial(EXAMPLE / f'benchmarks/kitchen_{task}/trial_001')
    state = trial['state']
    pose = state['robot/pose_world_wxyz']
    joints = dict(zip(trial['joint_names'], state['robot/joint_position'].tolist()))
    return RobotSetup(USD, tuple(pose[:3].tolist()), tuple(pose[3:].tolist()), joints, camera_rig())


def add_arguments(parser):
    parser.add_argument('--variant', choices=('baseline', 'diverse'), default='diverse',
                        help='baseline: task props only; diverse: extra kitchenware')
    parser.add_argument('--scene-usd', '--scene_usd', type=Path, help='override the scene USD')
    parser.add_argument('--robot-at', '--robot_at', choices=('bread', 'cleanup', 'none'),
                        default='bread', help='robot start pose from the first task trial')


def scene_path(args):
    name = 'kitchen_workarea.usda' if args.variant == 'baseline' else 'kitchen_workarea_diverse.usda'
    return args.scene_usd or KITCHEN / name


# Original KitchenTwin collection ranges, in scene-local metres/radians. The
# table's existing aisle displacement is added to plate Y. Appearance, furniture
# and robot spawn remain those of the selected trial.
PLATE_XY = ((.60, .80), (.145, .235))
CAN_XY = ((1.8395, 1.9395), (-1.341, -1.317))
SPREADER_YAW = np.deg2rad((-50., -10.))
HOLDER_RADIUS = (.19, .30)
ROOM = '/World/SceneSmith'


def _vertices(stage, path, relative_to=None):
    """Composed mesh vertices, retaining mesh scales, in a rigid or world frame."""
    from pxr import Gf, Usd, UsdGeom
    from scipy.spatial import ConvexHull

    cache, points = UsdGeom.XformCache(), []
    frame = Gf.Matrix4d(1.)
    if relative_to:
        transform = Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(relative_to)))
        transform.SetScale(Gf.Vec3d(1.))
        frame = transform.GetMatrix().GetInverse()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if prim.IsA(UsdGeom.Mesh):
            vertices = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        elif prim.IsA(UsdGeom.Boundable):
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), Usd.TimeCode.Default())
            vertices = list(product(*zip(*extent))) if extent else []
        else:
            continue
        matrix = cache.GetLocalToWorldTransform(prim) * frame
        points.extend(tuple(matrix.Transform(Gf.Vec3d(*map(float, p)))) for p in vertices)
    if not points:
        raise ValueError(f'{path}: collection randomization requires mesh geometry')
    points = np.asarray(points)
    return points[ConvexHull(points).vertices] if len(points) > 100 else points


def _box(points):
    lo, hi = np.asarray(points)[:, :2].min(0), np.asarray(points)[:, :2].max(0)
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def _separated(first, second, gap):
    """Conservative separating-axis clearance between convex XY footprints."""
    for polygon in (first, second):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for edge in edges:
            norm = np.linalg.norm(edge)
            if norm > 1e-12:
                axis = np.array([-edge[1], edge[0]]) / norm
                a, b = first @ axis, second @ axis
                if a.max() + gap <= b.min() or b.max() + gap <= a.min():
                    return True
    return False


def _circle_clear(center, radius, polygon, gap):
    """Circle-to-convex-polygon clearance, including the circle-inside case."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    delta = np.asarray(center) - polygon
    cross = edges[:, 0] * delta[:, 1] - edges[:, 1] * delta[:, 0]
    if np.all(cross >= -1e-12) or np.all(cross <= 1e-12):
        return False
    t = np.clip(np.sum(delta * edges, axis=1) / np.maximum(np.sum(edges**2, axis=1), 1e-20), 0, 1)
    return np.linalg.norm(delta - t[:, None] * edges, axis=1).min() >= radius + gap


class CollectionReset:
    """Seeded object placement on each DDS reset; frozen trials stay untouched.

    Keeps the trial's settled heights, dimensions and spreader lean. Sampling uses
    the original plate/can patches and holder annulus, with current mesh support
    and clearance checks. No USD edits, visual randomization or accumulated jitter.
    """

    def __init__(self, env, seed):
        from scipy.spatial.transform import Rotation

        self.env, self.trial = env, env.trial
        if not self.trial or self.trial['task'] not in ('kitchen_bread', 'kitchen_cleanup'):
            raise ValueError('--randomize requires a kitchen --trial')
        self.rng = np.random.default_rng(seed)
        self.base = {key: value.copy() for key, value in self.trial['state'].items()}
        self.rotation = Rotation.from_euler('z', self.trial['scene_yaw'])
        self.origin = np.asarray(self.trial['scene_position'])
        self.paths = self.trial['rigid_bodies']
        self.indices = {name: next(i for i, p in enumerate(self.paths)
                                  if p == f'{ROOM}/Props/{name}' or p.startswith(f'{ROOM}/Props/{name}/'))
                        for name in ('Plate', 'Bread', 'Spreader', 'OpenMilkCan')}
        if self.trial['task'] == 'kitchen_cleanup':
            self.indices['UtensilHolder'] = self.paths.index(self.trial['holder']['holder_path'])
        self.poses = self.base['rigid_objects/pose_world_wxyz'].astype(float).copy()
        self.poses[:, :3] = self.rotation.inv().apply(self.poses[:, :3] - self.origin)
        rotations = self.rotation.inv() * Rotation.from_quat(self.poses[:, [4, 5, 6, 3]])
        self.poses[:, 3:] = rotations.as_quat()[:, [3, 0, 1, 2]]
        stage = env.scene.stage
        self.geometry = {name: _vertices(stage, self.paths[i], self.paths[i]) for name, i in self.indices.items()}
        self.handle = _vertices(stage, f'{ROOM}/Props/Spreader/Handle', f'{ROOM}/Props/Spreader')
        table = self.rotation.inv().apply(_vertices(stage, f'{ROOM}/Furniture/TaskCounter/Top') - self.origin)
        self.table = np.array([table[:, :2].min(0), table[:, :2].max(0)])
        self.aisle = self.table[0, 1]  # authored front_y == 0
        self.plate_radius = np.abs(self.geometry['Plate'][:, :2]).max()
        self.obstacles = []
        for i, path in enumerate(self.paths):
            if i not in self.indices.values():
                points = self._points(_vertices(stage, path, path), self.poses[i])
                if points[:, 2].max() > table[:, 2].max() and points[:, 2].min() < table[:, 2].max() + .25:
                    self.obstacles.append(_box(points))
        for name in ('IslandBlender', 'NestedBowls', 'UtensilPile'):
            path = f'{ROOM}/Furniture/{name}'
            if stage.GetPrimAtPath(path):
                points = self.rotation.inv().apply(_vertices(stage, path) - self.origin)
                self.obstacles.append(_box(points))

    @staticmethod
    def _points(points, pose):
        from scipy.spatial.transform import Rotation
        return Rotation.from_quat(pose[[4, 5, 6, 3]]).apply(points) + pose[:3]

    def _supported(self, points, margin):
        return bool(np.all(points[:, :2] >= self.table[0] + margin)
                    and np.all(points[:, :2] <= self.table[1] - margin))

    def sample(self):
        """Return a new state; rejection failure never installs a partial reset."""
        from scipy.spatial.transform import Rotation

        indices, rng = self.indices, self.rng
        poses = self.poses.copy()
        plate, bread, tool = [poses[indices[name]] for name in ('Plate', 'Bread', 'Spreader')]
        original_plate, original_tool = self.poses[indices['Plate']], self.poses[indices['Spreader']]
        original_rotation = Rotation.from_quat(original_tool[[4, 5, 6, 3]])
        original_yaw = original_rotation.as_euler('xyz')[2]
        for _ in range(256):
            plate[:2] = rng.uniform(*np.asarray(PLATE_XY).T) + [0, self.aisle]
            bread[:2] = plate[:2] + rng.uniform(-.015, .015, 2)
            delta = Rotation.from_euler('z', rng.uniform(*SPREADER_YAW) - original_yaw)
            tool[:3] = plate[:3] + delta.apply(original_tool[:3] - original_plate[:3])
            tool[3:] = (delta * original_rotation).as_quat()[[3, 0, 1, 2]]
            plate_points = self._points(self.geometry['Plate'], plate)
            tool_points = self._points(self.geometry['Spreader'], tool)
            if not self._supported(plate_points, .01) or not self._supported(tool_points, .01):
                continue
            if self.trial['task'] == 'kitchen_bread':
                if np.linalg.norm(bread[:2] - plate[:2]) + .076 > self.plate_radius - .02:
                    continue
                if not _circle_clear(bread[:2], .076, _box(self._points(self.handle, tool)), .008):
                    continue
            if any(not _separated(_box(points), obstacle, .03)
                   for points in (plate_points, tool_points) for obstacle in self.obstacles):
                continue
            break
        else:
            raise ValueError('no supported kitchen object placement after 256 attempts')
        can = poses[indices['OpenMilkCan']]
        radius = np.abs(self.geometry['OpenMilkCan'][:, :2]).max() + .005
        can_ranges = np.asarray(CAN_XY)
        bin_bounds = np.array([[1.707, 2.072], [-1.398, -1.258]])
        low = np.maximum(can_ranges[:, 0], bin_bounds[:, 0] + radius)
        high = np.minimum(can_ranges[:, 1], bin_bounds[:, 1] - radius)
        if np.any(low > high):
            raise ValueError('can does not fit the original refrigerator-bin sampling range')
        can[:2] = rng.uniform(low, high)
        if 'UtensilHolder' in indices:
            holder = poses[indices['UtensilHolder']]
            # Original holder reach envelope is measured after the 37 cm approach,
            # not from the initial 80 cm plate-to-base standing distance.
            robot = self.rotation.inv().apply(self.base['robot/pose_world_wxyz'][:3] - self.origin)
            robot[:2] += [0, .37]
            robot_q = self.base['robot/pose_world_wxyz'][[4, 5, 6, 3]]
            yaw = (self.rotation.inv() * Rotation.from_quat(robot_q)).as_euler('xyz')[2]
            for _ in range(4096):
                radius = np.sqrt(rng.uniform(*np.square(HOLDER_RADIUS)))
                angle = rng.uniform(-np.pi, np.pi)
                holder[:2] = plate[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
                holder[3:] = Rotation.from_euler('z', rng.uniform(-np.pi, np.pi)).as_quat()[[3, 0, 1, 2]]
                corners = self._points(self.geometry['UtensilHolder'], holder)
                footprint = _box(corners)
                local = Rotation.from_euler('z', -yaw).apply(holder[:3] - robot)
                if not (self._supported(corners, .015) and np.linalg.norm(local[:2]) <= .60
                        and .10 <= local[0] <= .60 and -.45 <= local[1] <= .20):
                    continue
                if (_circle_clear(plate[:2], self.plate_radius, footprint, .015)
                        and all(_separated(footprint, obstacle, .015)
                                for obstacle in [*self.obstacles, _box(tool_points)])):
                    break
            else:
                raise ValueError('no clear reachable holder placement after 4096 attempts')
        state = {key: value.copy() for key, value in self.base.items()}
        poses[:, :3] = self.rotation.apply(poses[:, :3]) + self.origin
        poses[:, 3:] = (self.rotation * Rotation.from_quat(poses[:, [4, 5, 6, 3]])).as_quat()[:, [3, 0, 1, 2]]
        moved = list(indices.values())
        state['rigid_objects/pose_world_wxyz'][moved] = poses[moved]
        state['rigid_objects/velocity_world'][moved] = 0
        return state

    def __call__(self):
        state = self.sample()
        previous = self.trial['state']
        try:
            self.trial['state'] = state
            return self.env.reset()
        finally:
            self.trial['state'] = previous
