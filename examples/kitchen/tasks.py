"""Simulator side of the two G1-Sim-Dex3 kitchen tasks (import after the Isaac Sim app starts).

- `BreadTask` (Kitchen-Bread): advances the coating model after every physics step from the
  spreader, can and bread poses and the blade/bread contact points, and renders the coating
  on the blade and the bread with dynamic textures.
- `CleanupTask` (Kitchen-Cleanup): evaluates the holder placement after every physics step
  from the tool geometry, the right hand's aperture and the PhysX contact reports.

Both take their geometry and thresholds from the trial (trial.json), are constructed before
physics starts, bound to the environment afterwards, and expose `summary()` for the trace.
"""

import math
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from coating import CoatingModel
from holder import HolderEvaluator
from egox.robots.g1_sim_dex3.controller import HAND_CLOSED

ROOM = '/World/SceneSmith'


def contact_reports(paths):
    """PhysX contact report of the last physics step: (collider0, collider1, contacts) path pairs."""
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools

    headers, data = get_physx_simulation_interface().get_contact_report()
    for header in headers:
        if not header.num_contact_data:
            continue
        names = []
        for key in (header.collider0, header.collider1):
            if key not in paths:
                paths[key] = str(PhysicsSchemaTools.intToSdfPath(key))
            names.append(paths[key])
        begin = header.contact_data_offset
        yield names[0], names[1], [data[i] for i in range(begin, begin + header.num_contact_data)]


def enable_contact_reports(stage, bodies):
    import carb
    from pxr import PhysxSchema, Usd

    carb.settings.get_settings().set_bool('/physics/disableContactProcessing', False)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in bodies:
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)


class CoatingVisuals:
    """Coating on the blade faces and the bread top, blended over each surface's base material.

    Opinions live in an anonymous session sublayer; color and roughness/metallic maps are
    dynamic textures uploaded when the coating changes.
    """
    OPACITY_SCALE_M = .000045

    def __init__(self, stage, model, surfaces, appearance):
        import omni.ui as ui
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        self.stage, self.model, self.appearance = stage, model, appearance
        self._ui, self._token, self._surfaces = ui, uuid.uuid4().hex, []
        self._version = None
        self._layer = Sdf.Layer.CreateAnonymous('egox_coating_visuals.usda')
        stage.GetSessionLayer().subLayerPaths.insert(0, self._layer.identifier)
        looks = f'{ROOM}/EgoXCoatingLooks_{self._token}'
        bases = {(s['source'], s['side']): s['base'] for s in surfaces}
        with Usd.EditContext(stage, self._layer):
            blade = UsdGeom.Mesh(stage.GetPrimAtPath(f'{ROOM}/Props/Spreader/Blade'))
            bread = UsdGeom.Mesh(stage.GetPrimAtPath(f'{ROOM}/Props/Bread/Top'))
            self._uv(blade, model.blade_bounds_xyz[:, :2])
            self._uv(bread, model.bread_bounds_xy)
            for side, faces in enumerate(self._blade_faces(blade)):
                subset = UsdGeom.Subset.Define(stage, blade.GetPath().AppendChild(
                    'EgoXCoatingPositiveZ' if side == 0 else 'EgoXCoatingNegativeZ'))
                subset.CreateElementTypeAttr('face')
                subset.CreateFamilyNameAttr('materialBind')
                subset.CreateIndicesAttr(faces)
                self._surface(subset.GetPrim(), looks, f'Blade{side}', bases[('blade', side)], 'blade', side)
            UsdGeom.Subset.SetFamilyType(blade, 'materialBind', 'nonOverlapping')
            self._surface(bread.GetPrim(), looks, 'Bread', bases[('bread', 0)], 'bread', 0)
            # Liquid surface inside the can, at the fill height.
            config = model.config
            radius, fill = config.can_inner_radius_m * (1. - .00066 / .03966), config.can_fill_z_m
            mesh = UsdGeom.Mesh.Define(stage, f'{ROOM}/Props/OpenMilkCan/EgoXMilkSurface')
            angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
            mesh.CreatePointsAttr([Gf.Vec3f(0., 0., fill)] + [Gf.Vec3f(float(radius * np.cos(a)), float(radius * np.sin(a)), fill)
                                                              for a in angles])
            mesh.CreateFaceVertexCountsAttr([3] * 96)
            mesh.CreateFaceVertexIndicesAttr([v for i in range(96) for v in (0, i + 1, (i + 1) % 96 + 1)])
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateExtentAttr([Gf.Vec3f(-radius, -radius, fill), Gf.Vec3f(radius, radius, fill)])
            material = UsdShade.Material.Define(stage, f'{looks}/MilkSurface')
            shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Shader'))
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*appearance['color']))
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(appearance['roughness'])
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        self.flush()

    def _local_points(self, mesh):
        from pxr import UsdGeom
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        transform = np.asarray(UsdGeom.Xformable(mesh).GetLocalTransformation())
        return (np.column_stack([points, np.ones(len(points))]) @ transform)[:, :3]

    def _uv(self, mesh, bounds):
        from pxr import Gf, Sdf, UsdGeom
        uv = (self._local_points(mesh)[:, :2] - bounds[0]) / (bounds[1] - bounds[0])
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex')
        primvar.Set([Gf.Vec2f(*map(float, point)) for point in uv])
        primvar.BlockIndices()

    def _blade_faces(self, blade):
        """Faces of the +Z and -Z (including bevel) blade surfaces, from the mesh topology."""
        vertices = self._local_points(blade)
        counts = np.asarray(blade.GetFaceVertexCountsAttr().Get())
        faces = np.split(np.asarray(blade.GetFaceVertexIndicesAttr().Get()), np.cumsum(counts)[:-1])
        bounds = self.model.blade_bounds_xyz
        if self.model.blade_profile is None:
            return [[i for i, face in enumerate(faces) if np.allclose(vertices[face, 2], z, atol=1e-7)]
                    for z in (bounds[1, 2], bounds[0, 2])]
        result = [[], []]
        for i, face in enumerate(faces):
            points = vertices[face]
            normal = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
            norm = np.linalg.norm(normal)
            if norm > 1e-15 and abs(normal[2]) / norm > 1e-7:
                result[0 if normal[2] > 0 else 1].append(i)
        return result

    def _surface(self, prim, looks, name, base, source, side):
        from pxr import Sdf, UsdShade
        material = UsdShade.Material.Define(self.stage, f'{looks}/{name}')
        shader = UsdShade.Shader.Define(self.stage, material.GetPath().AppendChild('Shader'))
        shader.CreateIdAttr('UsdPreviewSurface')
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        reader = UsdShade.Shader.Define(self.stage, material.GetPath().AppendChild('UV'))
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        providers = []
        for channel in ('Color', 'ORM'):
            token = f'egox_coating_{self._token}_{name}_{channel}'
            providers.append(self._ui.DynamicTextureProvider(token))
            texture = UsdShade.Shader.Define(self.stage, material.GetPath().AppendChild(channel))
            texture.CreateIdAttr('UsdUVTexture')
            texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('dynamic://' + token))
            texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('raw')
            texture.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('clamp')
            texture.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('clamp')
            texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
            if channel == 'Color':
                shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), 'rgb')
            else:
                shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(), 'g')
                shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(), 'b')
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        self._surfaces.append({'source': source, 'side': side, 'providers': providers, 'last': None,
                               'base': {'color': np.asarray(base['color'], dtype=float),
                                        'roughness': float(base['roughness']), 'metallic': float(base['metallic'])}})

    def flush(self):
        """Upload the maps of surfaces whose coating changed (linear values)."""
        if self._version == self.model.version:
            return
        color_coat, rough_coat = np.asarray(self.appearance['color']), self.appearance['roughness']
        for surface in self._surfaces:
            thickness = (self.model.blade_thickness_m[surface['side']] if surface['source'] == 'blade'
                         else self.model.bread_thickness_m)
            if surface['last'] is not None and np.array_equal(thickness, surface['last']):
                continue
            weight = (1. - np.exp(-thickness / self.OPACITY_SCALE_M))[..., None]
            base = surface['base']
            color = np.concatenate([base['color'] * (1 - weight) + color_coat * weight, np.ones_like(weight)], axis=2)
            orm = np.concatenate([np.ones_like(weight), base['roughness'] * (1 - weight) + rough_coat * weight,
                                  base['metallic'] * (1 - weight), np.ones_like(weight)], axis=2)
            for provider, image in zip(surface['providers'], (color, orm)):
                buffer = np.ascontiguousarray(np.flipud(np.clip(image * 255, 0, 255).round().astype(np.uint8)))
                provider.set_data_array(buffer, [buffer.shape[1], buffer.shape[0]])
            surface['last'] = thickness.copy()
        self._version = self.model.version


class BreadTask:
    BODIES = ('Spreader', 'OpenMilkCan', 'Bread')

    def __init__(self, stage, trial):
        c = trial['coating']
        self.model = CoatingModel(c['config'], c['blade_bounds_xyz'], c['bread_outline_xy'], c['bread_top_z'],
                                  c['bread_contact_outline_xy'], c['blade_profile'])
        self.paths = [f'{ROOM}/Props/{name}' for name in self.BODIES]
        self.blade, self.crust = f'{ROOM}/Props/Spreader/Blade', f'{ROOM}/Props/Bread/Crust'
        enable_contact_reports(stage, (self.paths[0], self.paths[2]))
        self.visuals = CoatingVisuals(stage, self.model, c['visual_surfaces'], c['appearance'])
        self._paths, self.contact_steps = {}, 0

    def bind(self, env):
        self.env = env
        self.indices = [env.room.paths.index(path) for path in self.paths]
        self.plate = next(i for i, p in enumerate(env.room.paths) if p.startswith(f'{ROOM}/Props/Plate/'))
        fridge = env.room.articulation_index(f'{ROOM}/Props/Refrigerator')
        self.door = (fridge, env.room.dof_index(fridge, 'RightDoor'))

    def reset(self, state):
        self.model.reset()
        self.model.blade_thickness_m[:] = state['coating/blade_thickness_full']
        self.model.bread_thickness_m[:] = state['coating/bread_thickness_full']
        self.model.version += 1
        self.contact_steps = 0
        self.visuals.flush()

    def after_physics_step(self, dt):
        poses = self.env.room.rigid_poses()[self.indices]
        x, y, z, w = poses[2][3:]
        bread_up = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        points = []
        wanted = {self.blade, self.crust}
        for first, second, contacts in contact_reports(self._paths):
            if {first, second} != wanted:
                continue
            for contact in contacts:  # compressive top contacts only
                impulse, normal = np.asarray(contact.impulse, dtype=float), np.asarray(contact.normal, dtype=float)
                if (contact.separation <= .002 and abs(float(impulse @ normal)) > 1e-9
                        and abs(float(normal @ bread_up)) >= .5):
                    points.append(list(contact.position))
        if points:
            self.contact_steps += 1
        self.model.update(dt, *poses, contacts_world=np.asarray(points, dtype=float).reshape(-1, 3))

    def before_render(self):
        self.visuals.flush()

    def summary(self):
        return {**self.model.summary(), 'contact_steps': self.contact_steps}

    def trace(self, observation):
        """Per-sample inputs of the Kitchen-Bread scorer (world poses XYZ + WXYZ)."""
        room = self.env.room
        poses, velocities = room.rigid_poses(), room.rigid_velocities()
        result = {'door_angle': float(room.dof_positions(self.door[0])[self.door[1]])}
        for key, index in (('spreader', self.indices[0]), ('can', self.indices[1]), ('plate', self.plate)):
            result[f'{key}_pose'] = poses[index][[0, 1, 2, 6, 3, 4, 5]]
            result[f'{key}_velocity'] = velocities[index]
        for side in ('left', 'right'):
            result[f'{side}_wrist_pose'] = self.env.link_pose(f'{side}_wrist_yaw_link')
            result[f'{side}_eef_link_proprio'] = observation['obs'][f'{side}_eef_link_proprio']
        for key, value in self.summary().items():
            result[f'coating_{key}'] = value
        return result


class CleanupTask:
    def __init__(self, stage, trial):
        h = trial['holder']
        self.evaluator = HolderEvaluator(h['evaluation'])
        self.holder, self.tool = h['holder_path'], h['spreader_path']
        self.blade_points, self.handle_points = np.asarray(h['blade_points']), np.asarray(h['handle_points'])
        self.handle_center, self.initial_tool_z = np.asarray(h['handle_center']), h['initial_tool_z']
        enable_contact_reports(stage, (self.holder, self.tool))
        self._paths, self.state = {}, {}

    def bind(self, env):
        self.env = env
        self.holder_index, self.tool_index = env.room.paths.index(self.holder), env.room.paths.index(self.tool)
        self.hand = env.hand_ids[:7]  # right hand
        self.closed = HAND_CLOSED[:7]  # the open hand is all zeros

    def reset(self, state):
        self.evaluator.reset()
        self.state = {}
        self._update(0.)

    def _contacts(self):
        supported = touching = False
        for first, second, contacts in contact_reports(self._paths):
            names = (first, second)
            if not any(p.startswith(self.tool + '/') for p in names):
                continue
            other = next((p for p in names if not p.startswith(self.tool + '/')), '')
            close = any(c.separation <= .002 for c in contacts)
            supported |= close and other.startswith(self.holder + '/')
            touching |= close and '/Robot/' in other and 'right' in other.lower()
        return bool(supported), bool(touching)

    def _update(self, dt):
        env = self.env
        poses = env.room.rigid_poses()
        tool, holder = poses[self.tool_index], poses[self.holder_index]
        to_world = lambda points, pose: Rotation.from_quat(pose[3:]).apply(points) + pose[:3]  # noqa: E731
        to_holder = lambda points: Rotation.from_quat(holder[3:]).inv().apply(np.asarray(points) - holder[:3])  # noqa: E731
        handle_world = to_world(self.handle_center, tool)
        distance = float(np.linalg.norm(env.link_position_physx('right_wrist_yaw_link') - handle_world))
        delta = self.closed
        measured, targets = env.dof_positions_physx(self.hand), env.targets_numpy(self.hand)
        closure = float(np.clip(measured @ delta / (delta @ delta), 0, 1))
        commanded = float(np.clip(targets @ delta / (delta @ delta), 0, 1))
        supported, touching = self._contacts()
        velocity = env.room.rigid_velocities()[self.tool_index]
        holder_up = Rotation.from_quat(holder[3:]).apply([0., 0., 1.])
        self.state = self.evaluator.update(
            dt, blade_points_holder=to_holder(to_world(self.blade_points, tool)),
            handle_center_holder=to_holder(handle_world),
            handle_points_holder=to_holder(to_world(self.handle_points, tool)),
            holder_tilt_rad=float(math.acos(np.clip(holder_up[2], -1, 1))),
            linear_velocity_world=velocity[:3], angular_velocity_world=velocity[3:],
            right_grasp_command=1. - commanded, released=closure < .25 and not touching,
            hand_withdrawn=distance > .16, support_contact=supported,
            grasped=closure > .35 and (touching or distance < .13),
            lifted=bool(tool[2] > self.initial_tool_z + .025))
        self.state.update(right_hand_openness=1. - closure, right_hand_clearance_m=distance,
                          holder_support_contact=supported, right_hand_tool_contact=touching)

    def after_physics_step(self, dt):
        self._update(dt)

    def before_render(self):
        pass

    def summary(self):
        result = {}
        for key, value in self.state.items():
            if key == 'side_gap_m':
                result[key] = -1. if value is None else float(value)
            elif isinstance(value, dict):
                result.update({f'{key}_{k}': v for k, v in value.items()})
            elif isinstance(value, (bool, int, float)):
                result[key] = value
        return result

    def trace(self, observation):
        return {f'holder_{key}': value for key, value in self.summary().items()}


def make_task(stage, trial):
    return {'kitchen_bread': BreadTask, 'kitchen_cleanup': CleanupTask}[trial['task']](stage, trial)
