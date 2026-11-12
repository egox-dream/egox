"""Load YAML definitions and compose model settings without importing PyTorch."""
from copy import deepcopy
from pathlib import Path

import yaml

from egox.data.mapping import KEY_MASK_PATH_MAP, KEY_PATH_MAP

CONFIG_DIR = Path(__file__).parent
# Keep the released initialization order; newly added embodiments follow alphabetically.
_ROBOT_ORDER = ('Human-Dex5', 'G1-Dex5', 'G1-Sim-Dex3', 'R1Lite-Gripper',
                'xArm7-Dex4', 'LocoMan-Dex2', 'Z1-Tool')
_LABELS = {'dtype', 'layout', 'color_space', 'unit', 'frame', 'description'}
_METADATA = _LABELS | {'shape'}
_POOLING = {'nhead', 'dim_head', 'dropout'}
_CROSS_ATTENTION = {'crossattn_modality_dropout', 'crossattn_heads', 'crossattn_dim_head'}
# The model's constructors accept **kwargs, so a misspelled setting would otherwise do nothing.
_OPTIONS = {
    'Image': {'weights', 'resnet_model', 'split_dual', 'token_num'} | _POOLING,
    'proprioceptions': {'widths', 'tanh_end', 'ln', 'token_num'} | _POOLING,
    'action': _CROSS_ATTENTION | {'loss_weight'},
}
_EMBODIMENT_TOKENIZER = {'norm_upper_bound', 'norm_upper_bound_weight', 'modalities', 'token_num'} | _POOLING
_DREAM_HEAD = _CROSS_ATTENTION | {'dream_offsets', 'loss_weight', 'dream_mag_beta'}


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f'Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _load_yaml(path):
    with Path(path).open() as stream:
        config = yaml.load(stream, Loader=_UniqueLoader)
    if not isinstance(config, dict):
        raise ValueError(f'{path}: configuration must be a mapping')
    return config


def load_transformer_config(path=None):
    """Read the shared trunk, preserving action-query order."""
    return _load_yaml(path if path is not None else CONFIG_DIR / 'model/transformer.yaml')


def list_robot_configs(config_dir=None):
    """Discover robot YAMLs without maintaining a separate list of IDs."""
    folder = Path(config_dir) if config_dir is not None else CONFIG_DIR / 'robots'
    return sorted(path.stem for path in folder.glob('*.yaml'))


def _field_options(group, defaults, value, context):
    if not isinstance(value, dict):
        raise ValueError(f'{context}: expected a field mapping, e.g. {{shape: [6]}}')
    field = {**defaults, **value}
    _check_keys(field, _METADATA | _OPTIONS[group], context)
    shape = field.get('shape')
    if shape is not None:
        if (not isinstance(shape, list) or not shape
                or any(type(size) is not int or size <= 0 for size in shape)):
            raise ValueError(f'{context}.shape: expected positive integer dimensions, e.g. [6]')
        dimensions = 3 if group == 'Image' else 1
        if len(shape) != dimensions:
            raise ValueError(f'{context}.shape: expected {dimensions} dimensions')
    elif group != 'Image':
        raise ValueError(f'{context}.shape: numeric signals require a shape')
    for label in _LABELS:
        if label in field and (not isinstance(field[label], str) or not field[label]):
            raise ValueError(f'{context}.{label}: expected a nonempty string')
    if 'layout' in field and (group != 'Image' or field['layout'] not in ('HWC', 'CHW')):
        raise ValueError(f'{context}.layout: expected HWC or CHW for video')
    return field


def load_robot_config(embodiment, config_dir=None):
    """Read a robot YAML and expand defaults, keeping its structure and signal order.

    Each signal includes both metadata and model options. Collection/environment code
    can read its shape and data format directly, without importing PyTorch.
    """
    if not isinstance(embodiment, str) or not embodiment or Path(embodiment).name != embodiment:
        raise ValueError('embodiment must be a nonempty ID, not a path')
    folder = Path(config_dir) if config_dir is not None else CONFIG_DIR / 'robots'
    config = _load_yaml(folder / f'{embodiment}.yaml')
    _check_keys(config, {'embodiment', 'tokenizer', 'action', 'dreaming'}, embodiment)
    if config.get('embodiment') != embodiment:
        raise ValueError(f'{embodiment}: embodiment must match the filename')
    sections = config.get('tokenizer')
    _check_keys(sections, {'pooling', 'Image', 'proprioceptions'}, f'{embodiment}.tokenizer')
    names = set()
    for parent, group in ((sections, 'Image'), (sections, 'proprioceptions'), (config, 'action')):
        fields = parent.get(group)
        context = f'{embodiment}.{group}'
        if not isinstance(fields, dict):
            raise ValueError(f'{context}: expected a mapping of signals')
        defaults = fields.get('defaults', {})
        _check_keys(defaults, _METADATA | _OPTIONS[group], f'{context}.defaults')
        expanded = {}
        for name, value in fields.items():
            if name == 'defaults':
                continue
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f'{context}: field names must be nonempty and unique')
            names.add(name)
            expanded[name] = _field_options(group, defaults, value, f'{context}.{name}')
        parent[group] = expanded
    return config


def _check_keys(values, allowed, context):
    if not isinstance(values, dict):
        raise ValueError(f'{context}: expected a mapping')
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f'{context}: unknown settings {sorted(unknown, key=str)}; allowed: {sorted(allowed)}')


def _check_signals(names, context):
    unknown = set(names) - set(KEY_PATH_MAP)
    if unknown:
        raise ValueError(f'{context}: unknown signals {sorted(unknown, key=str)}')


def _model_options(field, group, context):
    """Pass only architecture settings to model constructors."""
    options = {key: value for key, value in field.items() if key not in _METADATA}
    _check_keys(options, _OPTIONS[group], context)
    if group != 'action' and 'token_num' not in options:
        raise ValueError(f'{context}: needs a token_num in its tokenizer settings')
    return options


def resolve_policy(robot, hidden_dim, exclude=()):
    """Return one robot's `tokenizer`, `action_head` and optional `embodiment_dream_head`.

    Signals are built in the robot file's order, which sets token positions: reordering
    them changes what a trained checkpoint sees. `exclude` leaves signals out entirely.
    """
    name = robot['embodiment']
    sections = robot['tokenizer']
    groups = (sections['Image'], sections['proprioceptions'], robot['action'])
    missing = [key for fields in groups for key in fields
               if key not in KEY_PATH_MAP or key not in KEY_MASK_PATH_MAP]
    if missing:
        raise ValueError(f'{name}: add the storage paths of {missing} to egox.data.mapping')
    _check_signals(exclude, f'{name}: exclude_signals')
    video, state, action = ({key: field for key, field in fields.items() if key not in exclude}
                            for fields in groups)
    if not action:
        raise ValueError(f'{name}: exclude_signals leaves no action signals')

    if not isinstance(sections.get('pooling'), dict) or set(sections['pooling']) != _POOLING:
        raise ValueError(f'{name}: tokenizer.pooling must set {sorted(_POOLING)}')
    tokenizer = {'hidden_dim': hidden_dim, **sections['pooling']}
    for key, field in video.items():
        tokenizer[key] = {'output_dim': hidden_dim,
                          **_model_options(field, 'Image', f'{name}.Image.{key}')}
    for key, field in state.items():
        tokenizer[key] = {'input_dim': field['shape'][0], 'output_dim': hidden_dim,
                          **_model_options(field, 'proprioceptions', f'{name}.proprioceptions.{key}')}
    heads = {key: {'input_dim': hidden_dim, 'output_dim': field['shape'][0],
                   **_model_options(field, 'action', f'{name}.action.{key}')}
             for key, field in action.items()}
    config = {'tokenizer': tokenizer, 'action_head': heads}

    dreaming = robot.get('dreaming')
    if dreaming is not None:
        _check_keys(dreaming, {'embodiment_tokenizer', 'head'}, f'{name}: dreaming')
        summary = dreaming.get('embodiment_tokenizer')
        _check_keys(summary, _EMBODIMENT_TOKENIZER, f'{name}: dreaming.embodiment_tokenizer')
        _check_keys(dreaming.get('head'), _DREAM_HEAD, f'{name}: dreaming.head')
        if not isinstance(summary.get('modalities'), list):
            raise ValueError(f'{name}: dreaming.embodiment_tokenizer.modalities must be a list')
        _check_signals(summary['modalities'], f'{name}: dreaming.embodiment_tokenizer.modalities')
        modalities = [key for key in summary['modalities'] if key in tokenizer]
        if not modalities:
            raise ValueError(f'{name}: none of the dreaming modalities are available')
        tokenizer['embodiment_tokenizer'] = {**summary, 'modalities': modalities}
        config['embodiment_dream_head'] = {'input_dim': hidden_dim, 'output_dim': hidden_dim, **dreaming['head']}
    return deepcopy(config)


def build_model_config(args, embodiments=None):
    """Compose configs for discovered embodiments, or preview the available robot YAMLs."""
    trunk_args = load_transformer_config(args.trunk_config_path)
    available = list_robot_configs(args.robot_config_dir)
    if not available:
        raise ValueError(f'No robot YAMLs found in {args.robot_config_dir or CONFIG_DIR / "robots"}')
    selected = set(embodiments if embodiments is not None else (args.robots or available))
    if not selected:
        raise ValueError('No embodiments to configure')
    unknown = selected - set(available)
    if unknown:
        raise ValueError(f'No robot YAMLs for {sorted(unknown)}')
    ordered = [name for name in _ROBOT_ORDER if name in selected] + sorted(selected - set(_ROBOT_ORDER))
    embodiment_args = {}
    for embodiment in ordered:
        cfg = resolve_policy(load_robot_config(embodiment, args.robot_config_dir), trunk_args['hidden_dim'],
                             args.exclude_signals or ())
        missing = set(cfg['action_head']) - set(trunk_args['output_len'])
        if missing:
            raise ValueError(f'{embodiment}: add {sorted(missing)} to output_len in the trunk configuration')
        for head in cfg['action_head'].values():
            head['action_horizon'] = args.chunk_size
        for key, tok in cfg['tokenizer'].items():
            if 'image' in key:
                tok['feature_norm'] = args.image_feature_norm
                tok['gradient_checkpoint'] = args.gradient_checkpoint
        summary = cfg['tokenizer'].get('embodiment_tokenizer')
        if summary is not None:
            for option, key in (('embodiment_norm_upper_bound', 'norm_upper_bound'),
                                ('embodiment_norm_upper_bound_weight', 'norm_upper_bound_weight')):
                if getattr(args, option) is not None:
                    summary[key] = getattr(args, option)
        if 'embodiment_dream_head' in cfg and args.embodiment_dream_mag_beta is not None:
            cfg['embodiment_dream_head']['dream_mag_beta'] = args.embodiment_dream_mag_beta
        embodiment_args[embodiment] = cfg
    return {'embodiment_args': embodiment_args, 'trunk_args': trunk_args,
            'image_sharing': args.image_sharing}
