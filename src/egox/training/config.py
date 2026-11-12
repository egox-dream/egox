"""Training settings and validation, independent of command-line parsing and PyTorch."""
from ast import literal_eval
from dataclasses import InitVar, asdict, dataclass, field
from typing import get_args, get_origin, get_type_hints
import warnings


@dataclass
class TrainConfig:
    """Shared defaults; scripts/training_scripts contains explicit task/stage recipes."""

    choices = {'image_sharing': ['none', 'encoder', 'tokenizer'],
               'lr_schedule': ['constant', 'cosine']}

    # Data and preprocessing.
    dataset_dir: list[str] = field(default_factory=list)
    default_embodiment: str | None = None
    sample_weights: list[float] | None = None
    train_ratio: float = 0.99
    main_cam_width: int = 1280  # Combined left + right image width for a stereo main camera.
    main_cam_height: int = 480
    wrist_cam_width: int = 640  # Each wrist camera defaults to 480p (640 x 480).
    wrist_cam_height: int = 480
    vel_rebalance: bool = False
    vel_rebalance_weight_x: float = 1.0
    vel_rebalance_weight_y: float = 2.0
    vel_rebalance_weight_yaw: float = 3.0
    vel_rebalance_threshold: float = 0.1

    # Architecture selection and global model overrides.
    robot_config_dir: str | None = None
    trunk_config_path: str | None = None
    robots: list[str] | None = None
    exclude_signals: list[str] | None = None
    chunk_size: int = 60
    image_sharing: str = 'none'
    image_feature_norm: bool = True
    embodiment_norm_upper_bound: float | None = None
    embodiment_norm_upper_bound_weight: float | None = None
    embodiment_dream_mag_beta: float | None = None

    # Optimization.
    batch_size: int = 32
    per_emb_batch_size: int | None = None
    num_steps: int = 10000
    lr: float = 1e-5
    lr_tokenizer: float = 1e-5
    lr_action_head: float = 1e-5
    lr_dream_head: float = 1e-6
    lr_trunk: float = 1e-5
    weight_decay: float = 1e-4
    lr_schedule: str = 'constant'
    cosine_min_lr_ratio: float = 0.05
    bf16: bool = True
    gradient_checkpoint: bool = False
    cudnn_benchmark: bool = True
    seed: int = 6
    gpu_id: int = 0

    # Checkpoints and logging.
    ckpt_dir: str = 'ckpt/'
    task_name: str | None = None
    validate_every: int = 1000
    max_val_batches: int = 52
    save_every: int = 5000
    snapshot_every: int = 500
    pretrained_path: str | None = None
    resume_dir: str | None = None
    auto_resume: bool = True
    wandb: bool = False
    wandb_name: str = 'MXT'

    print_config: bool = False

    # Constructor inputs only; saved configs keep the four width/height fields.
    main_cam_resolution: InitVar[str | tuple[int, int] | None] = None
    wrist_cam_resolution: InitVar[str | tuple[int, int] | None] = None

    def __post_init__(self, main_cam_resolution, wrist_cam_resolution):
        for camera, resolution in (('main', main_cam_resolution), ('wrist', wrist_cam_resolution)):
            if resolution is not None:
                if isinstance(resolution, str):
                    try:
                        resolution = literal_eval(resolution)
                    except (ValueError, SyntaxError):
                        resolution = None
                if not isinstance(resolution, tuple) or len(resolution) != 2:
                    raise ValueError(f'{camera}_cam_resolution: expected "(width,height)"')
                setattr(self, f'{camera}_cam_width', resolution[0])
                setattr(self, f'{camera}_cam_height', resolution[1])
        for name, annotation in get_type_hints(type(self)).items():
            if isinstance(annotation, InitVar):
                continue
            if not self._matches(getattr(self, name), annotation):
                raise ValueError(f'{name}: expected {annotation}, got {getattr(self, name)!r}')
        for name, choices in self.choices.items():
            if getattr(self, name) not in choices:
                raise ValueError(f'{name}: choose from {", ".join(choices)}')
        if self.sample_weights is not None and len(self.sample_weights) != len(self.dataset_dir):
            raise ValueError('--sample_weights needs one weight per --dataset_dir')
        for name in ('chunk_size', 'snapshot_every', 'max_val_batches',
                     'main_cam_width', 'main_cam_height', 'wrist_cam_width', 'wrist_cam_height'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')

    def to_dict(self):
        return asdict(self)

    def warn_resume_changes(self, saved):
        """Report changed training settings without blocking a compatible resume."""
        ignored = {'gpu_id', 'ckpt_dir', 'task_name', 'resume_dir', 'auto_resume',
                   'wandb', 'wandb_name', 'save_every', 'snapshot_every',
                   'pretrained_path', 'print_config'}
        changes = [f"  {name}: {saved.get(name, '<not recorded>')!r} -> {value!r}"
                   for name, value in self.to_dict().items()
                   if name not in ignored and (name not in saved or saved[name] != value)]
        if changes:
            warnings.warn('Resuming with changed training settings:\n' + '\n'.join(changes)
                          + '\nSaved optimizer moments, RNG, data cursors and normalization statistics '
                          'are retained; this is not an exact continuation.', stacklevel=2)

    @staticmethod
    def _base_type(annotation):
        if type(None) in get_args(annotation):
            return next(value for value in get_args(annotation) if value is not type(None))
        return annotation

    @staticmethod
    def _matches(value, annotation):
        if value is None:
            return type(None) in get_args(annotation)
        annotation = TrainConfig._base_type(annotation)
        if get_origin(annotation) is list:
            return isinstance(value, list) and all(TrainConfig._matches(item, get_args(annotation)[0]) for item in value)
        if annotation is float:
            return type(value) in (int, float)
        return type(value) is annotation
