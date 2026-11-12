# Configuration

Each `robots/<ID>.yaml` defines each signal once, with its data format and model
settings together. `loader.py` expands defaults and builds the model configuration.

```text
configs/
├── loader.py               # Load defaults and build model configurations
├── robots/
│   └── <ID>.yaml            # One complete definition per embodiment
└── model/
    └── transformer.yaml    # Shared trunk and action-query allocation
```

Shared training defaults live in `egox.training.config`. Task/stage commands live in
[scripts/training_scripts](../../../scripts/training_scripts/README.md).

## Robot files

See [G1-Dex5.yaml](robots/G1-Dex5.yaml) for the layout used by all robots.

| Section | Meaning |
|---|---|
| `embodiment` | Robot/dataset ID; must match the filename |
| `tokenizer.Image` | Cameras: data format, image encoder, and token count |
| `tokenizer.proprioceptions` | Observed numeric signals: shape, data format, MLP, and token count |
| `action` | Targets for one timestep: shape, data format, decoder, and loss weight |
| `dreaming.embodiment_tokenizer` | Pools the selected observations into embodiment tokens |
| `dreaming.head` | Predicts future embodiment tokens from the trunk output |

Omit `dreaming` or set it to `null` for plain MXT. Otherwise, configure both components:

```yaml
dreaming:
  embodiment_tokenizer:
    norm_upper_bound: 100.0
    norm_upper_bound_weight: 1.0e-5
    modalities: [main_image, robot_joint_pos_proprio]
    token_num: 8
  head:
    crossattn_modality_dropout: 0.1
    crossattn_heads: 8
    crossattn_dim_head: 32
    dream_offsets: [20, 40, 60]
    loss_weight: 0.3
    dream_mag_beta: 0.01
```

Within a section, `defaults` applies to every signal in that group; a named signal
overrides those defaults. `{}` uses defaults unchanged. Defaults never cross robot files.

| Setting | Meaning |
|---|---|
| `shape: [N]` | N scalar channels, including padding; excludes batch size and action horizon |
| `dtype`, `layout`, `color_space` | Decoded data format; `HWC` means height/width/channels, `RGB` means red/green/blue |
| `tokenizer.pooling` | Default attention head count (`nhead`), width per head (`dim_head`), and dropout for modality pooling |
| `token_num` | Number of learned tokens for a signal, or for the dreaming summary |
| `resnet_model`, `weights` | Image backbone and its initialization weights |
| `split_dual` | Encode the two halves of a stereo image separately |
| `widths`, `tanh_end`, `ln` | State MLP hidden widths, final tanh, and LayerNorm |
| `crossattn_heads`, `crossattn_dim_head`, `crossattn_modality_dropout` | Decoder attention head count, width per head, and attention dropout |
| `loss_weight` | Weight of the action or dreaming loss |
| `dreaming.embodiment_tokenizer.modalities` | Observations pooled into embodiment tokens |
| `dream_offsets` | Future frame offsets predicted by the dreaming head |
| `norm_upper_bound`, `norm_upper_bound_weight` | Soft token-norm limit and its penalty weight |
| `dream_mag_beta` | Weight of the magnitude term in the dreaming loss |

**Keep signal order stable:** image and proprioception order sets model token positions;
action and robot order also affect seeded initialization.

## Load and inspect

```python
from egox.configs.loader import load_robot_config, build_model_config
from egox.training.config import TrainConfig

robot = load_robot_config("Z1-Tool")
joint = robot["tokenizer"]["proprioceptions"]["robot_joint_pos_proprio"]
shape, dtype = joint["shape"], joint["dtype"]  # [6], "float32"; no PyTorch import
model_config = build_model_config(TrainConfig(robots=["Z1-Tool"]))
```

```bash
bash scripts/training_scripts/G1-Sim-Dex3/kitchen-Bread/pretrain.sh \
    --dataset_dir /data/pretrain --print-config
```

## Add an embodiment

Copy the closest robot YAML, update its ID, signals, and model sections, and include
episodes labeled with that ID. New signal names also need storage/mask entries in
`egox.data.mapping`; new action names need query allocations in the trunk's `output_len`.
No script changes are needed to register the new embodiment.
