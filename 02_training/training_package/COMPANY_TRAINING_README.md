# TB6_R5_ACT — company-ready training program

## Principle

Do not edit LeRobot's installed ACT source code.

For training-parameter iterations, edit/copy only:

`config/<task>/<policy>/<version>.sh`

Current baseline:

`config/pick_place_object_v1/act/v1.sh`

A substantial parameter change becomes `v2.sh`, `v3.sh`, etc.

## Output classification

Formal runs are saved automatically as:

`outputs/<task>/<policy>/<version>/train/<timestamp>/`

Example:

`outputs/pick_place_object_v1/act/v1/train/20260827_143000/`

Smoke runs are saved as:

`outputs/<task>/<policy>/<version>/smoke/<timestamp>/`

Each completed run also contains:

`experiment_config.sh`

LeRobot checkpoints contain their own `train_config.json`.

## Current ACT v1 baseline

Training:
- steps: 100000
- batch size: 8
- num workers: 4
- save every 20000 steps
- seed: 1000

ACT:
- chunk_size: 100
- n_action_steps: 100
- n_obs_steps: 1
- ResNet18 + ImageNet pretrained weights
- dim_model: 512
- heads: 8
- feedforward: 3200
- encoder layers: 4
- decoder layers: 1
- dropout: 0.1
- VAE enabled
- latent_dim: 32
- VAE encoder layers: 4
- KL weight: 10
- AdamW LR: 1e-5
- backbone LR: 1e-5
- weight decay: 1e-4
- image augmentation: disabled

## Laptop before transfer

Keep the already-generated real manifest:

`dataset_manifests/pick_place_object_v1_train.sha256`

Do not replace it with an empty placeholder.

## Company computer

After copying:
1. `TB6_R5_ACT/`
2. the complete `pick_place_object_v1_train/` dataset

activate the LeRobot 0.6.1 environment, then:

```bash
cd /path/to/TB6_R5_ACT

./01_train/run_preflight.sh /path/to/pick_place_object_v1_train
```

Only if preflight PASS:

```bash
./01_train/run_smoke_train.sh /path/to/pick_place_object_v1_train
```

Only if smoke PASS:

```bash
./01_train/run_train.sh /path/to/pick_place_object_v1_train
```


## Safety

These are offline training scripts only.
They do not import or use TB6-R5 RPC/Topic, Quest3, JogAnyJ3,
OmniPicker CAN, RealSense, PlaCo, or Pinocchio.
