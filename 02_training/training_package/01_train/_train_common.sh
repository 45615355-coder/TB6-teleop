#!/usr/bin/env bash
# Common helpers for TB6-R5 ACT training.
# This file is sourced by run_smoke_train.sh and run_train.sh.

tb6_die() {
    echo "[FAIL] $*" >&2
    exit 1
}

tb6_require_file() {
    [[ -f "$1" ]] || tb6_die "Missing file: $1"
}

tb6_require_dir() {
    [[ -d "$1" ]] || tb6_die "Missing directory: $1"
}

tb6_validate_config() {
    : "${TASK_NAME:?TASK_NAME missing}"
    : "${POLICY_NAME:?POLICY_NAME missing}"
    : "${TRAIN_VERSION:?TRAIN_VERSION missing}"
    : "${DATASET_REPO_ID:?DATASET_REPO_ID missing}"

    [[ "$POLICY_NAME" == "act" ]] || tb6_die "Only ACT is supported by these scripts. Got: $POLICY_NAME"

    (( CHUNK_SIZE > 0 )) || tb6_die "CHUNK_SIZE must be > 0"
    (( N_ACTION_STEPS > 0 )) || tb6_die "N_ACTION_STEPS must be > 0"
    (( N_ACTION_STEPS <= CHUNK_SIZE )) || \
        tb6_die "N_ACTION_STEPS ($N_ACTION_STEPS) must be <= CHUNK_SIZE ($CHUNK_SIZE)"

    (( N_OBS_STEPS == 1 )) || tb6_die "LeRobot 0.6.1 ACT currently requires N_OBS_STEPS=1"
    (( STEPS > 0 )) || tb6_die "STEPS must be > 0"
    (( BATCH_SIZE > 0 )) || tb6_die "BATCH_SIZE must be > 0"
    (( NUM_WORKERS >= 0 )) || tb6_die "NUM_WORKERS must be >= 0"
    (( SAVE_FREQ >= 0 )) || tb6_die "SAVE_FREQ must be >= 0"
    (( LOG_FREQ > 0 )) || tb6_die "LOG_FREQ must be > 0"

    [[ "$USE_VAE" == "true" || "$USE_VAE" == "false" ]] || tb6_die "USE_VAE must be true/false"
    [[ "$IMAGE_TRANSFORMS_ENABLE" == "true" || "$IMAGE_TRANSFORMS_ENABLE" == "false" ]] || \
        tb6_die "IMAGE_TRANSFORMS_ENABLE must be true/false"
}

tb6_print_config() {
    echo "Task          : $TASK_NAME"
    echo "Policy        : $POLICY_NAME"
    echo "Version       : $TRAIN_VERSION"
    echo "Seed          : $SEED"
    echo "Steps         : $STEPS"
    echo "Batch size    : $BATCH_SIZE"
    echo "Num workers   : $NUM_WORKERS"
    echo "Save freq     : $SAVE_FREQ"
    echo "Chunk size    : $CHUNK_SIZE"
    echo "Action steps  : $N_ACTION_STEPS"
    echo "Backbone      : $VISION_BACKBONE"
    echo "Dim model     : $DIM_MODEL"
    echo "Encoder layers: $N_ENCODER_LAYERS"
    echo "Decoder layers: $N_DECODER_LAYERS"
    echo "VAE           : $USE_VAE"
    echo "KL weight     : $KL_WEIGHT"
    echo "LR            : $LEARNING_RATE"
    echo "Backbone LR   : $BACKBONE_LEARNING_RATE"
    echo "Weight decay  : $WEIGHT_DECAY"
    echo "Image aug     : $IMAGE_TRANSFORMS_ENABLE"
}

tb6_train_args() {
    local dataset_root="$1"
    local output_dir="$2"
    local job_name="$3"
    local steps="$4"
    local batch_size="$5"
    local num_workers="$6"
    local save_freq="$7"
    local log_freq="$8"

    TB6_ARGS=(
        --dataset.repo_id="$DATASET_REPO_ID"
        --dataset.root="$dataset_root"
        --dataset.video_backend=pyav
        --dataset.image_transforms.enable="$IMAGE_TRANSFORMS_ENABLE"

        --policy.type=act
        --policy.device=cuda
        --policy.push_to_hub="$PUSH_TO_HUB"

        --policy.n_obs_steps="$N_OBS_STEPS"
        --policy.chunk_size="$CHUNK_SIZE"
        --policy.n_action_steps="$N_ACTION_STEPS"

        --policy.vision_backbone="$VISION_BACKBONE"
        --policy.pretrained_backbone_weights="$PRETRAINED_BACKBONE_WEIGHTS"

        --policy.dim_model="$DIM_MODEL"
        --policy.n_heads="$N_HEADS"
        --policy.dim_feedforward="$DIM_FEEDFORWARD"
        --policy.n_encoder_layers="$N_ENCODER_LAYERS"
        --policy.n_decoder_layers="$N_DECODER_LAYERS"
        --policy.dropout="$DROPOUT"

        --policy.use_vae="$USE_VAE"
        --policy.latent_dim="$LATENT_DIM"
        --policy.n_vae_encoder_layers="$N_VAE_ENCODER_LAYERS"
        --policy.kl_weight="$KL_WEIGHT"

        --policy.optimizer_lr="$LEARNING_RATE"
        --policy.optimizer_lr_backbone="$BACKBONE_LEARNING_RATE"
        --policy.optimizer_weight_decay="$WEIGHT_DECAY"

        --output_dir="$output_dir"
        --job_name="$job_name"
        --steps="$steps"
        --batch_size="$batch_size"
        --num_workers="$num_workers"
        --log_freq="$log_freq"
        --save_checkpoint=true
        --save_freq="$save_freq"
        --env_eval_freq=0
        --eval_steps=0
        --wandb.enable="$WANDB_ENABLE"
        --seed="$SEED"
    )
}
