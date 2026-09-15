#!/usr/bin/env bash
# ============================================================
# TB6-R5 + OmniPicker
# Task: pick_place_object_v1
# Policy: ACT
# Training configuration: v1
#
# This is the ONLY file normally edited when creating a new
# training-parameter version. For a real parameter iteration,
# copy v1.sh -> v2.sh and change only the intended parameters.
# ============================================================

# ---------- Experiment identity ----------
TASK_NAME="pick_place_object_v1"
POLICY_NAME="act"
TRAIN_VERSION="v1"
DATASET_REPO_ID="local/tb6_r5_pick_place_object_v1_train"

# ---------- Reproducibility ----------
SEED=1000

# ---------- Training process ----------
STEPS=100000
BATCH_SIZE=8
NUM_WORKERS=4
SAVE_FREQ=20000
LOG_FREQ=200

# ---------- ACT action chunk ----------
# Dataset is 50 Hz.
# CHUNK_SIZE=100 means the policy predicts 100 future actions = 2.0 s.
CHUNK_SIZE=100
N_ACTION_STEPS=100
N_OBS_STEPS=1

# ---------- Vision backbone ----------
VISION_BACKBONE="resnet18"
PRETRAINED_BACKBONE_WEIGHTS="ResNet18_Weights.IMAGENET1K_V1"

# ---------- ACT Transformer ----------
DIM_MODEL=512
N_HEADS=8
DIM_FEEDFORWARD=3200
N_ENCODER_LAYERS=4
N_DECODER_LAYERS=1
DROPOUT=0.1

# ---------- ACT VAE ----------
USE_VAE=true
LATENT_DIM=32
N_VAE_ENCODER_LAYERS=4
KL_WEIGHT=10.0

# ---------- Optimizer preset ----------
LEARNING_RATE=1e-5
BACKBONE_LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-4

# ---------- Dataset augmentation ----------
# Baseline v1: disabled. If later enabled, create a new training version.
IMAGE_TRANSFORMS_ENABLE=false

# ---------- External services ----------
PUSH_TO_HUB=false
WANDB_ENABLE=false
