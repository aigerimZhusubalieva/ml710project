#!/bin/bash
set -ex

# ------ Change the following to your settings ------

## Set the working directory
workdir=/home/mena.attia/Documents/ml710project

## Set the master and worker nodes (modify to your own nodes you allocated)
# NOTE: THE TRAINING ONLY WORKS ON ws-l4-xxx (except ws-l4-001) and ws-l6-xxx NODES
master_node=ws-l3-008
worker_node=ws-l1-017

# ------ End of changeable variables ------
BASE_PATH=$workdir
DATA_PATH=../imagenet_mini
DS_CONFIG=$workdir/ds_config.json
HOST_FILE=$workdir/hostfile
OUTPUT_DIR=$workdir/output/vgg16_output

# Hyperparameters for VGG16 training
GLOBAL_BATCH=64
MICRO_BATCH=32
ZERO_STAGE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

mkdir -p $OUTPUT_DIR

# Create the hostfile for DeepSpeed
cat <<EOT > $HOST_FILE
${master_node} slots=1
${worker_node} slots=1
EOT

# Create the DeepSpeed config file
cat <<EOT > $DS_CONFIG
{
  "train_batch_size": $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "gradient_accumulation_steps": 2,
  "steps_per_print": 10,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.0001
    }
  },
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": true,
    "initial_scale_power": 10
  },
  "wall_clock_breakdown": true
}
EOT

# Set environment variables for DeepSpeed
export NCCL_DEBUG=warn
export MAX_JOBS=10

# Start training
if [ "$(hostname)" == $master_node ]; then
    deepspeed --num_gpus 1 --num_nodes 2 --hostfile $HOST_FILE --no_ssh --node_rank=0 --master_addr $master_node --master_port=12345 train_vgg16_pipeline.py \
    --deepspeed \
    --deepspeed_config $DS_CONFIG \
    --epochs 10 \
    --batch-size $MICRO_BATCH \
    --lr 0.001 \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR
else
    deepspeed --num_gpus 1 --num_nodes 2 --hostfile $HOST_FILE --no_ssh --node_rank=1 --master_addr $master_node --master_port=12345 train_vgg16_pipeline.py \
    --deepspeed \
    --deepspeed_config $DS_CONFIG \
    --epochs 10 \
    --batch-size $MICRO_BATCH \
    --lr 0.001 \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR
fi