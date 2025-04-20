set -ex

## Set the working directory
workdir=/home/mena.attia/Documents/ml710project/DDP

BASE_PATH=$workdir
DATA_PATH=../imagenet_mini
DS_CONFIG=$workdir/DDP/ds_config.json
HOST_FILE=$workdir/DDP/hostfile
OUTPUT_DIR=$workdir/DDP/output/vgg16_output


GLOBAL_BATCH=128
MICRO_BATCH=64

master_node=ws-l6-001
worker_node=ws-l6-017

MASTER_PORT=29502
export MASTER_PORT=$MASTER_PORT

HOST_FILE=$workdir/hostfile

# Create the hostfile
cat <<EOT > $HOST_FILE
${master_node} slots=1
${worker_node} slots=1
EOT


# run this command to get the IP address
# hostname -I | awk '{print $1}' 
if [ "$(hostname)" == $master_node ]; then
  echo "Running on worker node ($(hostname))"
  torchrun   --nproc_per_node=1   --nnodes=2   --node_rank=0   --rdzv_backend=c10d --rdzv_endpoint=10.127.30.120:$MASTER_PORT  train.py --batch_size $MICRO_BATCH \
    --data_path $DATA_PATH

else
  echo "Running on worker node ($(hostname))"
  torchrun   --nproc_per_node=1   --nnodes=2   --node_rank=1   --rdzv_backend=c10d --rdzv_endpoint=10.127.30.120:$MASTER_PORT train.py --batch_size $MICRO_BATCH \
    --data_path $DATA_PATH 
fi