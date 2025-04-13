import torch
import torch.distributed as dist
import argparse
import datetime
import os
import deepspeed

def init_process(rank, size, fn, backend='nccl'):
    # Initialize the distributed process group.
    dist.init_process_group(backend=backend, rank=rank, world_size=size, timeout=datetime.timedelta(seconds=300))
    print(f"[Rank {rank}] Initialized the process group.")
    fn(rank, size)

def send_tensor(rank, size):
    # Rank 0 sends a tensor, Rank 1 receives it
    if rank == 0:
        tensor = torch.ones(4, dtype=torch.float32, device="cuda:0")  # A simple tensor
        print(f"[Rank 0] Sending tensor: {tensor}")
        dist.send(tensor, dst=1)  # Send tensor to rank 1
        print("[Rank 0] Tensor sent.")
    elif rank == 1:
        recv_tensor = torch.empty(4, dtype=torch.float32, device="cuda:0")  # Empty tensor to receive
        dist.recv(recv_tensor, src=0)  # Receive tensor from rank 0
        print(f"[Rank 1] Received tensor: {recv_tensor}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    # Ensure local_rank is provided (DeepSpeed launcher sets this)
    parser.add_argument("--local_rank", type=int, default=0)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Get the local rank for the current process.
    rank = int(os.environ["RANK"])
    size = int(os.environ["WORLD_SIZE"])
    print(rank)


    # Run the dummy send/recv process with initialization
    init_process(rank, size, send_tensor)

if __name__ == "__main__":
    main()
