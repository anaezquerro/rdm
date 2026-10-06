import torch.distributed as dist
import torch, os 



def is_distributed() -> bool:
    return "LOCAL_RANK" in os.environ or "TORCHELASTIC_RUN_ID" in os.environ

WORLD_SIZE = int(os.environ['WORLD_SIZE']) if is_distributed() else 1

def local_rank() -> int:
    return int(os.environ['LOCAL_RANK'])

def is_main():
    if is_distributed():
        return local_rank() == 0 
    else:
        return True 
    
def sync(x: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(x, op=op)
    return x

def setup():
    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    return torch.device(device)

def cleanup():
    dist.destroy_process_group()