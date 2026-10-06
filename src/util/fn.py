import torch, random, os
import numpy as np 

def nunique(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return (x.sort(dim=dim)[0].diff(dim=dim) > 0).sum(dim) + 1

    
def to(device: torch.device, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: x.to(device) if isinstance(x, torch.Tensor) else x for name, x in tensors.items()}

def acc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    if preds.ndim > targets.ndim:
        preds = preds.argmax(-1)
    return float((preds == targets).sum()/len(targets)*100)


def avg(x) -> float:
    total, n = 0, 0
    for i in x:
        total += i 
        n += 1
    return total/n

def folderpath(path: str) -> str:
    return '/'.join(path.split('/')[:-1])

def filename(path: str, extension: bool = True) -> str:
    name = path.split('/')[-1]
    if not extension:
        name = '.'.join(name.split('.')[:-1])
    return name 

def max_value_ratio(x: torch.Tensor, dim: int = None) -> tuple[torch.Tensor, float]:
    """Returns the maximum value of a tensor and its ratio."""
    values, counts = x.unique(sorted=True, dim=dim, return_counts=True)
    return values[counts.argmax()], counts.max()/counts.sum()

def flatten(items, depth: int = -1):
    res = []
    for item in items:
        if not any(isinstance(item, typ) for typ in [int, float, str]) and depth != 0:
            res += flatten(item, depth-1)
        else:
            res.append(item)
    return res 
            
            
def is_integer(x: torch.Tensor) -> bool:
    return torch.equal(x, x.round())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
def random_range(start: int, end: int) -> list: 
    values = list(range(start, end+1))
    random.shuffle(values)
    return values


def shuffle(items) -> list:
    items = list(items).copy() # ensure a copy
    random.shuffle(items)
    return items


def split(x: torch.Tensor, chunks: list[int], dim: int = -1) -> list[torch.Tensor]:
    for i, c in enumerate(chunks):
        if c == -1:
            chunks[i] = x.size(dim) + 1 - sum(chunks)
            break 
    print(x.shape, chunks, )
    return x.split(chunks, dim=dim)