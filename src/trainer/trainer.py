from __future__ import annotations
import torch, os, logging
from tqdm import tqdm 
from torch import nn 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader

from src.trainer.opt import Optimizer
from src.trainer.dataset import Dataset
from src.trainer.metric import Metric
from src.util import to, is_main, is_distributed, logger, flatten, local_rank, WORLD_SIZE, seed_worker
from src.env import TQDM_DISABLE, DETERMINISTIC, SEED, PARALLELISM, DEBUG


class Trainer:
    DATASET: Dataset
    DTYPE= dict(f32=torch.float32, bf16=torch.bfloat16)
    
    def __init__(
        self, 
        model: nn.Module, 
        opt: Optimizer,
        device: torch.device | int,
        dtype: torch.dtype | str
    ):
        if not DEBUG:
            model = torch.compile(model)
        if is_distributed():
            self.model = DDP(model.to(device), device_ids=[device], output_device=device)
        else:
            self.model = model.to(device)
        self.opt = opt
        self.device = torch.device(device) if isinstance(device, int) or isinstance(device, str) else device
        self.dtype = self.DTYPE[dtype] if isinstance(dtype, str) else dtype
    
    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.raw_model.parameters())

    def start_logger(self, name: str, path: str | None = None, level = logging.DEBUG):
        if is_main():
            self.log = logger(name, path=path, level=level)
    
    def info(self, text: str):
        if is_main():
            self.log.info(text)
            
    def debug(self, text: str):
        if is_main():
            self.log.debug(text)
            
    def warn(self, text: str):
        if is_main():
            self.log.warning(text)
    
    def to(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return to(self.device, batch)

    def collate(self, samples: list[dict[str, torch.Tensor | int]]) -> dict[str, torch.Tensor]:
        batch = dict()
        keys = sorted(set(flatten(s.keys() for s in samples)))
        for key in keys:
            values = [sample[key] for sample in samples]
            if isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            elif isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            else:
                batch[key] = values 
        return batch
    
    def loader(self, data: Dataset, batch_size: int, shuffle: bool = False):
        args = dict(batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate)
        if PARALLELISM:
            args |= dict(
                batch_size=batch_size, 
                num_workers=4, 
                persistent_workers=True, 
                pin_memory=True,
                prefetch_factor=2,
                multiprocessing_context='fork'
            )
        if DETERMINISTIC:
            g = torch.Generator()
            g.manual_seed(SEED)
            args |= dict(worker_init_fn=seed_worker, generator=g)
        if is_distributed():
            args['sampler'] = DistributedSampler(data, num_replicas=WORLD_SIZE, rank=local_rank(), shuffle=args.pop('shuffle'), drop_last=True)
        return DataLoader(data, **args)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(opt={self.opt}, device={self.device}, dtype={self.dtype})'
    
    def train(
        self,
        train: Dataset,
        path: str,
        max_steps: int,
        batch_size: int,
        interval: int = 5000
    ):
        os.makedirs(path, exist_ok=True)
        self.model.train()

        # log information
        exists = os.path.exists(f'{path}/train.log')
        self.start_logger('train', path=f'{path}/train.log')

        # update number of samples depending on batch size and number of steps 
        train.n = int(self.opt.batch_size*max_steps)
        loader = self.loader(train, batch_size=batch_size, shuffle=True)

        if not exists:
            self.info(self.model)
            self.info(f'Saving training at {path}\n' + '\n'.join(f'{name} = {repr(value)}' for name, value in locals().items() if name != 'kwargs'))
        print(f'Number of parameters: {self.num_params}')
        print(f'Number of trainable parameters: {self.num_trainable_params}')
        with tqdm(desc=f'train', total=len(loader), disable=not is_main(), mininterval=30.0) as bar:
            for batch in loader:
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    loss = self.train_step(**self.to(batch))
                self.opt(
                    loss=sum(loss.values()), 
                    model=self.raw_model, 
                    num_samples=batch_size,
                )
                bar.set_postfix_str(', '.join(f'{name}={x:.5f}' for name, x in (loss | self.opt.status).items()))
                bar.set_description_str(f'[{self.opt.step}/{max_steps}]')
                bar.update(1)

                if (self.opt.step + 1) % interval == 0:
                    self.save(f'{path}/last.pt')
                    slope = self.opt.slope(interval)
                    self.info(f'Saving checkpoint: step={self.opt.step}, loss={self.opt.loss(interval)}, slope={slope}')
                    if slope < 0:
                        self.save(f'{path}/improved.pt')
                if (self.opt.step + 1) % 50000 == 0: # save every 50k steps 
                    self.save(f'{path}/{int(self.opt.step)}.pt')

        self.info(f'Finished training')
        
    def evaluate(
        self, 
        data: Dataset,
        batch_size: int,
        path: str | None = None,
        **kwargs
    ) -> Metric:
        self.model.eval()
        data.training = False
        print(f'Number of parameters: {self.num_params}')
        print(f'Number of trainable parameters: {self.num_trainable_params}')
        if path:
            os.makedirs(path, exist_ok=True)
        if self.opt.ema is not None:
            self.opt.ema.broadcast(self.raw_model)
        loader = self.loader(data, batch_size=batch_size, shuffle=False)
        metric = data.METRIC.to(self.device)
        with tqdm(desc='eval', leave=False, total=len(loader), disable=TQDM_DISABLE or not is_main()) as bar:
            for step, batch in enumerate(loader):
                batch = self.to(batch)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    pred = self.eval_step(**batch, **kwargs)
                if path:
                    metric(step, path, **pred, **batch)
                else:
                    metric(**pred, **batch)
                bar.set_postfix_str(repr(metric))
                bar.update(1)
        return metric.sync()

    def predict(
        self, 
        data: Dataset,
        batch_size: int,
        path: str, 
        **kwargs
    ) -> Metric:
        self.model.eval()
        data.training = False 
        if self.opt.ema is not None:
            self.opt.ema.broadcast(self.raw_model)
        loader = self.loader(data, batch_size=batch_size, shuffle=False)
        if path:
            os.makedirs(f'{path}/eval', exist_ok=True)
            os.makedirs(f'{path}/pred', exist_ok=True)
        with tqdm(desc='predict', leave=False, total=len(loader), disable=TQDM_DISABLE or not is_main()) as bar:
            for i, batch in enumerate(loader):
                batch = self.to(batch)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    pred = self.pred_step(**batch, **kwargs)
                self.save_pred(step=i, path=f'{path}/pred', **pred, **batch, transform=data.transform)
                bar.update(1)

    @property
    def raw_model(self) -> nn.Module:
        model = self.model 
        if isinstance(model, DDP):
            model = model.module 
        if model.__class__.__name__ == 'OptimizedModule':
            model = model._orig_mod
        return model
                
    def load_state_dict(self, path: str) -> Trainer:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(state['model'])
        self.opt.load_state_dict(state['opt'])
        return self
        
    def save(self, path: str):
        if is_main():
            model_state = self.raw_model.state_dict()
            state = dict(model=model_state, opt=self.opt.state)
            torch.save(state, path)
                
    @property
    def METRIC(self) -> Metric:
        raise NotImplementedError
            
    def train_step(self, **kwargs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError
    
    @torch.no_grad()
    def pred_step(self, **kwargs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError
    
    @torch.no_grad()
    def eval_step(self, **kwargs) -> dict[str, torch.Tensor]:
        return self.pred_step(**kwargs)

    def save_pred(self, *args, **kwargs):
        ...
            