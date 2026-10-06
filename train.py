from src.trainer import *
from src.task import *
from src.util import Config, cleanup, setup, is_distributed, WORLD_SIZE

import torch, os
from argparse import ArgumentParser

def load_transform(args) -> dict[str, list[Transform]]:
    transform = []
    for name, conf in args.conf.data.transform:
        transform.append(Transform(name, **conf))
    return transform 

def load_dataset(args, split: str) -> Dataset:
    args.conf.data.transform = load_transform(args)
    data = DATASET[args.data].from_folder(name=args.data, split=split, **args.conf.data)
    return data 

def train(args):
    trainer = TRAINERS[args.name].from_args(args, DATASET[args.data])
    if os.path.exists(f'{args.path}/last.pt') and args.load:
        print(f'Loaded trainer from {args.path}/last.pt')
        trainer.load_state_dict(f'{args.path}/last.pt')
    train = load_dataset(args, split='train')
    trainer.train(
        train, 
        path=args.path,
        **args.conf.train,
    )

if __name__ == '__main__':

    parser = ArgumentParser('Training mode')
    parser.add_argument('name', type=str, choices=TRAINERS.keys(), help='Model to train')
    parser.add_argument('-d', '--data', type=str, default='sudoku', help='Dataset selection')
    parser.add_argument('-p', '--path', type=str, help='Path to save the training results')
    parser.add_argument('-dt', '--dtype', type=str, default='bf16', choices=Trainer.DTYPE.keys(), help='Training precision')
    parser.add_argument('--load', action='store_true', default=False, help='Whether to load the weights from the path')
    args = parser.parse_args()

    # parse configuration file 
    args.conf = Config.from_yaml(f'{args.path}/config.yaml')
    os.makedirs(args.path, exist_ok=True)

    if is_distributed():
        args.device = setup()
        print(f'Activating distributed training in device {args.device}/{WORLD_SIZE}')
        train(args)
        cleanup()
    else:
        args.device = torch.device('cuda')
        train(args)