from src.trainer import *
from src.task import *
from src.util import set_seed, Config
import src.env as env
from train import load_dataset

import torch, os
from argparse import ArgumentParser

def predict(args):
    trainer = TRAINERS[args.name].from_args(args, DATASET[args.data])
    trainer.load_state_dict(args.path)
    folder = os.path.dirname(args.path)
    if args.level: 
        args.conf.data.level = args.level
    data = load_dataset(args, args.split)
    path = f'{folder}/{args.num_steps}-steps-{args.split}-{args.sampling}'
    metric = trainer.predict(
        data, 
        path=path,
        **args.conf.predict, 
        sampling=args.sampling,
        num_steps=args.num_steps,
    )
    metric.save(f'{path}.mt')

if __name__ == '__main__':
    parser = ArgumentParser('Evaluation mode')
    parser.add_argument('name', type=str, choices=TRAINERS.keys(), help='Model to train')
    parser.add_argument('-d', '--data', type=str, default='sudoku', help='Dataset selection')
    parser.add_argument('-p', '--path', type=str, help='Path to save the training results')
    parser.add_argument('-c', '--conf', type=str, default='config', help='Configuration file')
    parser.add_argument('--sampling', type=str, default='parallel', help='Sampling mode')
    parser.add_argument('-s', '--split', type=str, default='test', help='Evaluation set')
    parser.add_argument('--num-steps', default=100, type=int, help='Number of denoising steps')
    parser.add_argument('--level', default=None, help='Sudoku level', type=int, nargs='+')
    parser.add_argument('-dt', '--dtype', type=str, default='f32', choices=Trainer.DTYPE.keys(), help='Training precision')
    args = parser.parse_args()
    set_seed(env.SEED)
    args.conf = Config.from_yaml(f'{os.path.dirname(args.path)}/{args.conf}.yaml')
    args.device = torch.device('cuda')
    metric = predict(args)
