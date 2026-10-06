import os

DATA_FOLDER = os.environ.get('DATA_FOLDER', 'datasets')
RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER', 'results')
EVAL_FOLDER = os.environ.get('RESULTS_FOLDER', 'eval')
TQDM_DISABLE = os.environ.get('TQDM_DISABLE', False)
DETERMINISTIC = os.environ.get('DETERMINISTIC', False)
SEED = int(os.environ.get('SEED', 123))
FONT = f'datasets/Roboto-Regular.ttf'
PARALLELISM = int(os.environ.get('PARALLELISM', True))
DEBUG = bool(int(os.environ.get('DEBUG', False)))