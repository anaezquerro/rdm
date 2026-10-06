# **[RDM](): [R]()elational Abstractions for Spatial Reasoning with [D]()iffusion [M]()odels**

Hi👋 This is the implementation of our preprint: *Relational Abstractions for Spatial Reasoning with Diffusion Models*!

## Requirements and installation

This code was tested in [Python 3.12.11](https://www.python.org/downloads/release/python-31211/), NVIDIA v580 and CUDA 13.0. All our models were trained with [PyTorch 2.9.1](https://pytorch.org/get-started/previous-versions/). See [requirements.txt](requirements.txt) to install the required libraries.

The source code requires setting three key environment variables at [src/env.py](src/env.py):

- `$RESULTS_FOLDER`: To store trained models. 
- `$DATA_FOLDER`: To store and load datasets.
- `$EVAL_FOLDER`: To store trained evaluators (only for Sudoku, LogicFace and Counting Polygons).

By default, these three folders are initialized inside the project folder:

```python
# src/env.py
# -- environment variables (change if needed) ---
DATA_FOLDER = 'datasets'
RESULTS_FOLDER = 'results'
EVAL_FOLDER = 'eval'
```

Follow these steps to reproduce and use our code:

1. Download the trained evaluators from [eval.zip](https://drive.google.com/file/d/1JqDkG6JEhadCJUd6WtDE2hu_NYHNeGnF/view?usp=sharing) and move all its content to the selected `$EVAL_FOLDER`.
2. The repository already has lightweight files in [datasets/](datasets) from which more samples or different train/test splits can be obtained. Large files (with exact train/test splits) are available in [datasets.zip](https://drive.google.com/file/d/17mNODwYlJfHsf1-UZ2ummfBU2qljphec/view?usp=drive_link). Unzip the file and move all its content to the selected `$DATA_FOLDER`.
    - For Coldoku, we use [SRM](https://github.com/Chrixtar/SRM) original MNIST Sudoku dataset. Download the *mnist_sudoku.npy* and save it in `$DATA_FOLDER`.
    - For Counting Polygons, we also use the *circle_position_radius.npy* file. Download it and save it in `$DATA_FOLDER`.
3. We provide the pretrained slot autoencoders from our paper, so only diffusion models need to be trained again. Download the file [results.zip](https://drive.google.com/file/d/1XSPog7EWc_ZnaOE_nicGzf8yfdBiyZRe/view?usp=sharing) and move all its content to the selected `$RESULTS_FOLDER`. 


## Training 

Our models are trained from the script [train.py](train.py) by setting the approach as first argument, and specifying the dataset, folder to store results and denoiser architecture. For instance, the following command runs the baseline model for Akari with the U-Net denoiser:

```shell
python3 train.py base -d akari -p results/akari/baseline  --load 
```

**Arguments**: 
- `name`: Positional argument (required), specifies the approach: `sae` for the slot-autoencoder, `base` for the diffusion baseline, `abs` for the RDM approach.
- `--data` (`-d`): Dataset to use (`akari`, `sudoku`, `tangram`, `logicface`).
- `--path` (`-p`): Folder to store the results. Note that this folder needs to contain inside the YAML file (named as [config.yaml]()) with the model and training configuration..
- `--load`: Optional argument. If specified, it will load the last checkpoint stored in `path`.

**Configuration files**: We included the configuration files used to train the models of our paper in the [config/](config/) folder. Some parameters might need to be modified to adjust hardware specifications (e.g. training batch-size) and load slot auto-encoder weights. Inside the folder [config/](config/) there are four subfolders, one per task, each containing 5 or 6 configuration files:

- [baseline.yaml](): Configuration file to train the baseline.
- [slot-k](): Configuration file to train the slot-autoencoder with $k$ slots. 
- [clue.yaml](): Configuration file to train RDM with the *clue* approach (no CFG, only condition on solutions).
- [solution.yaml](): Configuration file to train RDM with the *solution* approach (CFG).
- [full.yaml](): Configuration file to train RDM with the *full* approach.


**Compatibility with DDP**: This code supports [Data Distributed Parallel (DDP)](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) training by simply running the [train.py](train.py) script with the desired [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html) specifications. For example:

```shell
torchrun --nproc-per-node=2 train.py base -d akari -p results/akari/baseline --load
```

It is *not* necessary to configure the batch size in the configuration file: our implementation will manage the call to the optimizer to keep the batch-size stable.


## Evaluation 

Once the model is trained, the last checkpoint (`last.pt`) is stored in the folder specified as argument. Evaluation is executed from files [eval.py](eval.py) or [predict.py](predict.py). Use [eval.py](eval.py) for fast evaluation and to obtain the final metric; or [predict.py](predict.py) to also store the generated images. 

The example below runs the evaluation on the test set of Akari using 100 sampling steps:

```shell
python3 eval.py abs -d akari -p results/akari/baseline/last.pt -s test --num-steps 100
```


## Data Generation

We have the original scripts to automatically generate Akari, Tangram and LogicFace samples. Do not hesitate to [contact us](mailto:ana.ezquerro@tugraz.at) if you are interested in augmenting our benchmark with more samples or different configurations (e.g. varying Akari layouts, increasing Tangram colors). 









