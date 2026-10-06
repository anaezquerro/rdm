# **[RDM](): [R]()elational Abstractions for Spatial Reasoning with [D]()iffusion [M]()odels**

Hi👋! This is the implementation of the anonymous ICLR 2027 submission "*Relational Abstractions for Spatial Reasoning with Diffusion Models*".

## Requirements and installment

This code was tested in [Python 3.12.11](https://www.python.org/downloads/release/python-31211/), NVIDIA v580 and CUDA 13.0. All our models were trained with [PyTorch 2.9.1](https://pytorch.org/get-started/previous-versions/). See [requirements.txt](requirements.txt) to install the required libraries.

The source code requires setting to key environment variables at [src/env.py](src/env.py):

- `$RESULTS_FOLDER`: To store trained models. Our implementation will read from this folder the trained evaluators on some tasks.
- `$DATA_FOLDER`: To store and load datasets.

By default, these two folders are initialized on the user folder (`$HOME`) and RDM subfolder:

```python
# src/env.py
# ------------ key variables ----------------------
DATA_FOLDER = os.environ['HOME'] + '/rdm/datasets'
RESULTS_FOLDER = os.environ['HOME'] + '/rdm/results/'
```


Once these two variables are fixed, copy all the content from the folder [eval/](eval/) to `$RESULTS_FOLDER`:

```shell
cp -r eval/* $RESULTS_FOLDER
```

This will copy the trained evaluators to `$RESULTS_FOLDER`, and our implementation will directly read them.


## Data preparation

Our spatial reasoning benchmark consists of four datasets: (1) Akari, (2) Colored Sudoku, (3) Tangram and (4) LogicFace. See the following instructions to generate them and store them in `$DATA_FOLDER`.

1. **Akari dataset**: Akari samples are artificially generated in a $11 \times 11$ grid. Use the script [datasets/prepare-akari.py](datasets/prepare-akari.py) to generate the samples and splits. The training and evaluation sets will be automatically stored at [`$DATA_FOLDER`/akari](). 

```shell
PYTHONPATH=. python3 datasets/prepare-akari.py
```
2. ***Coldoku*: Colored Sudoku dataset**: We use the Sudoku dataset from [SRM](https://github.com/Chrixtar/SRM) ([datasets/sudokus.npy](datasets/sudokus.npy)), which needs to be stored in  [`$DATA_FOLDER`/sudoku]().
```shell
cp datasets/sudokus.npy $RESULTS_FOLDER/sudoku/
```

3. **Tangram dataset**: Download the SVG anotations of the Kilogram dataset ([Ji et al., 2022](https://aclanthology.org/2022.emnlp-main.38/)) in the [official repository](https://github.com/lil-lab/kilogram/tree/main/dataset/tangrams-svg) and place them in the folder [`$DATA_FOLDER`/tangram](datasets/kilogram/tangrams). Then, run the [datasets/prepare-tangram.py](datasets/prepare-tangram.py) script to generate the Tangram shuffles and split the dataset. 
```shell 
PYTHONPATH=. python3 datasets/prepare-tangram.py
```

4. **LogicFace dataset**: Download the CelebAMask-HQ dataset ([Lee et al., 2019](https://arxiv.org/abs/1907.11922)) from the [official repository](https://github.com/switchablenorms/CelebAMask-HQ) and place the image folder [CelebA-HQ-img]() and attribute annotations [CelebAMask-HQ-attribute-anno.txt]() at [`$DATA_FOLDER`/logicface](). Then, run the script at [datasets/prepare-logicface.py]() to create the dataset triplets and store all images in a single PyTorch file for fast training.

```shell
PYTHONPATH=. python3 datasets/prepare-logicface.py
```

The final result at `$DATA_FOLDER` should look like this:

```
$DATA_FOLDER/
    akari/
        high.pt
        low.pt
        test.pt
        train.pt
    logicface/
        CelebA-HQ-img/
        CelebAMask-HQ-attribute-annot.txt
        train.pkl
        test.pkl
    tangram/
        shuffles.npy
        test.npy
        train.npy
        kilogram/
    sudoku/
        train.pt
        test.pt 
```

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

To train RDM approaches it is required to first train the slot-autoencoder, and specify in the configuration file the same slot auto-encoder configuration and the path of the checkpoint. 

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












