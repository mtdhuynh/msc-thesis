<p align="center">
  <a href="https://github.com/mtdhuynh/ms-thesis" alt="MS Thesis Repository">
    <img src="images/colon-logo.png" height="250">
  </a>
  <h1 align="center">Automated Colorectal Polyp Detection in Colonoscopy</h1>
</p>


<table align="center" style="background-color: rgba(0,0,0,0); border: none; background: transparent;">
  <tr style="background-color: rgba(0,0,0,0); border-collapse: none; background: transparent;">
    <td>In collaboration with:
  <tr style="background-color: rgba(0,0,0,0); border: none; background: transparent;">
    <!-- <td><a href="https://nearlab.polimi.it/medical/" alt="NEAR Lab Website" target="_blank">
  		<img src="images/near-lab-logo.jpg" alt="NEAR Lab Logo" width=130>
  	  </a></th> -->
    <td><a href="https://www.polimi.it/" alt="PoliMi Website" target="_blank">
		<img src="images/polimi-logo.png" alt="PoliMi Logo" width=350>
	  </a>
	  <a href="https://www.humanitas.it/" alt="Humanitas Website" target="_blank">
		<img src="images/humanitas-logo.svg" alt="Humanitas Logo" height=85>
	  </a>
</table>

<p align="center">
  <img src="https://img.shields.io/badge/Release-alpha-yellow.svg" />
</p>


Automated colorectal polyp detection using Deep Learning in colonoscopy-video images from the [SUN Dataset](http://sundatabase.org/). 

# Table of Contents
* [Introduction](#introduction)
* [Installation](#installation)
  1. [Download the SUN Dataset](#1-download-the-sun-dataset)
  2. [Install conda](#2-install-conda)
  3. [Clone the repository](#3-clone-the-repository)
  4. [Install the dependencies](#4-install-the-dependencies)
* [Repository Structure](#repository-structure)
* [Usage](#usage)
  1. [Dataset](#dataset)
  2. [Training](#training)
  3. [Inference](#inference)
* [Contacts & Acknowledgements](#contacts--acknowledgements)

# Introduction

Detection of colorectal polyps during colonoscopy procedures is a time-consuming, expensive, and attention-demanding task, requiring an expert's supervision.

The goal of the project is to benchmark state-of-the-art Object Detection models to correctly identify and locate such polyps in an automatised fashion, to speed up and improve their diagnosis. 

For this purpose, data from the [SUN Dataset](http://sundatabase.org/) will be used for training the models, and evaluation will be carried out against available commercial products. 

# Installation

### 1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### 2. Clone the repository
```
git clone https://github.com/mtdhuynh/ms-thesis.git
```

### 3. Install the dependencies
**Note**: if working from a Windows OS, the following commands must be run from the `Anaconda prompt`.

Ensure no previous conda environments exist:
```
cd ms-thesis
conda env remove --name ms-thesis
```

**[OPTIONAL]** Update `conda`:
```
conda update conda
```

Install the conda environment from the requirements file:
```
conda env create -f src/requirements.yml
```

Verify that the environment was installed correctly:
```
conda env list
```

Activate the conda environment:
```
conda activate ms-thesis
```

# Repository Structure

The repository has been structured as follows:
```
├───data
│   ├───00_zip
│   ├───01_raw
│   ├───02_intermediate
│   ├───03_primary
|   |   ├───images
|   |   └───labels
│   ├───04_model_input
|   |   ├───config
|   |   ├───train
|   |   └───val
│   ├───05_models
│   ├───06_model_output
|   |   └───runs
│   └───07_reporting
│
├───docs
├───images
├───notebooks
├───src
│   ├───data
│   ├───detection_models
|   |   └───model_zoo
│   ├───losses
│   ├───lr_schedulers
│   ├───metrics
│   ├───optimizers
│   ├───utilities
|   ├───main.py
|   └───requirements.yaml
|
├───slurm.sh
```


| **Folder** | **Description** | **Tags** |
|:---------------------|:----------------|:---------|
| `data` | Contains all the data, models, and model outputs needed and produced by the project. It follows a tree structure as defined in [this data engineering convention](https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71) to help building reproducible Data Science/Machine Learning pipelines. | `raw data`, `models`, `outputs` |
| `images` | Contains images, plots, icons, logos, etc. used in the repository's documentation (`README.md`, presentations, reports, etc.). | `documentation`, `images`, `icons`, `plots`, `logos` |
| `notebooks` | Contains `Jupyter` notebooks used for exploration, experiments, tests, etc. | `jupyter`, `experiments`, `tests` |
| `docs` | Contains documentation, slide decks, and reports regarding the project. | `documentation`, `reports` |
| `src` | Contains all the source code and scripts used in the project. | `python`, `pytorch`, `conda`, `code` |

This is only a high-level overview of the repository's structure at a glance. The interested reader should go ahead and inspect each single folder and file.

# Usage

## Dataset

### 1. Download the SUN Dataset

Follow the instructions at the bottom of the [SUN Dataset website](http://sundatabase.org/) to request access to the data.

Then, download the `zip` folders to [`data/00_zip`](./data/00_zip/). Make sure to have enough space on disk. The overall size of the SUN dataset is roughly 67GB. 


### 2. Extract the `zip` folders

After downloading is done, extract the `zip` folders into [`data/01_raw`](./data/01_raw/). 

You should end up with **101** folders:
- 100 `case` folders, containing 100 video sequences, each with a different polyp.
- 1 `annotation_txt` folder, containing the bounding box annotations for each image in each sequence/case.

Negative frames come from the first 13 cases. When unzipping the `zip` folders, make sure that the negative frames go in the corresponding `case` folders, where also positive frames reside.

### 3. Organize `data` folder

Open [`notebooks/data_management.ipynb`](./notebooks/data_management.ipynb) and run cell by cell.

This notebook takes care of organizing images and annotations in the corresponding `data` folders (according to the [data engineering convention](https://towardsdatascience.com/the-importance-of-layered-thinking-in-data-engineering-a09f685edc71)] mentioned above). 

Particularly, it will:
* Read SUN annotations and create custom annotation files (you can inspect the template here: [`data/02_intermediate/annotation_template.json`](./data/02_intermediate/annotation_template.json)). 
* Save/move **all** images and corresponding custom annotations to [`data/03_primary/images`](./data/03_primary/images/) and [`data/03_primary/labels`](./data/03_primary/labels/), respectively. A single [`labels.json`](./data/03_primary/labels.json) file containing all the annotations will be created too. 
* Split the dataset into `training` and `validation` split, on a per-class basis (i.e., both `train` and `val` split will contain the same proportion of class instances), and save the data splits in [`data/04_model_input/train`](./data/04_model_input/train/) and [`data/04_model_input/val`](./data/04_model_input/val/), respectively.


## Training

The following commands are used to train an object detection model of choice from scratch on the SUN dataset. 

### 1. Select a training configuration `yaml` file

In [`data/04_model_input/config`](./data/04_model_input/config/) you can find a few presets for training available models with out-of-the-box hyperparameters. 

If you wish to define your own configuration file, please do follow the structure of those `yaml` files.

### 2. Run `train.py`

Make sure you are in the `ms-thesis` project folder, and that the corresponding `conda` environment is active. Then, run:
```
python src/train.py
```

This will run the model training with default settings on the available device.

For a list of command-line arguments, please run:
```
python src/train.py --help
```

For example, to train with a custom configuration file, on the first `GPU` available, without logging and running `tensorboard` at the end of the training, run:
```
python src/train.py --config <path/to/custom/config_file.yaml> --device cuda:0 --run-tensorboard --no-verbose
```

Every other training hyperparameter should be edited in the config `yaml` file. 

### 3. Inspect results

Results from the training are saved by default to [`data/06_model_output/runs`](./data/06_model_output/runs/). The `runs` folder contains a folder for each different **model** (i.e., `arch` field in the configuration file). Each model folder contains each run's output in another unique folder (identified by `<YYYY-MM-DD_hh_mm_ss>_<SLURM_JOB_ID>`).

Each result folder contains:
* Output log file (`.log`).
* `tensorboard` run file (`events.out.tfevents.[...]`).
* Configuration file (`.yaml`).
* `models` folder, containing `checkpoint_<N>.pt` and `best_model.pt` (state dicts at epoch #N and at best loss/metric).

To inspect the results for each run, open a `tensorboard` session:
```
tensorboard --logdir data/06_model_output/runs
```
The command will prompt to a browser local window with the `tensorboard` UI.

For more information about `tensorboard`, see the [docs](https://www.tensorflow.org/tensorboard).

## Inference

**TBD**

# Contacts & Acknowledgements

This project is being carried out and developed as part of my Master's Thesis, in collaboration with the [Medical Robotics Section (MRSLab)](https://nearlab.polimi.it/medical/) of [PoliMi's NEARLab](https://www.deib.polimi.it/eng/deib-labs/details/67) and [Humanitas Research Hospital](https://www.humanitas.it/). 

If you have any further inquiries, questions, or would like any additional information, please feel free to contact any of the belowed-mentioned contacts.

| **Name** | **Position** | **Role** |
|:---------|:-------------|:---------|
| [Minh Tam Davide Huynh](https://www.linkedin.com/in/minh-tam-huynh/) | M.Sc. Biomedical Engineering Student @ Politecnico di Milano | Developer |
| [Alessandro Casella](https://nearlab.polimi.it/medical/alessandro-casella/) | Ph.D Biomedical Engineering Student @ Politecnico di Milano | Supervisor |
| [Prof. Elena De Momi](https://nearlab.polimi.it/medical/elenadem/) | Associate Professor in the Electronic, Information and Bioengineering Department (DEIB) @ Politecnico di Milano | Supervisor |