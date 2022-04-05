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
  2. [Install conda](#2-install-condahttpsdocscondaioprojectscondaenlatestuser-guideinstallindexhtml)
  3. [Clone the repository](#3-clone-the-repository)
  4. [Install the dependencies](#4-install-the-dependencies)
* [Usage](#usage)
* [Contacts & Acknowledgements](#contacts--acknowledgements)

# Introduction

Detection of colorectal polyps during colonoscopy procedures is a time-consuming, expensive, and attention-demanding task, requiring an expert's supervision.

The goal of the project is to benchmark state-of-the-art Object Detection models to correctly identify and locate such polyps in an automatised fashion, to speed up and improve their diagnosis. 

For this purpose, data from the [SUN Dataset](http://sundatabase.org/) will be used for training the models, and evaluation will be carried out against available commercial products. 

# Installation

### 1. Download the SUN Dataset

Follow the instructions at the bottom of the [SUN Dataset website](http://sundatabase.org/) to request access to the data.

### 2. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

### 3. Clone the repository
```
git clone https://github.com/mtdhuynh/ms-thesis.git
```

### 4. Install the dependencies
**Note**: if working from a Windows OS, the following commands must be run from the `Anaconda prompt`.

Ensure no previous conda environments exist:
```
cd ms-thesis
conda env remove --name ms-thesis
```

Install the conda environment from the requirements file:
```
conda env create -f src/requirements.yml
```

Verify that the environment was installed correctly:
```
conda env list
```

Activate the conda environmet:
```
conda activate ms-thesis
```

# Usage

# Contacts & Acknowledgements

This project is being carried out and developed as part of my Master's Thesis, in collaboration with the [Medical Robotics Section (MRSLab)](https://nearlab.polimi.it/medical/) of [PoliMi's NEARLab](https://www.deib.polimi.it/eng/deib-labs/details/67) and [Humanitas Research Hospital](https://www.humanitas.it/). 

If you have any further inquiries, questions, or would like any additional information, please feel free to contact any of the belowed-mentioned contacts.

| **Name** | **Position** | **Role** |
|:---------|:-------------|:---------|
| [Minh Tam Davide Huynh](https://www.linkedin.com/in/minh-tam-huynh/) | M.Sc. Biomedical Engineering Student @ Politecnico di Milano | Developer |
| [Alessandro Casella](https://nearlab.polimi.it/medical/alessandro-casella/) | Ph.D Biomedical Engineering Student @ Politecnico di Milano | Supervisor |
| [Prof. Elena De Momi](https://nearlab.polimi.it/medical/elenadem/) | Associate Professor in the Electronic, Information and Bioengineering Department (DEIB) @ Politecnico di Milano | Supervisor |
