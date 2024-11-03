# Code repository for analysis of preeclampsia
This repository contains all code and results supporting the analysis of our manuscript titled 'First trimester vaginal microbes and immune factors are early predictors of preeclampsia.' 

Outline of the Repository
------------------
| Folder/file | Description |
|--|--|
| `code/` | Contains code to run all analyses (reads from `data/`, writes to `results`/) |
| `code/immune-factor-associations` | analyses of immune factor associations (Fig. 2) |
| `code/microbe-associations` | analyses of microbe associations (Fig. 3) |
| `code/multiomic-associations` | analysis of immune factor associations (Fig. 4)|
| `code/predictions` | Contains code to run all prediciton analysies (Fig. 5)|
| `data/` | Placeholder to directory containing all input data used to run analyses |
| `results/` | Contains all figures images |
| `results/immune-factor-associations` | results of immune factor associations (Fig. 2) |
| `results/microbe-associations` | results of microbe associations (Fig. 3) |
| `results/multiomic-associations` | results of immune factor associations (Fig. 4)|
| `results/predictions` | results for all prediciton analysies (Fig. 5)|
| `preprocessing/` | Contains commands used for preprocessing & quantification of metagenomic samples |
| `run.sh` | Script to run all analyses |

How to run analyses
------------------
All analyses can be executed by executing the command `bash run.sh` from this repository's home directory. Running this script will populate all folders in the `results` directory with the files
