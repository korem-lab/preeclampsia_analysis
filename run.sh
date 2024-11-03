### code to run all analyses
## running this script will populate the `results` folder with all figure panels

## Immune factor associations
cd code/immune-factor-associations
python run_IF_analysis.py


## Microbe associations
cd ../microbe-associations
python run-association-analyses.py


## multiomic associations
cd ../multiomic-associations
python 01-set-up-analyses.py
# Rscript 02-run-diablo.R
python 03-make_networkx_plots.py

## prediction analyses
cd ../predictions
python 01-run-multilevel-predictions.py
python 02-plot_predictions.py
python 03-cross-study-evaluation.py

