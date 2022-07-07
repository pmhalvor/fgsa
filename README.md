# FgFlex: A flexible multitasking sequence-labeler for fine-grained sentiment analysis

This repository contains the code developed for an experiment researching fine-grained sentiment analysis models for Norwegian text. Our thesis documenting the experimentation done on this code can be found [here](https://github.com/pmhalvor/fgsa/blob/master/FgFlex.pdf).


Using the IMN [(He et al. 2019)](https://github.com/ruidan/IMN-E2E-ABSA) and RACL [(Chen et al. 2020)](https://github.com/NLPWM-WHU/RACL) as baselines, we synthesis a our novel `FgFlex` model for easy experimentation on attention relations between the subtasks of such models. 


# Requirements
For reproducibility purposes, we outline the system requirements along with package dependencies needed to run this code.

## System requirements
Our Python runtime is 3.7.4.
Both Windows 10 and Linux Red Hat 8.5 operating systems were used under development and testing, ensuring cross-platform functionality.

### GPU
For faster training times, we recommend the use of a GPU node (especially for the larger models). 
Specifically, we made use of `cuda` interfaces, that are automatically enabled if `cuda` is detected through PyTorch.

## Dependencies
The machine learning framework we used for development was PyTorch.
Specific version of all the modules we used are presented below and in our `requirements.txt` file, for easier pip-installing.

| pip module | version |
|-|-|
|torch | 1.7.1 |
|numpy | 1.18.1 |
|transformers | 4.15.0 |
|nltk | 3.6.7 |
|pytest | 6.1.1 |
|pandas | 1.3.5 |
|sklearn| 1.0.2 |


# How to
There are three main ways to use this code: preprocessing, training single models, and studying hyperparameter configurations.

## Preprocess
In order to begin preprocessing step, make sure the [NoReC$_{fine}$](https://github.com/ltgoslo/norec_fine) data is downloaded to the same directory level this repository is cloned to. 
The output path to where the data will be written to can be configured in `src/config.py`.

Once the raw data is downloaded, the file `src/preprocess.py` can be run directly from the command-line. 
```
cd src/
python preprocess.py
```
This code restructures the NoReC$_{fine}$ data to the same IMN format used in both baselines.



## Training
To train a single model, with our best configurations, one can call the `src/train.py` file.
This file expects IMN data to be stored at the location specified in `src/config.py`.
Note, if you want to test this architecture on English data, you should also update the `BERT_PATH` in `src/config.py` to ensure an English BERT model is used to generate embeddings, instead of the default Norwegian, NorBERT2. 

Again, simple call the train file from the command-line:
```
python train.py
``` 
This will train a single instance of the `FgFlex` model from scratch, except for the pre-trained NorBERT2 embeddings. 
The final state of the model will be saved for later use at `../checkpoints/<model-name>.pt`.
The name of the model would need to be configured in top lines of the `train.py` file, to store different variations of this best model. 



## Experimenting
Our hand-made `Study` class can be used to test different hyperparameter configurations.
In addition to the preprocessed data, and correctly specified BERT paths, a `Study` requires a JSON configuration stored in the `studies/` directory.
For example, if we wanted to re-run the study on layers for the `FgFlex` model, you would call:

```python
python study.py fgflex/layers.json
```

The `.json` is not necessary, but is included here for consistency. 
Note, that the `studies/` directory prefix is not needed. 








