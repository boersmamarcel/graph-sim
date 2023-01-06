# Graph similarity
This repository contains code to reproduce the results on the synthetic and public datasets.

# Installation
Here we share an installation guide for Mac OS X and Ubuntu. 
Note that, due to a bug, AutoML will not work in Mac OS X (https://github.com/automl/auto-sklearn/issues/1591).

## Ubuntu
First create a python environment
```
conda create -n gsim python=3.7
```
and activate it
```
conda activate gsim
```
then install all the required packages
```
pip install -r requirements.txt
```
make sure to install tensorflow separately, we use this to generate the tensorboards which display the results
```
pip install tensorflow
```
and finally, we need to install the POT library to calculate the Wasserstein distance.
Optionally, to speed things up, you can install pathos for multi-core processing.

```
pip install --user -U https://github.com/PythonOT/POT/archive/master.zip
pip install pathos
pip install p_tqdm
```

### Mac
Please follow the installation steps for Ubuntu, but install Tensorflow as follows.

Make sure to install tensorflow, for M1 macs run:
```
python -m pip install tensorflow-macos
pip install tensorflow-metal
```
Then install all other requirements:

Run `pip install -r requirements.txt`

The current version of POT, 0.7, does not yet include some important bug fixes. Using the latest version is therefore recommended. This is done via 
`pip install --user -U https://github.com/PythonOT/POT/archive/master.zip` 

If you want to calculate the Wasserstein distances in parallel, you need the Pathos package: `pip install pathos`, or for a pretty progress bar p_tqdm: `pip install p_tqdm` which uses Pathos in the background. 

## Automl
For Mac OS X we have the following bug:
https://github.com/automl/auto-sklearn/issues/1591


# Run experiments
We included various .sh files, these list the experiments you could do.

* experiments_public.sh: run the experiments on the public dataset with svm classifier
* experiments_public_automl.sh: see above, but we use automl 
* experiments_public_random.sh: we use a random set of reference graphs
* experiments_synthetic_different_classes.sh: classifies different classes
* experiments_synthetic_same_class.sh: classifies graphs from the same class, but with different settings

You can define your own experiments, execute the following command for guidance
```
python experiments.py -h 
```
to get the help menu, or modify the below example command
```
python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip --online -i 8 -r 128 --categorical --model wwlr --n_jobs 12 --gridsearch
```

## Generate fake data
To validate our results, we use publicly available dataset and synthetic data.
Here we describe how to generate the synthetic data.

The following command stores the generated datasets in the `data/` folder
```
python generate_fake_data.py
```



