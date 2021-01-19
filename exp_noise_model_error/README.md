# exp_noise_model_error

Code for the experiments comparing the theoretical, expected noise model error to the empirical measurements.

## Structure
* code: code for all experiments
* code/noise_estimation: shared files for noise estimation
* data: location for the datasets
* plots: location for the plots

## Installation
```
# Execute in this directory
conda create --name noise-matrix-estimation python=3.7
source activate noise-matrix-estimation
pip install numpy==1.18.1 scipy==1.4.1 matplotlib==3.1.2 Keras==2.3.1 tensorflow==2.1.0

# For the experiments with NoisyNER, the dataset needs to be obtained from
# https://github.com/uds-lsv/NoisyNER. Follow the instructions there and then
# copy the created files into the data/noisy-ner directory.

# For the experiments with Clothing1M, the dataset needs to be obtained from the original authors
# and then their "annotations" directory copied into the data/clothing1m directory.
``` 

## Running

The experiment files have the naming structure *exp_dataset_sampling.py* where dataset is *mnist*, *ner* or 
*clothing1m* and sampling is *fixed_ni* and *var_ni* (for Fixed or Variable Sampling). 
Each experiment file can be directly run from the code directory with python to obtain the corresponding plots. 

To run all, do

```
# Execute in this directory
source activate noise-matrix-estimation
cd code
python exp_mnist_syn-noise_fixed_ni.py
python exp_mnist_syn-noise_var_ni.py
python exp_clothing1m_fixed_ni.py
python exp_clothing1m_var_ni.py
python exp_ner_fixed_ni.py
python exp_ner_var_ni.py
```

The resulting plots are stored in the plots directory.
