
# exp_estonian

Code for the experiments studying the relatioinship between theoritical noise matrix estimation error and models' test performance. The code in this directory is for the experiments on the NoisyNER dataset.

## [](#structure)Structure

-   code: code for running experiments on Estonian dataset
-   data: location for the datasets and fastText embedding
    - data/fastText: fastText pretrained embedding for Estonian
    - data/estner_diff_noise: all 7 noisy datasets (*estner_noisy_round01.conll-estner_noisy_round07.conll*) + 
    clean dataset (*estner_true.conll*)
-   estonian_outputs: location for results of the experiments

## [](#installation)Installation
### conda environment

```
# Execute in this directory
conda create --name estonian python=3.6
source activate estonian 
conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch
pip install numpy==1.17.5 scipy==1.4.1 matplotlib==3.1.2
```

### fastText
fastText needs to be installed to get the word embeddings used in training (please follow the installation instruction on [fastText github](https://github.com/facebookresearch/fastText)).
Also, download the fastText word embedding for Estonian from [fastText embedding](https://fasttext.cc/docs/en/crawl-vectors.html) 
(cc.et.300.bin.gz). Copy them into the data/fastText subdirectory.

### Data

You can get the NoisyNER dataset from https://github.com/uds-lsv/NoisyNER Follow the instructoins there.
Then rename NoisyNER_labelset{labeling_round}_all.tsv into estner_noisy_round0{labeling_round}.conll and estner_clean_train.tsv into estner_true.conll 

## [](#running)Running

### Preprocessing

First, we need to preprocess the data by ready text files from *data/estner_diff_noise*. 
This preprocess will save word embeddings for each word in the corpus and store it in a pickle file. In training the
model only needs to access the pickle files. 
To run preprocess, do

```
# Execute in this directory
source activate estonian
cd code
python preprocess_est.py
```

### Experiments

Second, to run the experiments as described in section 7, we need to run *run_experiments.py* with some arguments. 
In the following we give an example of arguments we used in the paper:

```
# Execute in this directory
source activate estonian
cd code
python3 run_experiments.py \
--output_root ../estonian_outputs \
--num_times 100 \
--batch_size 128 \
--exp_settings se_acc_scale \
--train_settings global_cm \
--random_seed 5555 \
--ns 100 1100 100 \
--uniform_sampling
```

The resulting plots are stored in the *estonian_outputs* directory.
