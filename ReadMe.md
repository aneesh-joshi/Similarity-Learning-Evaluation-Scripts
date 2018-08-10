# Similarity Learning Evaluation Scripts and Dataset
This repo contains:
- code to evaluate and benchmark some similarity learning models
- data links and data readers for some similarity learning datasets
- code for baselines on these datasets
- a report with a detailed study of similarity learning

Please refer to [Report.md](Report.md) for the detailed study which explains what Similarity Learning is, the tasks that it involves and different models used for it. (Highly recommended to understand current repo)

## Getting Started

You can just run the `download_and_setup.sh` script to do everything I will describe below.

Also, heck out [this notebook](SLEvalDownloadExample.ipynb) which does everything from cloning this repo to setting things up (no training)

### Downloading Datasets
We have several datasets which can be used to evaluate your Similarity Learning model. All you have to do is:

	cd data/
	python get_data.py

This will download:
- WikiQA Dataset
- SNLI
- SICK

Additionally, we need the Quora Duplicate Questions Dataset and Glove Embedding vectors. Luckily, they are available in [gensim-data](https://github.com/RaRe-Technologies/gensim-data) and will be automagically downloaded when needed. You need to have gensim for this. gensim and other dependencies can be installed easily with:

	python setup.py

For getting the SQUAD-T dataset, you will have to convert the existing SQUAD dataset. For this, use the [misc_scripts/squad2QA.py](https://github.com/aneesh-joshi/Similarity-Learning-Evaluation-Scripts/blob/master/misc_scripts/squad2QA.py).

	python squad2QA.py --squad_path ../data/train-v1.1.json

`train-v1.1.json` is the squad file with span level data which should be downloaded by `get_data.py`
The QA dataset will be saved in the `data` folder and accessed from there by the `eval_wikiqa`

Additionally, you also need the `trec_eval` binary for evaluating QA datasets. You can get it easily from `misc_scripts/get_trec.py`by:

	python get_trec.py

This will clone the trec repo and make the trec binary. (Must have C compiler!)  
The `trec_eval` binary exists in the trec folder. Use this binary to evaluate qrels* and pred* created by evaluation scripts (for WikiQA and InsuranceQA). Please read the docs in [eval_wikiqa.py](https://github.com/aneesh-joshi/Similarity-Learning-Evaluation-Scripts/blob/master/evaluation_scripts/WikiQA/eval_wikiqa.py)

### Running Evaluations
The models:  
- DRMM TKS
- MatchPyramid
- BiDirectional Attention Flow - T (senTence-level)  
can be found in the folder `sl_eval/models/`

The folder `evaluation_scripts/` contains files and folders for running these models on different datasets.

You can run each model by executing their script. So, to evaluate SICK:

	python eval_sick.py

For datasets like SICK, SNLI, Quora Duplicate Questions : The result will be printed in the terminal
For WikiQA and InsuranceQA : The result is saved as a qrels and a pred file which can be evaluated using `trec_eval` binary

Since there is a random seed set, the number of threads is limited to 1. This reduces processing speeds.
Ideally, you should run scripts like :

	python eval_sick.py > eval_sick_train_log.txt

This will save the outputs in the txt which can be read later.

Waiting for the running all evaluations can get tedious. For such cases, I have included the [evaluation_scripts/train_all.sh](https://github.com/aneesh-joshi/Similarity-Learning-Evaluation-Scripts/blob/master/evaluation_scripts/runall.sh) script. Run it like:

	./train_all.sh > train_log

and let it run. Check the logs for accuracy scores. run trec_eval for the others on the saved pred and qrels files.

### About folders
- **data_readers:** contains readers for the different datasets. You can even use them independently of this repo.
- **evaluation_scripts:** scripts to evaluate models on different datasets.
- **data:** has the `get_data.py` script and holds most of your datasets after downloading them.
- **sl_eval:** contains the model implementations.
- **misc_scripts:** holds misc scripts which have different uses
- **_images:** images for the final report (ignore it)
- **old_stuff:** legacy content which should really be phased out!


**Notes:**
- Make sure you take a look at the Issues for contribution ideas and for getting an idea of current ideas.
- Read the docs in the scripts. They are mostly well documented ans should make understanding things a lot easier.
