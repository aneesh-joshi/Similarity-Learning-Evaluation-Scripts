# Similarity Learning Evaluation Scripts and Dataset
This repo contains:
- code to evaluate and benchmark some similarity learning models
- data links and data readers for some similarity learning datasets
- code for baselines on these datasets
- a report with a detailed study of similarity learning

Please refer to [Report.md](TODO) for the detailed study which explains what Similarity Learning is, the tasks that it involves and different models used for.

## Getting Started
### Downloading Datasets
We have several datasets which can be used to evaluate your Similarity Learning model. All you have to do is:

	cd data/
	python get_data.py

This will download:
- WikiQA Dataset
- SNLI
- SICK

Additionally, we need the Quora Duplicate Questions Dataset and Glove Embedding vectors. Luckily, they are available in [gensim-data](TODO) and will be automagically downloaded when needed.

### Running Evaluations
The models:  
- DRMM TKS
- MatchPyramid
- BiDirectional Attention Flow - T (senTence-level)  
can be found in the folder `sl_eval/models/`

The folder `evaluation_scripts/` contains files and folders for running these models on different datasets.

#### Note on running evaluation

