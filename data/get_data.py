"""
Utility script to download the datsets for Similarity Learning
Currently supports:
- WikiQA
- SNLI
- SICK
- SQUAD
- insurance_qa

Additional datasets like
- Quora Duplicate Questions
- Glove
will be downloaded when needed by gensim-data

Example Usage:
--------------

To get all needed datasets:
$ python get_data.py

To get wikiqa
$ python get_data.py --datafile wikiqa

Options are:
- wikiqa
- snli
- sick
- squad1
- insurance_qa


"""
import requests
import argparse
import zipfile
import logging
import os

logger = logging.getLogger(__name__)

# The urls and filepaths of currently supported files
wikiqa_url, wikiqa_file = "https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/", "WikiQACorpus.zip"  # noqa
snli_url, snli_file = "https://nlp.stanford.edu/projects/snli/", "snli_1.0.zip"
SICK_url, SICK_file = "http://clic.cimec.unitn.it/composes/materials/", "SICK.zip"
SQUAD2_train_url, SQUAD2_train_file = "https://rajpurkar.github.io/SQuAD-explorer/dataset/", "train-v2.0.json"
SQUAD1_train_url, SQUAD1_train_file = "https://rajpurkar.github.io/SQuAD-explorer/dataset/", "train-v1.1.json"

InsuranceQA_git_link = "https://github.com/codekansas/insurance_qa_python.git"

def download(url, file_name, output_dir, unzip=False):
    """Utility function to download a given file from the given url
    Paramters:
    ---------
    url: str
        Url of the file, without the file

    file_name: str
        name of the file ahead of the url path

    Example:
    url = www.example.com/datasets/
    file_name = example_dataset.zip
    """
    logger.info("Downloading %s" % file_name)
    req = requests.get(url + file_name)
    file_save_path = os.path.join(output_dir, file_name)
    try:
        with open(file_save_path, "wb") as code:
            code.write(req.content)
            logger.info("Download of %s complete" % file_name)
    except Exception as e:
        logger.info(str(e))

    if unzip:
        logger.info("Unzipping %s" % file_name)
        with zipfile.ZipFile(file_save_path, "r") as zip_ref:
            zip_ref.extractall(path=output_dir)
        logger.info("Unzip complete")


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--datafile', default='all',
                        help='file you want to download. Options: - wikiqa, snli, sick, squad1, insurance_qa')
    parser.add_argument('--output_dir', default='./',
                        help='the directory where you want to save the data')

    args = parser.parse_args()
    if args.datafile == 'wikiqa':
        download(wikiqa_url, wikiqa_file, args.output_dir, unzip=True)
    elif args.datafile == 'snli':
        download(snli_url, snli_file, args.output_dir, unzip=True)
    elif args.datafile == 'sick':
        download(sick_url, sick_file, args.output_dir, unzip=True)
    elif args.datafile == 'squad1':
        download(SQUAD1_train_url, SQUAD1_train_file, args.output_dir)
    elif args.datafile == 'insurance_qa':
        os.system('git clone ' + InsuranceQA_git_link)
    else:
        logger.info("Downloading all files.")
        download(wikiqa_url, wikiqa_file, args.output_dir, unzip=True)
        download(snli_url, snli_file, args.output_dir, unzip=True)
        download(SICK_url, SICK_file, args.output_dir, unzip=True)
        download(SQUAD1_train_url, SQUAD1_train_file, args.output_dir)
        # os.system('git clone ' + InsuranceQA_git_link)
