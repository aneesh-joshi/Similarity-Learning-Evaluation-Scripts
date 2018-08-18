echo "Get virtualenv"
pip install virtualenv

echo "Create a virtual environment"
virtualenv sl_env

echo "Activate the env"
source sl_env/bin/activate

echo "Install the requirements"
pip install -r requirements.txt

echo "Download the needed data"
cd data/
python get_data.py

echo "Convert the SQUAD data set to SQUAD-T (QA) dataset"
cd ../misc_scripts/
python squad2QA.py --squad_path ../data/train-v1.1.json

echo "Get trec eval binary"
python get_trec.py
