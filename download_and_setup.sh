python setup.py
cd data/
python get_data.py
cd ../misc_scripts/
python squad2QA.py --squad_path ../data/train-v1.1.json
python get_trec.py
