echo "Running Eval on Quora Question Pairs"
python eval_qp.py

echo "Running Eval on SICK"
python eval_sick.py

echo "Running eval on SNLI"
python eval_snli.py

echo "Running eval on WikiQA"
cd WikiQA
python eval_wikiqa.py

echo "Running eval on InsuranceQA"
cd ../InsuranceQA
python eval_iqa.py

echo "Running Glove + NN  baseline on Quora"
cd ..
python eval_baseline_qp.py

echo "Running Glove + NN baseline on SICK"
python eval_baseline_sick.py

echo "Running Glove + NN baseline on SNLI"
python eval_baseline_snli.py
