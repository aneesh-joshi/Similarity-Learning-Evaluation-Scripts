"""This script will get the trec_eval repo and `make` it

More can be read at:
- https://github.com/usnistgov/trec_eval
- https://trec.nist.gov/trec_eval/

Warning: You should have Make instlled with a C compiler
"""

import os

os.system('git clone https://github.com/usnistgov/trec_eval.git')
os.chdir('trec_eval/')
os.system('make')