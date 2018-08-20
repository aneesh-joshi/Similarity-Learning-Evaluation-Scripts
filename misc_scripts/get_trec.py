"""This script will get the trec_eval repo and `make` it

More can be read at:
- https://github.com/usnistgov/trec_eval
- https://trec.nist.gov/trec_eval/

Warning: You should have Make instlled with a C compiler
"""

import os

if os.path.isdir('trec_eval'):
	print('Removing existing old repo')
	os.system('rm -rf trec_eval')
os.system('git clone https://github.com/usnistgov/trec_eval.git')
os.chdir('trec_eval/')
os.system('git checkout de6a29f8ba9312c73f978aa9739695aa8ebf48eb')
os.system('make')