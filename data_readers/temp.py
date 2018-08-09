to_write = ""

with open('SQUAD-T-QA+.tsv', 'r',  encoding='utf-8') as f:
	old_line = ""
	for i, line in enumerate(f):
		if line[0] != 'Q':
			to_write = to_write[:-2] + '\t'
		to_write += line		

with open('last_hop', 'w', encoding='utf-8') as f:
	f.write(to_write)
