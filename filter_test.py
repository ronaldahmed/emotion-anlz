import os,sys
import pdb

all_fn = sys.argv[1]
test_fn = sys.argv[2]

tmp = open(test_fn,"r").read().strip("\n").split("\n")

test_text = []
for line in tmp[1:]:
	line = line.replace('""','"')
	if line[0] == '"':
		line = line[1:]
	test_text.append(line)

test_file = open("test.tsv","w")

# pdb.set_trace()

found = set()

for line in open(all_fn,"r"):
	line = line.strip("\n")
	if line=="": continue
	text,label = line.split("\t")
	is_in = False
	for test_chars in test_text:
		if any([ 
				 text[:20] == test_chars[:20],
				 text[:30] == test_chars[:30],
				]):
			is_in = True
			found.add(test_chars)
			break
	#
	if is_in:
		print(line,file=test_file)
#
test_file.close()

# for test_chars in test_text:
# 	if test_chars not in found:
# 		print(test_chars)
# 		pdb.set_trace()

