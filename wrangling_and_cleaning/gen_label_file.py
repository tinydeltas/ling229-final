from pandas import read_csv
import sys

gold = read_csv(sys.argv[1], encoding='utf-8')["is_romantic"]
gold = [0 if lab == "Romantic" else 1 for lab in gold]

f = open(sys.argv[2], "w")
for lab in gold:
    f.write(str(lab) + "\n")
f.flush()
f.close()