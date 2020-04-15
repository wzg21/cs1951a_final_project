import csv
from collections import defaultdict
import pandas as pd

b_file_path = "./businesses_nyc.csv"
c_file_path = "./categories_nyc.txt"

d = defaultdict(int)

df = pd.read_csv(b_file_path)
categories = df['category_alias']
for index, row in df.iterrows():
    # print(row['category_alias'])
    categories = row['category_alias'].split(",")
    for c in categories:
    	d[c] += 1


c_count = open(c_file_path,"a") 
 
for w in sorted(d, key=d.get, reverse=True):
    print(w, d[w])
    c_count.write(w + "," + str(d[w]) + "\n")

print(len(d))
c_count.close()