import csv
from collections import defaultdict
import pandas as pd

b_file_path = "./businesses_nyc.csv"
c_file_path = "./pop_categories_nyc.txt"
b_category_file_path = "./businesses_nyc_small.csv"
df = pd.read_csv(b_file_path)


def count_categories():
    d = defaultdict(int)

    # df = pd.read_csv(b_file_path)
    categories = df['category_alias']
    for index, row in df.iterrows():
        # print(row['category_alias'])
        categories = row['category_alias'].split(",")
        for c in categories:
            d[c] += 1

    c_count = open(c_file_path, "a")

    for w in sorted(d, key=d.get, reverse=True):
        print(w, d[w])
        c_count.write(w + "," + str(d[w]) + "\n")

    print("number of categories: " + str(len(d)))
    c_count.close()


def pop_categories():
    m = {}
    pcs = []
    for line in open('popular_categories.txt', 'r'):
        line = line.strip('\n')
        l = line.split(',')
        pcs.append(l[0])
        for c in l:
            m[c] = l[0]
    return pcs, m


# generate a smaller businesses dataset with popular categories as indicator variables
def clean_categories():
    popular_categories, m = pop_categories()
    print(popular_categories)
    dictOfpop = {i: [] for i in popular_categories}
    print(dictOfpop)
    b_keys = ["id", "name", "review_count", "category_alias", "rating", "latitude", "longitude", "delivery", "pickup",
              "restaurant_reservation", "price", "city", "zip_code", "country", "state",
              "distance"] + popular_categories

    # write headers to csv files
    with open(b_category_file_path, 'w', newline='') as b_csv:
        writer = csv.writer(b_csv)
        writer.writerow(b_keys)

    for index, row in df.iterrows():
        row_small = row[['id', 'name', 'review_count', 'category_alias', 'rating', 'latitude', 'longitude', 'delivery',
                         'pickup', 'restaurant_reservation', 'price', 'city', 'zip_code', 'country', 'state',
                         'distance']]
        row_small_list = row_small.values.tolist()

        cat = [0 for i in range(len(popular_categories))]
        categories = row['category_alias'].split(",")
        for c in categories:
            if c in popular_categories:
                dictOfpop[c].append(row['rating'])
        # write_to_csv(b_category_file_path, new_row)

    c_count = open(c_file_path, "a")

    for key in dictOfpop.keys():
        c_count.write(key + "," + str(len(dictOfpop[key])) + "," + str(sum(dictOfpop[key])/len(dictOfpop[key])) + "\n")
        # print(key)
        # print(len(dictOfpop[key]))
        # print(sum(dictOfpop[key])/len(dictOfpop[key]))
    c_count.close()

def write_to_csv(file, row):
    with open(file, 'a', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


# count_categories()
clean_categories()

