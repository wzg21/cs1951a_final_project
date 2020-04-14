import csv
import json

ri_zipcodes_file = "./crawl/zip_codes_ma.txt"
ri_business_path = "./raw_data/businesses/ma"
ri_reviews_path = "./raw_data/reviews/ma"

ri_zipcodes = open(ri_zipcodes_file, "r").read().split('\n')

b_output_path = "./businesses_ma.csv"
r_output_path = "./reviews_ma.csv"

def write_to_csv(file, row):
    with open(file, 'a') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(row)


b_keys = ["id", "alias", "name", "image_url", "is_closed", "url", "review_count", "category_alias", "category_title",
    "rating", "latitude", "longitude", "delivery", "pickup", "restaurant_reservation", "price", "address1", "address2", "address3", "city", "zip_code",
    "country", "state", "display_address", "phone", "display_phone", "distance"]

r_keys = ["id", "url", "text", "rating", "time_created",  "user_id",  "user_profile_url",  "user_image_url", "user_name", "business_id"]

# write headers to csv files
with open(b_output_path, 'w', newline='') as b_csv:
    writer = csv.writer(b_csv)
    writer.writerow(b_keys)
    
with open(r_output_path, 'w', newline='') as r_csv:
    writer = csv.writer(r_csv)
    writer.writerow(r_keys)


# parse data in json files
for zipcode in ri_zipcodes:
    b_json_path = ri_business_path + "/" + "raw_data_businesses_" + zipcode + ".json"
    r_json_path = ri_reviews_path + "/" + "raw_data_reviews_" + zipcode + ".json"
    businesses = []
    reviews = []
<<<<<<< Updated upstream
    print(zipcode)

    try:
        with open(b_json_path, "r") as b:
            b_data = json.load(b)
    except ValueError:
        print ("json error")
        continue
    except:
        raise
=======
    
    with open(b_json_path, "r") as b:
        b_data = json.load(b)
>>>>>>> Stashed changes
    
    for b_item in b_data:
        b_zipcode = b_item['location']['zip_code']
        b_id = b_item['id'] 
        if (b_zipcode != zipcode):
            continue
        if 'price' not in b_item or len(b_item["price"]) == 0:
            continue
        businesses.append(b_item)

        try: 
            with open(r_json_path, "r") as r:
                r_data = json.load(r)
            for r_item in r_data:     
                r_bid = r_item['business_id']
                if r_bid == b_id:
                    reviews.append(r_item)
        except ValueError:
            print ("json error")
            continue
        
        
    # write to csv files
    for b in businesses:
        row = []

        i = 0
        for key in b:
            if (key == b_keys[i]):
                if key == "price":
                    row.append(len(b[key]))
                else:
                    row.append(b[key])
                i += 1
            else:
                if (key in b_keys):
                    ind = b_keys.index(key)
                    for j in range(ind - i):
                        row.append("")
                    # row.append(b[key])
                    if key == "price":
                        row.append(len(b[key]))
                    else:
                        row.append(b[key])
                    i = ind
                    i += 1
                else:
                    if (key == "categories"):
                        ind = b_keys.index("category_alias")
                        for j in range(ind - i):
                          row.append("")
                        alias_str = ""
                        title_str = ""
                        get_list = b[key]
                        for l in get_list:
                            alias_str += l["alias"] + ","
                            title_str += l["title"] + ","
                        row.append(alias_str[:-1])
                        row.append(title_str[:-1])
                        i = ind + 2
                    elif (key == "coordinates"):
                        ind = b_keys.index("latitude")
                        for j in range(ind - i):
                          row.append("")
                        row.append(b[key]["latitude"])
                        row.append(b[key]["longitude"])
                        i = ind + 2
                    elif (key == "location"):
                        ind = b_keys.index("address1")
                        for j in range(ind - i):
                          row.append("")
                        row.append(b[key]["address1"])
                        row.append(b[key]["address2"])
                        row.append(b[key]["address3"])
                        row.append(b[key]["city"])
                        row.append(b[key]["zip_code"])
                        row.append(b[key]["country"])
                        row.append(b[key]["state"])
                        display_addr = ",".join(b[key]["display_address"])
                        row.append(display_addr)
                        i = ind + 8
                    elif (key == "transactions"):
                        ind = b_keys.index("delivery")
                        for j in range(ind - i):
                            row.append("")
                        if "delivery" in b[key]:
                            row.append(1)
                        else:
                            row.append(0)
                        if "pickup" in b[key]:
                            row.append(1)
                        else:
                            row.append(0)
                        if "restaurant_reservation" in b[key]:
                            row.append(1)
                        else:
                            row.append(0)
                        i = ind+3


        if (i < len(b_keys)):
            for j in range(len(b_keys) - 1):
                row.append("")

        write_to_csv(b_output_path, row)
                
    for r in reviews:
        row = []
        for key in r:
            if (key != "user"):
                row.append(r[key])
            else:
                row.append(r[key]["id"]) 
                row.append(r[key]["profile_url"]) 
                row.append(r[key]["image_url"])
                row.append(r[key]["name"]) 
        write_to_csv(r_output_path, row)
        
