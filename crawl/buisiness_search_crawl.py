import requests
import json
import os

API_KEY = 'S8LDt8IkT8Iq9NWqggevbFQcSecXOFnYAmZK7msYXNGgk5Qj7zSHmwhl-_6ZEROSYNCqCF1aIErc_R9rQmwvjrj-jFPp38Lzu24xlj8mhMzdw-c90VtGAhLIr5hlXnYx'
HEADERS = {'Authorization': 'Bearer %s' % API_KEY}
URL = 'https://api.yelp.com/v3/businesses/search'
LIMIT = 50
BUSINESSES_DATA_FILE = '../raw_data/businesses/raw_data_businesses_'
REVIEWS_DATA_FILE = '../raw_data/reviews/raw_data_reviews_'
ZIP_CODE_FILE = 'zip_codes.txt'
START_I = 0



f = open(ZIP_CODE_FILE, 'r')
locations = f.readlines()
print('There are %d zip codes.' % len(locations))
f.close()

for i in range(START_I, len(locations)):
	location = locations[i].strip('\n')
	fb = open(BUSINESSES_DATA_FILE + location + '.json', 'a')
	fb.write('[')
	fv = open(REVIEWS_DATA_FILE + location + '.json', 'a')
	fv.write('[')

	params = \
	{
	'term':'Restaurants',
	'location':location
	}
	response = requests.get(URL, params=params, headers=HEADERS)
	if response.status_code == 200:
		real_number = response.json()['total']
	else:
		raise RuntimeError('Error %s' % response.status_code)
	#Yelp API only supports 1000 results.
	number = min(1000, real_number)
	print('There are %d restaurants in %s, take the first %d ones.' % (real_number, location, number))

	for offset in range(0, number, LIMIT):
		print('Processing the %dth batch.' % (offset//LIMIT))
		limit = min(LIMIT, number - offset)
		params = \
		{
		'term':'Restaurants',
		'location':location,
		'limit':str(limit),
		'offset':str(offset)
		}
		response = requests.get(URL, params=params, headers=HEADERS)

		if response.status_code == 200:
			for j, business in enumerate(response.json()['businesses']):
				# print(business['name'])
				if offset != 0 or j != 0:
					fb.write(', ')
				json.dump(business, fb)

				response = requests.get('https://api.yelp.com/v3/businesses/' + business['id'] + '/reviews', headers=HEADERS)
				if response.status_code == 200:
					for k, review in enumerate(response.json()['reviews']):
						# print(review['text'])
						if offset != 0 or j != 0 or k != 0:
							fv.write(', ')
						review['business_id'] = business['id']
						json.dump(review, fv)
				else:
					raise RuntimeError('Error %s' % response.status_code)

		else:
			raise RuntimeError('Error %s' % response.status_code)

	fb.write(']')
	fb.close()
	fv.write(']')
	fv.close()


