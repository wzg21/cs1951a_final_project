# cs1951a_final_project

2020 Spring CS1951A-Data Science Final Project. Our project plans to predict which kind of restaurant people should open if they would like to have one in a specific state. 

## Structure of repo 

There are two kinds of data, one is raw data which is in the folder `raw_data` and the other one is CSV files which are cleaned data. 

`raw_data` contains two subdirctories: `businesses`, `reviews`. Each subdirectory consists of several folders. For one folder, it contains json files for one specific state.

## Structure of json files

### Data in dirctory `businesses`

For each json file, its suffix represents a zipcode. And the data in the file is the restaurants around that zipcode. 

Each datafile has the following keys and values:

* `id`: The unique identifier for each restaurant
* `alias`: The alias for each restaurant
* `name`: The name of the restaurant
* `image_url`: A related image of the restaurant
* `is_closed`: A `True` or `False` label which represents whethe the restaurant has closed
* `url`: The link of the restaurant on yelp
* `review_count`: Total number of reviews on yelp for the restaurant
* `categories`: A list of dictionaries. Each item in the list represents the label of this restaurant which has two attributes: `alias`, `title`
* `rating`: The average rating of this restaurant
* `coordinates`: The coordinate of the restaurant consisted of two attributes: `latitude`, `longitude`
* `transactions`: A list which may contain `delivery` and `pickup` 
* `price`: A string uses several `$` to present the average price of the restaurant
* `location`: A dictionary which uses `address1`, `address2`, `address3`, `city`, `zip_code`, `country`, `state` and `display_address` to show the location of the restaurant
* `phone`: Phone number of the restaurant
* `display_phone`: The phone number displayed on yelp of the restaurant
* `distance`: The distance between the restaurant and the given zip code

### Data in dirctory `reviews`

For each json file, its suffix represents a zipcode. And the data in the file is the reviews of the restaurants around that zipcode.

Each datafile has the following keys and values:

* `id`: The unique identifier of the review
* `url`: The link of this review on yelp
* `text`: The text content of the review
* `rating`: The rating of the review (Range from 1 to 5)
* `time_created`: The time that the review creates.(YYYY-MM-DD HH:mm:ss)
* `user`: User information contains `id`, `profile_url`, `image_url`, `name`
* `business_id`: The identifier of the restaurant which maps the review to business.json

## Structure of CSV files

The CSV file contains data after cleaning. There are two CSV files for each state and the suffix of the file represents the state it belongs to.

### Data in file `businesses.csv`

Each line in the file is one restaurant in the state. One file has 25 columns:

* `id`: The unique identifier for each restaurant
* `alias`: The alias for each restaurant
* `name`: The name of the restaurant
* `image_url`: A related image of the restaurant
* `is_closed`: A `True` or `False` label which represents whethe the restaurant has closed
* `url`: The link of the restaurant on yelp
* `review_count`: Total number of reviews on yelp for the restaurant
* `category_alias`: The alias of the restaurant's label
* `category_title`: The label of this restaurant
* `rating`: The average rating of this restaurant
* `latitude`: The latitude of the restaurant
* `longitude`: The longitude of the restaurant
* `transactions`: A list which may contain `delivery` and `pickup` 
* `price`: A string uses several `$` to present the average price of the restaurant
* `address1`: Address 1 of the restaurant
* `address2`: (Optinal) Address 2 of the restaurant
* `address3`: (Optinal) Address 3 of the restaurant
* `city`: The city of the restaurant
* `zip_code`: The zip code of the restaurant
* `country`: The country of the restaurant
* `state`: The state of the restaurant
* `display_address`: The exact address of the restaurant
* `phone`: Phone number of the restaurant
* `display_phone`: The phone number displayed on yelp of the restaurant
* `distance`: The distance between the restaurant and the given zip code

### Data in file `reviews.csv`

Each line in the file is a review of one restaurant in the state. One file has 10 columns:

* `id`: The unique identifier of the review
* `url`: The link of this review on yelp
* `text`: The text content of the review
* `rating`: The rating of the review (Range from 1 to 5)
* `time_created`: The time that the review creates.(YYYY-MM-DD HH:mm:ss)
* `user_id`: The ID of the user who writed the review
* `user_profile_url`: The profile link of the user on yelp
* `user_image_url`: The link of the user's image on yelp
* `user_name`: User's name on yelp
* `business_id`: The identifier of the restaurant which maps the review to business.json

## Notes

Currently, we only have two CSV files: `businesses.csv`, `reviews.csv` which contain cleaned data for restaurants in Rhode Island. Due to rate limit, the collection for the restaurants in MA and NY is still in progress but we are about to complete it.