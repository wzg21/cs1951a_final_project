import pandas as pd
import matplotlib.pyplot as plt

ri_business_path = "./businesses.csv"
ri_reviews_path = "./reviews.csv"

b_df = pd.read_csv(ri_business_path)
print(b_df.describe())
# b_df.hist(column="distance")
# plt.show()

# b_df.hist(column="review_count")
# b_df.hist(column="review_count", range=(0, 750), bins = 20)
# plt.show()

# b_df.hist(column="rating")
# plt.show()

# b_df.hist(column="zip_code")
# plt.show()

# b_df["city"].value_counts().plot(kind='bar')
# plt.show()

# b_df["zip_code"].value_counts().plot(kind='bar')
# plt.show()


b_df["price"].value_counts().plot(kind='bar')
plt.show()