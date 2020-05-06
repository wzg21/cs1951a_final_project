import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

def t_test(col):
    sample_0 = df[df[col]==0]
    sample_1 = df[df[col]==1]
    t_statistic, p_value = stats.ttest_ind(sample_0['rating'], sample_1['rating'], equal_var=False)
    print('t-test between rating and {}:'.format(col))
    print('t_statistic:', t_statistic)
    print('p_value:', p_value)

def anova(col, values):
    samples = []
    for v in values:
        sample = df[df[col]==v]['rating']
        samples.append(sample)
    f_statistic, p_value = stats.f_oneway(*samples)
    print('anova test between rating and {}:'.format(col))
    print('f_statistic:', f_statistic)
    print('p_value:', p_value)

def linear_regression(col):
    sns.jointplot(x=col, y='rating', data=df, kind='hex')
    plt.show()
    sns.regplot(x=col, y='rating', data=df)
    plt.show()


df = pd.read_csv('data/businesses_nyc_small.csv')
df.columns = map(str.capitalize, df.columns)

# all transaction types
all_trans_types = ['Delivery', 'Pickup', 'Restaurant_reservation']

# all 23 categories 
all_cats = ['Italian', 'Breakfast_brunch', 'Newamerican', 'Pizza', 'Tradamerican', 'Sandwiches', 'Chinese', 'Bars', 'Japanese', 'Mexican',
'Seafood', 'Delis', 'Coffee', 'Salad', 'Mediterranean', 'French', 'Indpak', 'Thai', 'Asianfusion', 'Korean', 'Latin', 'Desserts', 'Bakeries']

# all 23 category ratios 
all_cats_ratios = [c + "_ratio" for c in all_cats]

# non-category variables 
locations = ['Latitude', 'Longitude']

sns.set()

price = ['Price']
t = ['Transaction Type Features', 'Category Features', 'Category Ratio Features', 'Location Features', 'Price Feature']
i = 0
for g in [all_trans_types, all_cats, all_cats_ratios, locations, price]:
    cur_df = df[['Rating'] + g]
    plt.figure(figsize=(12, 9))
    sns.heatmap(cur_df.corr(), cmap=sns.diverging_palette(220, 10, as_cmap=True), center=0)
    plt.title('Correlation Heatmap for Rating and '+t[i], fontsize='x-large', fontweight='semibold')
    plt.tight_layout()
    plt.savefig(fname="D:/Brown_MS/spring2020/cs1951a/cs1951a_final_project/pictures/data\\"+t[i]+'.svg',format="svg")
    i += 1

# for col in ['delivery', 'pickup', 'restaurant_reservation', 'italian', 'breakfast_brunch', 'newamerican', 'pizza', 'tradamerican', 'sandwiches', 'chinese', 'bars', 'japanese', 'mexican',
# 		'seafood', 'delis', 'coffee', 'salad', 'mediterranean', 'french', 'indpak', 'thai', 'asianfusion', 'korean', 'latin', 'desserts', 'bakeries']:
#         t_test(col)
#         print('\n')

# anova('price', [1, 2, 3, 4])

# linear_regression('latitude')
# linear_regression('longitude')

plt.figure(figsize=(12, 9))

sns.set(style="whitegrid")
sns.boxplot(x='Delivery', y='Rating', data=df, palette="Paired")
plt.title('Rating Distribution for Delievery=0 and Delievery=1', fontsize='large', fontweight='semibold')
plt.box(False)
plt.savefig(fname="D:/Brown_MS/spring2020/cs1951a/cs1951a_final_project/pictures/data/deliever.svg",format="svg")

sns.set(style="whitegrid")
sns.boxplot(x='Pickup', y='Rating', data=df, palette="Paired")
plt.title('Rating Distribution for Pickup=0 and Pickup=1', fontsize='large', fontweight='semibold')
plt.box(False)
plt.savefig(fname="D:/Brown_MS/spring2020/cs1951a/cs1951a_final_project/pictures/data/pickup.svg",format="svg")

sns.set(style="whitegrid")
sns.boxplot(x='Restaurant_reservation', y='Rating', data=df, palette="Paired")
plt.title('Rating Distribution for Reservation=0 and Reservation=1', fontsize='large', fontweight='semibold')
plt.box(False)
plt.savefig(fname="D:/Brown_MS/spring2020/cs1951a/cs1951a_final_project/pictures/data/reservation.svg",format="svg")

sns.set(style="whitegrid")
sns.boxplot(x='Price', y='Rating', data=df, palette="Paired")
plt.title('Rating Distribution for each Price Level', fontsize='large', fontweight='semibold')
plt.box(False)
plt.savefig(fname="D:/Brown_MS/spring2020/cs1951a/cs1951a_final_project/pictures/data/price.svg",format="svg")

boxplot_df = pd.DataFrame([])
for i, cat in enumerate(all_cats):
    currgroup = df[df[cat] == 1]
    stars_df = pd.DataFrame([])
    stars_df['Rating'] = currgroup.Rating
    stars_df['Category'] = currgroup[cat].name
    stars_df['mean'] = currgroup.Rating.mean()
    boxplot_df = pd.concat([boxplot_df, stars_df])
boxplot_df = boxplot_df.sort_values(['mean']).reset_index(drop=True)
fig, ax = plt.subplots()
fig.set_size_inches(8, boxplot_df['Category'].nunique()/4)
ax = sns.boxplot(x='Rating', y='Category', data=boxplot_df, palette="Paired")
plt.title('Rating Distribution for each Category', fontsize='large', fontweight='semibold')
plt.box(False)
plt.tight_layout()
plt.savefig(fname="D:/Brown_MS/spring2020/cs1951a/cs1951a_final_project/pictures/data/cat.svg",format="svg")


