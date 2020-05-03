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


df = pd.read_csv('businesses_nyc_small.csv', usecols=['rating', 'latitude', 'longitude', 'delivery', 'pickup', 'restaurant_reservation', 'price',
		'italian', 'breakfast_brunch', 'newamerican', 'pizza', 'tradamerican', 'sandwiches', 'chinese', 'bars', 'japanese', 'mexican',
		'seafood', 'delis', 'coffee', 'salad', 'mediterranean', 'french', 'indpak', 'thai', 'asianfusion', 'korean', 'latin', 'desserts', 'bakeries'])
# print('pearson correlation:\n', df.corr())
# print('\n')
# sns.heatmap(df.corr(), vmin=-0.3, vmax=0.3)
# plt.show()

# for col in ['delivery', 'pickup', 'restaurant_reservation', 'italian', 'breakfast_brunch', 'newamerican', 'pizza', 'tradamerican', 'sandwiches', 'chinese', 'bars', 'japanese', 'mexican',
# 		'seafood', 'delis', 'coffee', 'salad', 'mediterranean', 'french', 'indpak', 'thai', 'asianfusion', 'korean', 'latin', 'desserts', 'bakeries']:
#         t_test(col)
#         print('\n')

# anova('price', [1, 2, 3, 4])

# linear_regression('latitude')
# linear_regression('longitude')

sns.boxplot(x='delivery', y='rating', data=df, palette="Paired")
plt.show()

sns.boxplot(x='pickup', y='rating', data=df, palette="Paired")
plt.show()

sns.boxplot(x='restaurant_reservation', y='rating', data=df, palette="Paired")
plt.show()

sns.boxplot(x='price', y='rating', data=df, palette="Paired")
plt.show()

boxplot_df = pd.DataFrame([])
for i, cat in enumerate(['italian', 'breakfast_brunch', 'newamerican', 'pizza', 'tradamerican', 'sandwiches', 'chinese', 'bars', 'japanese', 'mexican',
		'seafood', 'delis', 'coffee', 'salad', 'mediterranean', 'french', 'indpak', 'thai', 'asianfusion', 'korean', 'latin', 'desserts', 'bakeries']):
    currgroup = df[df[cat] == 1]
    stars_df = pd.DataFrame([])
    stars_df['rating'] = currgroup.rating
    stars_df['category'] = currgroup[cat].name
    stars_df['mean'] = currgroup.rating.mean()
    boxplot_df = pd.concat([boxplot_df, stars_df])
boxplot_df = boxplot_df.sort_values(['mean']).reset_index(drop=True)
fig, ax = plt.subplots()
fig.set_size_inches(8, boxplot_df['category'].nunique()/4)
ax = sns.boxplot(x='rating', y='category', data=boxplot_df, palette="Paired")
plt.show()


