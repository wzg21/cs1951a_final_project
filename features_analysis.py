import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

def t_test(col):
    sample_0 = df[df[col]==0]
    sample_1 = df[df[col]==1]
    t_statistic, p_value = stats.ttest_ind(sample_0['rating'], sample_1['rating'])
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
print('pearson correlation:\n', df.corr())
print('\n')
sns.heatmap(df.corr())
plt.show()

for col in ['delivery', 'pickup', 'restaurant_reservation', 'italian', 'breakfast_brunch', 'newamerican', 'pizza', 'tradamerican', 'sandwiches', 'chinese', 'bars', 'japanese', 'mexican',
		'seafood', 'delis', 'coffee', 'salad', 'mediterranean', 'french', 'indpak', 'thai', 'asianfusion', 'korean', 'latin', 'desserts', 'bakeries']:
        t_test(col)
        print('\n')

anova('price', [1, 2, 3, 4])

linear_regression('latitude')
linear_regression('longitude')


