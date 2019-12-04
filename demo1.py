import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn


def prepare_country_stats(p1, p2):
    _a = {'GDP per capita': [], 'Life satisfaction': []}
    _p1 = p1[['Country', 'Value']]
    _p2 = p2[['Country', '2015']]
    c = pd.merge(_p1, _p2, on='Country', how='left')
    c.to_csv('./dataset/results.csv')
    _a['GDP per capita'] = c['Value'].as_matrix()
    _a['Life satisfaction'] = c.as_matrix(columns=c.columns[1:2])
    return _a


#  Load the data
oecd_bli = pd.read_csv("./dataset/oecd_bli_2015.csv", thousands=',')
_gdp_per_capita = pd.read_csv("./dataset/gdp_per_capita.xls", thousands=',', delimiter='\t', encoding='latin1',
                             na_values="n/a")
df = _gdp_per_capita
gdp_per_capita = df.loc[df['Estimates Start After'].isin(['2015'])]
gdp_per_capita.to_csv('./dataset/results1.csv')

#  Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

#  Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

#  Select a linear model
lin_reg_model = sklearn.linear_model.LinearRegression()

#  Train the model
lin_reg_model.fit(X, y)

#  Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(lin_reg_model.predict(X_new))  # outputs [[ 5.96242338]]
