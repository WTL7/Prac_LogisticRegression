
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ad_data = pd.read_csv('advertising.csv')

print ad_data.head()
print ad_data.describe()
print ad_data.info()

sns.set_style('whitegrid')
sns.distplot(ad_data['Age'], kde = False, bins = 30)

#sns.jointplot(x = 'Age', y = 'Area Income', data = ad_data, kind = 'kde')

#sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = ad_data)

sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', 
        data = ad_data, kind = 'reg', color = 'g')

sns.pairplot(data = ad_data, hue = 'Clicked on Ad')

ad_data.drop(['Ad Topic Line','City','Country','Timestamp'], axis = 1, inplace = True)

plt.show()

#----------machine learning------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
        ad_data.drop('Clicked on Ad', axis = 1), ad_data['Clicked on Ad'])

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print classification_report(y_test, predictions)        