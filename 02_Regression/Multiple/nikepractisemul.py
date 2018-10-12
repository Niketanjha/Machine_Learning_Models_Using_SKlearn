import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("50_Startups.csv")

x=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label=LabelEncoder()
x[:,3]=label.fit_transform(x[:,3])
onehot=OneHotEncoder(categorical_features=[3])
x=onehot.fit_transform(x).toarray()
x=x[:,1:]

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
linear_regressor=LinearRegression()
linear_regressor.fit(x_train,y_train)

prediction=linear_regressor.predict(x_test)

plt.scatter(y_test,prediction)

import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[0,3]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

linear_regressor.fit(x_train[:,2:3],y_train)
prediction=linear_regressor.predict(x_test[:,2:3])

plt.plot(x_test[:,2:3],prediction,color='red')
plt.scatter(x_test[:,2:3],y_test,color='blue')