import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor=SVR()
regressor.fit(x,y)

y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid)),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('Truth or Bluff(svr model)')
plt.xlabel('Position level')
plt.ylabel('salary')
plt.show()