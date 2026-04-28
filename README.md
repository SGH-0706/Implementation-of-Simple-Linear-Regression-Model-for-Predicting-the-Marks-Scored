# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start
2.Import required libraries
3.Create dataset (Hours and Scores) 
4. Convert data into DataFrame 
5.Separate input (X) and output (Y) 
6.Split dataset into training and testing sets 
7.Create Linear Regression model 
8.Train the model using training data 
9.Predict values using test data 
10.Plot training set results 11 Plot testing set results 12 Calculate MSE, MAE, RMSE 13Display results 14Stop

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Srinithi Muthukumar
RegisterNumber:  212224240161
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


data = {
    'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7],
    'Scores': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25]
}

df = pd.DataFrame(data)

print(df)

X = df[['Hours']].values
Y = df['Scores'].values

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=1/3, random_state=0)


reg = LinearRegression()
reg.fit(Xtrain, Ytrain)


Ypred = reg.predict(Xtest)

print("Predicted Values:", Ypred)


plt.scatter(Xtrain, Ytrain, color='orange')
plt.plot(Xtrain, reg.predict(Xtrain), color='red')
plt.title("Training Set")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


plt.scatter(Xtest, Ytest, color='blue')
plt.plot(Xtest, reg.predict(Xtest), color='green')
plt.title("Test Set")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


mse = mean_squared_error(Ytest, Ypred)
mae = mean_absolute_error(Ytest, Ypred)
rmse = np.sqrt(mse)

print("MSE :", mse)
print("MAE :", mae)
print("RMSE :", rmse)

*/
```

## Output:
<img width="1303" height="308" alt="image" src="https://github.com/user-attachments/assets/13d5c3af-4c49-4edc-bbb8-bcd9c9431f7a" />

<img width="944" height="674" alt="image" src="https://github.com/user-attachments/assets/54c03f83-0a60-4cf4-b69f-2f1d0f4f6823" />

<img width="953" height="748" alt="image" src="https://github.com/user-attachments/assets/17d4fda8-2d9b-406a-a7d9-1c4537943a00" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
