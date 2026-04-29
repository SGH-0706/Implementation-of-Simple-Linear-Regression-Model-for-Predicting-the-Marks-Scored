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
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train, x_test ,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title ("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test ,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='Red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE= ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE= ',mae)
*/
```

## Output:
<img width="859" height="641" alt="image" src="https://github.com/user-attachments/assets/12418762-b7ad-4769-be6b-61d74a2c96e8" />

<img width="421" height="301" alt="image" src="https://github.com/user-attachments/assets/cd69d5f0-43e7-4b84-871e-4d4309ce0955" />


<img width="851" height="667" alt="image" src="https://github.com/user-attachments/assets/d9e3068c-e0ae-4206-81de-cd9f0e24e4f8" />


<img width="883" height="744" alt="image" src="https://github.com/user-attachments/assets/704c2bae-4e57-4e99-8237-a04557cf07d7" />


<img width="968" height="751" alt="image" src="https://github.com/user-attachments/assets/2f741bd3-1546-48b7-a4a7-85006b059a82" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
