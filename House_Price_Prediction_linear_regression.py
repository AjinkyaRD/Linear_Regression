# House price prediction:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import os
print('select train.csv file from Data folder:')
df=pd.read_csv(filedialog.askopenfilename(initialdir=os.getcwd(),title='SELECT TRAINING DATA'))

# z-score-Normalization of Data:
df['LotArea']=(df['LotArea']-df['LotArea'].mean())/df['LotArea'].std()
df['OverallQual']=(df['OverallQual']-df['OverallQual'].mean())/df['OverallQual'].std()
df['OverallCond']=(df['OverallCond']-df['OverallCond'].mean())/df['OverallCond'].std()
df['TotalBsmtSF']=(df['TotalBsmtSF']-df['TotalBsmtSF'].mean())/df['TotalBsmtSF'].std()
df['GrLivArea']=(df['GrLivArea']-df['GrLivArea'].mean())/df['GrLivArea'].std()
df['1stFlrSF']=(df['1stFlrSF']-df['1stFlrSF'].mean())/df['1stFlrSF'].std()
y_mean = df['SalePrice'].mean()
y_std = df['SalePrice'].std()
df['SalePrice']=(df['SalePrice']-df['SalePrice'].mean())/df['SalePrice'].std()

# Split of data:
train_data=df.sample(frac=0.8,random_state=1596)
validation_data=df.drop(train_data.index)

# Training data:
x_train=np.array(train_data[['LotArea','OverallQual','OverallCond','TotalBsmtSF','GrLivArea','1stFlrSF']])
y_train=np.array(train_data['SalePrice'])

# Cost calculation (per iteration):
def cost(x_train,y_train,w,b):
    m=y_train.shape[0]
    cost=0
    for i in range(m):
        f=np.dot(x_train[i],w)+b
        cost+=(f-y_train[i])**2
    total_cost=(1/(2*m))*cost
    return total_cost

# calculation of derivative term:
def grad(x_train,y_train,w,b):
    m=y_train.shape[0]
    dw,db=0,0
    for i in range(m):
        f=np.dot(x_train[i],w)+b
        dw+=(f-y_train[i])*x_train[i]
        db+=(f-y_train[i])
    return dw/m,db/m

# Applying gradient_descent algorithm:
def gradient_descent(x_train,y_train,w,b,alfa):
    co=[]
    iter=[]
    m=y_train.shape[0]
    for i in range(3000):
        co.append(cost(x_train,y_train,w,b))
        iter.append(i)
        d_w,d_b=grad(x_train,y_train,w,b)
        w-=(alfa*d_w)
        b-=(alfa*d_b)
    return co,iter,w,b
# Input:
w=np.zeros(6)
b=0.
print('wait ...')
p,q,w_last,b_last=gradient_descent(x_train,y_train,w,b,0.001)
# plotting of data:
plt.plot(q,p,c='g')
plt.xlabel('No. of Iterations')
plt.ylabel('Cost')
plt.title('Cost Vs Iterations')
plt.grid()
plt.show()

# Cross-Validation:
x_valid=np.array(validation_data[['LotArea','OverallQual','OverallCond','TotalBsmtSF','GrLivArea','1stFlrSF']]) 
y_valid=np.array(validation_data['SalePrice'])
def Predict(x_valid,w,b):
    m=x_valid.shape[0]
    y_output=np.zeros(m)
    for i in range(m):
        y_output[i]=(np.dot(x_valid[i],w)+b)
    return y_output
y_valid_out=Predict(x_valid,w_last,b_last)

m=y_valid.shape[0]
error=0
for i in range(m):
    error+=((y_valid[i]-y_valid_out[i])**2)
MSE=(error/m)

# #  Denormalization :
y_valid_out=(y_valid_out*y_std)+y_mean
y_valid=(y_valid*y_std)+y_mean

# Final_output:
validation_output=pd.DataFrame(y_valid_out)
print('Cross Validation Output:')
print(f'MSE ERROR of cross validation: {MSE}')

# checking for x_test data:
# Importing Data:
test_data=pd.read_csv(filedialog.askopenfilename(initialdir=os.getcwd(),title='SELECT TESTING DATA'))
# z-score-Normalization of Data:
test_data['LotArea']=(test_data['LotArea']-test_data['LotArea'].mean())/test_data['LotArea'].std()
test_data['OverallQual']=(test_data['OverallQual']-test_data['OverallQual'].mean())/test_data['OverallQual'].std()
test_data['OverallCond']=(test_data['OverallCond']-test_data['OverallCond'].mean())/test_data['OverallCond'].std()
test_data['TotalBsmtSF']=(test_data['TotalBsmtSF']-test_data['TotalBsmtSF'].mean())/test_data['TotalBsmtSF'].std()
test_data['GrLivArea']=(test_data['GrLivArea']-test_data['GrLivArea'].mean())/test_data['GrLivArea'].std()
test_data['1stFlrSF']=(test_data['1stFlrSF']-test_data['1stFlrSF'].mean())/test_data['1stFlrSF'].std()

#Training data:
x_test=np.array(test_data[['LotArea','OverallQual','OverallCond','TotalBsmtSF','GrLivArea','1stFlrSF']])

#Prediction for test data:
def Predict(x_test,w,b):
    m=x_test.shape[0]
    y_output=np.zeros(m)
    for i in range(m):
        y_output[i]=(np.dot(x_test[i],w)+b)
    return y_output
y_output=Predict(x_test,w_last,b_last)

#Denormalization :
y_output=(y_output*y_std)+y_mean

#Final_output:
test_data=pd.DataFrame(y_output)
print(test_data)