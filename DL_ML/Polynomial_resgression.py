import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#y = mx+b
#m= y-b
#y(x) = m1x^2 + m2x + b
#find m1,m2,b
# def gradient_descent(m1_now,m2_now, b_now, x,y, L):
#     m1_gradient = 0
#     m2_gradient = 0
#     b_gradient = 0

#     n = len(x)

#     for i in range(n):
#         # x = points.iloc[i].input_x
#         # y = points.iloc[i].output_y

#         m1_gradient += -(2/n) 
#         m2_gradient += -(2/n) * x * (y - (m2_now * x + b_now))
#         b_gradient += -(2/n) * (y - (m2_now * x + b_now))

#     m = m2_now - m2_gradient * L
#     b = b_now - b_gradient * L

#     return m, b
# a =  
data = pd.read_csv('data.csv')
x = data.iloc[:,0:1].values
y = data.iloc[:,1:].values

lin_regs = LinearRegression()
lin_regs.fit(x,y)
poly_regs = PolynomialFeatures(degree=2)
x_poly = poly_regs.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# print(lin_regs.predict([[10]]))
# print(lin_reg_2.predict(poly_regs.fit_transform([[10]])))
plt.scatter(x,y,color="blue")
plt.plot(x,lin_reg_2.predict(poly_regs.fit_transform(x)),color="red")
plt.title("Bluff detection model(Linear Regression)")
plt.xlabel("Input_distance")
plt.ylabel("Output_heigh")
plt.show()
