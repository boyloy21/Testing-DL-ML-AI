import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
# x = np.array([2,3,4])
# y = np.array([1,2,3])
# y_bar = np.average(y)
# m = 0.1
# b = 0.05
# y_est = m*x + b
# SST = np.sum((y-y_bar)**2)
# SSR = np.sum((y_est - y_bar)**2)
# SSE = SST + SSR
# R_square = SSR/SST

data = pd.read_csv('Estimate/rice.csv')

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].distance
        y = points.ilco[i].time
        total_error += (y- (m*x + b))**2
    total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].distance
        y = points.iloc[i].time

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b

m = 0
b = 0.0
L = 0.005
epochs = 3000

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, L)
    # y = m*data.input_x + b
    # print(y)

print(m,b)
plt.scatter(data.distance, data.time, color = "black")
plt.plot(list(range(0, 2)), [m * x + b for x in range(0, 2)], color="red")
plt.show()
