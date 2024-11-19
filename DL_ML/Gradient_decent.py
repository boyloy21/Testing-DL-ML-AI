import numpy as np

# y = mx + b
# Find m and b
def gradient_decent(x, y):
    m_curr = 0
    b_curr = 0
    iteration = 1000
    n = len(x)
    learning_rate = 0.003

    for i in range(iteration):
        y_predicted = m_curr * x + b_curr
        cost = (1/n)* sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))

# x = np.array([1,2,3,4,5])
# y = np.array([5,7,9,11,13])
x = np.array([0.675,1.35,2.03,2.714,3.39,4.07,4.75,5.42,6.084,6.729,7.357,7.964,8.55,9.11,9.65])
y = np.array([0.003,0.013,0.03,0.053,0.081,0.115,0.154,0.196,0.242,0.29,0.34,0.39,0.445,0.498,0.55])
gradient_decent(x,y)