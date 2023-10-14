import numpy as np

n = 1000
x = np.random.normal(1,2,size = n)
p = 0 #近似期望
for i in range(n):
    p = (1-1/(i+1))*p + (1/(i+1))*(2*x[i]+19*np.sqrt(abs(x[i])) +3)
print(p)