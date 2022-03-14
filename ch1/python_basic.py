import numpy as np
import matplotlib.pyplot as plt
array = np.array([1, 3, 4, 5])
print(array)
print(type(array))

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x)
print(y)
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x / 2.0)

a = np.array([[1, 2], [3, 4]])
print(a)
print(a.shape)
print(a.dtype)
print(a * 10)
b = np.array([10, 20])
print(a * b)
print(a > 2)
print(a[a > 2])
print(a[: , 0])

x = np.arange(0, 6, 0.1)
y1= np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.xlabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()

