import numpy as np

print("1차원 배열")
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)
print(type(A.shape))
print(A.shape[0])
print("")

print("2차원 배열")
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)
print("")

print("행렬 곱")
A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(np.dot(A, B)) # 교환법칙은 성립되지 않는다.
print("")

print("1x2 dot 2x1")
A = np.array([3, 4])
B = np.array([[1], [2]])
print(np.dot(A, B))
print("(3 x 1) + (4 X 2)")

print("1x2 dot 1x2")
print(np.dot([1, 2], [3, 4]))
print("(1 x 3) + (2 x 4)")

print("1x2 dot 2x3 => 1 x 3")
X = np.array([1, 2])
print(X.shape)
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W.shape)
Y = np.dot(X, W)
print(Y)

print("3x2 dot 1x2")
X = np.array([[1, 2], [3, 4], [5, 6]])
Y = np.array([7, 8])
print(np.dot(X,Y))