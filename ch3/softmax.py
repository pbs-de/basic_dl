import numpy as np


# overFlow 문제가 발생할 수 있음
def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))


y = softmax(np.array([0.3, 2.9, 4.0]))
print(y)


# 각 요소 빼기 a의 최대값을 소프트맥스 함수의 인자로 넘기기
def enhanced_softmax(a):
    return np.exp(a - np.max(a)) / np.sum(np.exp(a - np.max(a)))


a = np.array([1010, 1000, 990])
y = enhanced_softmax(a)
print(y)


#softmax function 결과는 공리적 확률의 조건을 따른다.
a = np.array([0.3, 2.9, 4.0])
y = enhanced_softmax(a)
print(y)
print(np.sum(y))
#즉 각 출력 원소를 확률로 해석하여 제일 높은 확률을 선택한다. 그를 통해 [분류]한다.
