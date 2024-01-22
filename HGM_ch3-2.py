'''''''''
선형회귀
'''''''''

# k-최근접 이웃의 한계
import numpy as np

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )

from sklearn.model_selection import train_test_split
# 훈련 세트와 테스트 세트로 나누기
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)
# 훈련 세트와 테스트 세트 -> 2차원 배열로 변환
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# 모델 훈련
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)
print(knr.predict([[50]]))  # 길이가 50인 농어의 무게를 예측


import matplotlib.pyplot as plt
# 50cm 농어의 이웃 구하기
distances, indexes = knr.kneighbors([[50]])

# 훈련 세트의 산점도
plt.scatter(train_input, train_target)
# 훈련 세트 중에서 이웃 샘플만 다시 plotting
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
plt.scatter(50, 1033, marker='^')  # 50cm 농어 데이터
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 이웃 샘플의 타깃의 평균
print(np.mean(train_target[indexes]))
# 훈련 세트의 범위를 벗어난 새로운 샘플 = 길이 100cm
print(knr.predict([[100]]))  # 길이 50cm인 농어와 동일하게 예측

# 다시 plotting 해서 확인해보기
distances, indexes = knr.kneighbors([[100]])  # 100cm 농어의 이웃 구하기
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
plt.scatter(100, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()






# 이럴 경우, 아예 다른 알고리즘을 사용해본다. => 선형 회귀
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
# 모델 훈련 (모델 = 선형회귀)
lr.fit(train_input, train_target)
print(lr.predict([[50]]))  # 50cm 농어에 대한 예측
print(lr.coef_, lr.intercept_)  # 선형회귀식의 parameter

plt.scatter(train_input, train_target)
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))



# 그런데, 일차방정식 (직선)으로는 잘 안 맞는 것 같다. => 차수 늘려보기 (다항회귀)
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
print(train_poly.shape, test_poly.shape)

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))  # 다항회귀(2차식)으로 길이 50cm인 농어 예측
print(lr.coef_, lr.intercept_)  # 다항회귀식의 parameter


# 구간별 직선을 그리기 위해 -> 15~49까지 정수 배열 만들기
point = np.arange(15, 50)
# 훈련 세트의 산점도
plt.scatter(train_input, train_target)

# 15~49까지 2차 방정식 그래프 plotting
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
plt.scatter([50], [1574], marker='^')  # 50cm 농어 데이터
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 다항회귀식(2차 방정식)의 결정계수 R^2
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
