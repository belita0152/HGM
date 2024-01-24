'''''''''
특성 공학과 규제
'''''''''

# 데이터 준비
import pandas as pd
df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)

import numpy as np
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
)

# 데이터셋 -> train set/test set로 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42)

# scikit-learn의 변환기 transformer(특성을 만들거나, 전처리하는 class)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures()
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))

poly = PolynomialFeatures(include_bias=False)
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))

poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)
poly.get_feature_names_out()
test_poly = poly.transform(test_input)


# 다중 회귀 모델 훈련하기
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)

lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))  # output = -144.40564483377855

# 특성의 개수를 늘리면 선형 모델을 강력해지고, 훈련세트에 대해 거의 완벽하게 학습한다.
# 하지만, 훈련세트에 너무 overfitting 되어 -> 테스트 세트에서는 제대로 예측하지 X
# 보통 model.score() 값이 음수로 나오는 경우: 샘플 개수 < feature 개수 -> overfitting이 발생하는 경우다.

# 따라서, 모델이 훈련세트를 과도하게 학습하지 못하도록 훼방 = 즉 overfitting 되지 않도록 규제 regularization 해야 한다.

# 규제 regularization
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 선형 회귀 모델에 규제를 추가한 모델 2가지 => 1) 릿지 Ridge() 2) 라쏘 Lasso()

# 릿지모델 Ridge()
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# 적절한 alpha 값을 찾기 위해 alpha 값에 대한 R^2 값의 그래프를 plotting
import matplotlib.pyplot as plt
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 릿지 모델
    ridge = Ridge(alpha=alpha)
    # 모델 훈련
    ridge.fit(train_scaled, train_target)
    # 훈련점수와 테스트점수 저장
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))


plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
# 그래프로 확인해본 결과, alpha=0.1일 때 optimal

ridge = Ridge(alpha=0.1) # alpha 값은 hyperparameter -> 사전에 직접 지정해주어야 한다
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target)) # 0.99
print(ridge.score(test_scaled, test_target)) # 0.98
# score 비교 => 모델이 overfitting or underfitting 없이 적절히 훈련되었음을 확인.


# 라쏘모델 Lasso()
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    # 라쏘모델
    lasso = Lasso(alpha=alpha, max_iter=10000)
    # 모델훈련
    lasso.fit(train_scaled, train_target)
    # 훈련점수와 테스트점수 저장
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))

# 적절한 alpha값을 찾기 위해 alpha값에 대한 R^2값의 그래프를 plotting
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
# plotting 결과 => alpha = 10일 때 optimal

lasso = Lasso(alpha=10)  # alpha는 hyperparameter -> 사전에 직접 지정해주어야 한다
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))  # 0.98
print(lasso.score(test_scaled, test_target))  # 0.98

# score 비교 => overfitting or underfitting 없이 모델이 적절히 훈련되었음을 확인함.


# 라쏘 모델은 계수 값을 0으로 만들 수 있음
# 모델에 입력해준 55개의 feature 중 실제로 모델이 사용한 feature 개수를 확인하고자 함
print(np.sum(lasso.coef_ == 0))  # 40

# 즉, 라쏘모델은 입력해준 55개의 feature 중 40개를 사용하지 않고, 55-40=15개의 feature만을 사용하였음.
