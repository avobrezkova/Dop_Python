import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Генерация датасета
X, Y, coef = datasets.make_regression(n_samples=100000, n_features=20, n_informative=10, n_targets=1,
                                      noise=5, coef=True, random_state=2)

# Стандартизация признаков с использованием scikit-learn
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Установка случайного сида и инициализация весов
np.random.seed(9)
init_W = np.random.randn(X.shape[1])

# Функция расчета среднеквадратичной ошибки (MSE)
def calc_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Функция градиентного спуска
def gradient_descent(W, X, y, iterations, eta=1e-4):
    n = X.shape[0]

    errors = []
    w_list = [W.copy()]
    for i in range(0, iterations):
        y_pred = np.dot(X, W)
        err = calc_mse(y, y_pred)
        errors.append(err)

        dQ = 2 / n * X.T @ (y_pred - y)  # градиент функции ошибки
        W -= (eta * dQ)
        w_list.append(W.copy())

        if i % (iterations / 10) == 0:
            print(f'Итерация: {i}, ошибка {err}')

    print(f'Финальная MSE: {calc_mse(y, np.dot(X, W))}')
    return W, errors, w_list

# Функция стохастического градиентного спуска
def stochastic_gradient_descent(W, X, Y, iterations, eta=1e-4, size=1):
    n = X.shape[0]

    errors = []
    w_list = [W.copy()]
    for i in range(0, iterations):
        train_ind = np.random.randint(X.shape[0], size=size)

        y_pred = np.dot(X[train_ind], W)
        W = W - eta * 2 / Y[train_ind].shape[0] * np.dot(X[train_ind].T, y_pred - Y[train_ind])

        error = calc_mse(Y, np.dot(X, W))
        errors.append(error)
        w_list.append(W)

        if i % (iterations / 10) == 0:
            print(f'Итерация: {i}, ошибка {error}')

    print(f'Финальная MSE: {calc_mse(Y, np.dot(X, W))}')
    return W, errors, w_list

# Применение градиентного спуска
weights_GD, errors_GD, w_list_GD = gradient_descent(init_W, X, Y, iterations=5000, eta=1e-3)

# Применение стохастического градиентного спуска
weights_SGD, errors_SGD, w_list_SGD = stochastic_gradient_descent(init_W, X, Y, iterations=5000, eta=1e-3, size=1)

# Визуализация изменения функционала ошибки
plt.plot(range(len(errors_GD)), errors_GD, color='b', label='GD')
plt.plot(range(len(errors_SGD)), errors_SGD, color='g', label='SGD')

plt.title('MSE')
plt.xlabel('Iteration number')
plt.ylabel('MSE')
plt.xlim(0, 1000)
plt.legend()
plt.show()


w_list_GD = np.array(w_list_GD)
w_list_SGD = np.array(w_list_SGD)

# Визуализируем изменение весов (красной точкой обозначены истинные веса, сгенерированные в начале)
plt.figure(figsize=(13, 6))
plt.title('Compare SGD and GD')
plt.xlabel(r'$w_1$')
plt.ylabel(r'$w_2$')

plt.scatter(w_list_GD[:, 0], w_list_GD[:, 1], color='b')
plt.plot(w_list_GD[:, 0], w_list_GD[:, 1], color='b', label='GD')

plt.scatter(w_list_SGD[:, 0], w_list_SGD[:, 1], color='g')
plt.plot(w_list_SGD[:, 0], w_list_SGD[:, 1], color='g', label='SGD')
plt.scatter(coef[0], coef[1], c='r')

plt.legend()
plt.show()