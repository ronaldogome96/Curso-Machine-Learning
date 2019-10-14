import pandas as pd

base = pd.read_csv('plano_saude2.csv')

X = base.iloc[:, 0:1].values
y = base.iloc[:, 1:2].values

#Teste com  kernel linear
from sklearn.svm import SVR
regressor_linear = SVR(kernel = 'linear')
regressor_linear.fit(X, y)

#Grafico do kernel linear
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor_linear.predict(X), color = 'red')
#Mostra o score com o kernel linear
regressor_linear.score(X, y)

#Teste com kernel poly
regressor_poly = SVR(kernel = 'poly', degree = 3)
regressor_poly.fit(X, y)

#Grafico do kernel polly
plt.scatter(X, y)
plt.plot(X, regressor_poly.predict(X), color = 'red')
#Mostra o score com o kernel polly
regressor_poly.score(X, y)


#Para usar o rbf, é necessario escalonar
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

#Teste com kernel rbf
regressor_rbf = SVR(kernel = 'rbf')
regressor_rbf.fit(X, y)

#Grafico do kernel rbf
plt.scatter(X, y)
plt.plot(X, regressor_rbf.predict(X), color = 'red')
regressor_rbf.score(X, y)

#Faz uma serie de transformações para o resultado ser escalonado e depois sair do escalonamento
#Nao roda no meu pc
previsao1 = scaler_y.inverse_transform(regressor_linear.predict(scaler_x.transform(40)))
previsao2 = scaler_y.inverse_transform(regressor_poly.predict(scaler_x.transform(40)))
previsao3 = scaler_y.inverse_transform(regressor_rbf.predict(scaler_x.transform(40)))

