import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)


def transformar_sexo(valor):
    if valor == "female":
        return 1
    else:
        return 0 

train['Sex_binario'] = train['Sex'].map(transformar_sexo)

variaveis = ['Sex_binario', 'Age']

x = train[variaveis]
y = train['Survived']

x = x.fillna(-1) #substitui valores vazios

modelo.fit(x, y)

x_prev = test[variaveis]
test['Sex_binario'] = test['Sex'].map(transformar_sexo)
x_prev = test[variaveis]
x_prev = xprev.fillna(-1)

p = modelo.predict(x_prev)

sub = pd.Series(p, index=test['PassengerId'], name="Survived")

sub.to_csv("primeiro_modelo.csv", header=True)
