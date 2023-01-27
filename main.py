import streamlit as st
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as sts
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure
from IPython.core.pylabtools import figsize
import itertools
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_squared_log_error, r2_score,mean_absolute_percentage_error

matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

def print_hi():
    df = pd.read_csv('input.csv')
    st.title("MLP Predictor")
    st.subheader("Welcome to our Application")
    # load model
    x = df.iloc[:, 0:15]
    y = df.iloc[:, -1]
    # select
    feature_score, x_k, f_s1 = k_best(x, y)
    #scal
    # обучение на сокращенном наборе k-best
    x_train, x_test, y_train, y_test = scal(x_k, y)
    # обучение на сокращенном наборе k-best
    knn, pred2 = pred('knn', x_train, x_test, y_train)
    acc2 = accuracy_model(y_test, pred2)

    if st.button("Analyze"):
        t=round(acc2, 3) * 100
        if t >50:
            custom_emoji = ':blush:'
            st.info('Happy : {}'.format(custom_emoji))
            st.success("Точность модели: {}".format(t))
        else:
            custom_emoji = ':confused:'
            st.info('Confused : {}'.format(custom_emoji))
            st.success("Polarity Score is: {}".format(t))


def k_best(x,y):
    bestfeature=SelectKBest(score_func=chi2,k='all')
    fit=bestfeature.fit(x,y)
    df_score=pd.DataFrame(fit.scores_)
    df_column=pd.DataFrame(x.columns)
    feature_score=pd.concat([df_column,df_score],axis=1)
    feature_score.columns=['Specs', 'Score']
    feature_score=feature_score.sort_values(by='Score', ascending=False)
    print('Значимость показателей')
    print(feature_score)
# выбираем показатели по важности
    f_s1=feature_score[feature_score['Score']>20]
    drop_list1 = f_s1['Specs']
    #drop_list1.head(7)
#feature_score
    x_1=x.loc[:,drop_list1]#3
    print('Используя метод фльтрации Хи2 для отбра показателей мы получили следующий результат. Наиболее информативными показателями являются:')
    print(drop_list1)
    print('Создадим новую выборку состоящую толко из этих показателей.')
    print(x_1)
    return feature_score,x_1,f_s1

def scal(x,y):
#шкалируем весь набор данных
    scaler = MinMaxScaler(feature_range=(0,1))
#назначение показателейдля шклирования
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
# обучающая и тестовая выборки по полным данным
# деление на обучающую и тестовую выборки: 80 % - 20 %
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.3)
    return x_train, x_test, y_train, y_test

# определяем функцию для оценки модели
def accuracy_model(y_test, model_pred):
# доля правильных ответов алгоритма: Точность = (истинное положительное + истинно отрицательное значение) / всего
    acc = accuracy_score(y_test, model_pred)
    print(f"The accuracy score for method is: {round(acc,3)*100}%")
    return acc

def pred(vid, x_train, x_test,y_train):
    if vid == 'log':
        model = LogisticRegression().fit(x_train, y_train)
        pred = model.predict(x_test)
    elif vid == 'knn':
        model = KNeighborsClassifier(n_neighbors = 2).fit(x_train, y_train)
        pred = model.predict(x_test)
    elif vid == 'tree':
        model = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=6, random_state=0).fit(x_train, y_train)
        pred = model.predict(x_test)
    elif vid == 'svm':
        model = SVC().fit(x_train, y_train)
        pred = model.predict(x_test)
    elif vid == 'mlp':
        model = MLPClassifier(activation='relu', solver='lbfgs', alpha=1e-5, max_iter=1000, hidden_layer_sizes=(5, ), random_state=1).fit(x_train, y_train)
        pred = model.predict(x_test)
    elif vid == 'for':
        model = RandomForestClassifier(criterion='entropy',max_depth=2, random_state=1).fit(x_train, y_train)
        pred = model.predict(x_test)
    else:
        print("Такого метода нет")
    return model, pred

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
