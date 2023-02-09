import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, recall_score, precision_score, roc_curve, roc_auc_score, auc

pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import seaborn as sns

def print_hi():
    df = pd.read_csv('input.csv')
    st.markdown("<h1 style='text-align: center; color: black;'>Прогнозирование послеоперационных осложнений</h1>", unsafe_allow_html=True)
    # load model
    x = df.iloc[:, 0:15]
    y = df.iloc[:, -1]
    # select
    feature_score, x_k, f_s1,drop_list1 = k_best(x, y)
    # обучение на сокращенном наборе k-best
    x_train, x_test, y_train, y_test = scal(x_k, y, size=0.4)
    # обучение на сокращенном наборе k-best
    knn, pred2 = pred('knn', x_train, x_test, y_train)
    acc2 = accuracy_model(y_test, pred2)

    option = st.sidebar.selectbox('Mode', ['Загрузка выборки', 'Обучение', 'Тестирование'])

    if option == "Загрузка выборки":
        st.dataframe(df.head())
        st.sidebar.subheader(' Исследование')
        st.markdown("Установите флажок на боковой панели, чтобы просмотреть набор данных.")
        if st.sidebar.checkbox('Основная информация'):

            if st.sidebar.checkbox("Показать столбцы"):
                st.subheader('Список столбцов')
                all_columns = df.columns.to_list()
                st.write(all_columns)

            if st.sidebar.checkbox('Статистика'):
                st.subheader('Описание статистических данных')
                st.write(df.describe())
            if st.sidebar.checkbox('Пропуски?'):
                st.subheader('Наличие пропусков')
                st.write(df.isnull().sum())

            if st.sidebar.checkbox('Информативные показатели'):
                st.markdown('#### Используя метод фльтрации Хи2 для отбра показателей мы получили следующий результат.\n #### Наиболее информативными показателями являются:')
                st.write(drop_list1)

    elif option == 'Обучение':
        st.sidebar.subheader(' Исследование')
        st.set_option('deprecation.showPyplotGlobalUse', False)

        if st.sidebar.checkbox('Показать набор для обучения'):
            st.write(x_k.head(50))
            st.write('Размер выборки: ', x_k.shape)
            st.write('Статистика: \n', x_k.describe())

        size = st.sidebar.slider('Установите размер тестовой выборки', min_value=0.2, max_value=0.4)
        # обучение на сокращенном наборе k-best
        x_train, x_test, y_train, y_test = scal(x_k, y,size)
        if st.sidebar.checkbox('Вывод обучающих и тестовых наборов'):
            st.write('X_train: ', x_train.shape)
            st.write('y_train: ', y_train.shape)
            st.write('X_test: ', x_test.shape)
            st.write('y_test: ', y_test.shape)

        model = st.sidebar.selectbox('Mode', ['Логистическая регрессия', 'К-ближайший соседей', 'Дерево решений', 'Метод опорных векторов', 'Многослойный персептрон', 'Случайный лес'])
        if model == 'Логистическая регрессия':
            vid = 'log'
            # обучение на сокращенном наборе k-best
            knn, pred2 = pred(vid, x_train, x_test, y_train)
            acc2 = accuracy_model(y_test, pred2)
            t = round(acc2, 3) * 100
            st.success("Точность  модели {0}: {1:9.2f}".format(model,t))

            if st.sidebar.checkbox('Показать матрицу неточности'):
                # матрица неточности
                cnf_matrix = confusion_matrix(y_test, pred2)

                fig, ax = plt.subplots()
                ax.xaxis.set_label_position("top")
                plt.tight_layout()
                plt.title('Матрица неточности \n', y=1.1)
                sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Reds", fmt='g')
                st.write(fig)

        if model == 'К-ближайший соседей':
            vid = 'knn'
            # обучение на сокращенном наборе k-best
            knn, pred2 = pred(vid, x_train, x_test, y_train)
            acc2 = accuracy_model(y_test, pred2)
            t = round(acc2, 3) * 100
            st.success("Точность  модели {0}: {1:9.2f}".format(model,t))
            if st.sidebar.checkbox('Показать матрицу неточности'):
                # матрица неточности
                cnf_matrix = confusion_matrix(y_test, pred2)

                fig, ax = plt.subplots()
                ax.xaxis.set_label_position("top")
                plt.tight_layout()
                plt.title('Матрица неточности \n', y=1.1)
                sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Reds", fmt='g')
                st.write(fig)
        if model == 'Дерево решений':
            vid = 'tree'
            # обучение на сокращенном наборе k-best
            knn, pred2 = pred(vid, x_train, x_test, y_train)
            acc2 = accuracy_model(y_test, pred2)
            t = round(acc2, 3) * 100
            st.success("Точность  модели {0}: {1:9.2f}".format(model,t))
            if st.sidebar.checkbox('Показать матрицу неточности'):
                # матрица неточности
                cnf_matrix = confusion_matrix(y_test, pred2)

                fig, ax = plt.subplots()
                ax.xaxis.set_label_position("top")
                plt.tight_layout()
                plt.title('Матрица неточности \n', y=1.1)
                sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Reds", fmt='g')
                st.write(fig)
        if model == 'Метод опорных векторов':
            vid = 'svm'
            # обучение на сокращенном наборе k-best
            knn, pred2 = pred(vid, x_train, x_test, y_train)
            acc2 = accuracy_model(y_test, pred2)
            t = round(acc2, 3) * 100
            st.success("Точность  модели {0}: {1:9.2f}".format(model,t))
            if st.sidebar.checkbox('Показать матрицу неточности'):
                # матрица неточности
                cnf_matrix = confusion_matrix(y_test, pred2)

                fig, ax = plt.subplots()
                ax.xaxis.set_label_position("top")
                plt.tight_layout()
                plt.title('Матрица неточности \n', y=1.1)
                sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Reds", fmt='g')
                st.write(fig)
        if model == 'Многослойный персептрон':
            vid = 'mlp'
            # обучение на сокращенном наборе k-best
            knn, pred2 = pred(vid, x_train, x_test, y_train)
            acc2 = accuracy_model(y_test, pred2)
            t = round(acc2, 3) * 100
            st.success("Точность  модели {0}: {1:9.2f}".format(model,t))
            if st.sidebar.checkbox('Показать матрицу неточности'):
                # матрица неточности
                cnf_matrix = confusion_matrix(y_test, pred2)

                fig, ax = plt.subplots()
                ax.xaxis.set_label_position("top")
                plt.tight_layout()
                plt.title('Матрица неточности \n', y=1.1)
                sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Reds", fmt='g')
                st.write(fig)
        if model == 'Случайный лес':
            vid = 'for'
            # обучение на сокращенном наборе k-best
            knn, pred2 = pred(vid, x_train, x_test, y_train)
            acc2 = accuracy_model(y_test, pred2)
            t = round(acc2, 3) * 100
            if t > 50:
                custom_emoji = ':blush:'
                st.info('{}'.format(custom_emoji))
                st.success("Точность  модели {0}: {1:9.2f}".format(model,t))
            else:
                custom_emoji = ':confused:'
                st.info('{}'.format(custom_emoji))
                st.success("Точность  модели {0}: {1:9.2f}".format(model,t))
            if st.sidebar.checkbox('Показать матрицу неточности'):
                # матрица неточности
                cnf_matrix = confusion_matrix(y_test, pred2)

                fig, ax = plt.subplots()
                ax.xaxis.set_label_position("top")
                plt.tight_layout()
                plt.title('Матрица неточности \n', y=1.1)
                sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Reds", fmt='g')
                st.write(fig)

    elif option == "Тестирование":
        # Print shape and description of the data
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if st.sidebar.checkbox('Классифицировать пациента?'):
            bilirubin = st.number_input("Билирубин")
            neutrophils = st.number_input("Нейрофилы")
            amylase = st.number_input("Амилазе")
            duration = st.number_input("Длительность операции")
            lymphocytes = st.number_input("Лимфоциты")
            if st.button("Прогноз"):
                # тестирование
                pred9 = knn.predict([[bilirubin, neutrophils, amylase, duration, lymphocytes]])
                #st.success(pred9.round(1)[0])
                if pred9 ==0:
                    custom_emoji = ':blush:'
                    st.info('{}'.format(custom_emoji))
                    st.success("Осложнений нет")
                else:
                    custom_emoji = ':confused:'
                    st.info('{}'.format(custom_emoji))
                    st.success("Осложнения есть")


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
    return feature_score,x_1,f_s1,drop_list1

def scal(x,y,size):
#шкалируем весь набор данных
    scaler = MinMaxScaler(feature_range=(0,1))
#назначение показателейдля шклирования
    x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
# обучающая и тестовая выборки по полным данным
# деление на обучающую и тестовую выборки: 80 % - 20 %
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = size)
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
