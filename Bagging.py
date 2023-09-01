"""
1. Load the `load_wine` data from 'sklearn.datasets'. Exclude Class 2 objects from the data.
Scale the features using the `Standard Scale r` class with default hyperparameters.
Train logistic regression and evaluate the importance of features.
Specify the name of the attribute that turned out to be the least significant.

Note that the target value lies on the `target" key, the matrix of feature objects lies on the `data" key
"""

from sklearn.datasets import load_wine
data = load_wine()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
pd.set_option('display.max_columns', None)

print(data)


X, y = data['data'], data['target']

y = np.expand_dims(y, axis=1)
y.shape

data1 = np.concatenate((X, y), axis=1)
data1

#data2 = np.delete(data1, np.where(data1[:, -1] == 2), axis=0)

data_new = data1[data1[:, -1] !=2]

X_cut = np.delete(data_new, 13, axis=1)
X_cut.shape

y_cut = np.delete(data_new, np.s_[0:13], axis=1)
y_cut.shape

y_target = np.squeeze(y_cut, axis=-1)
y_target.shape

X_data = pd.DataFrame(data=X_cut, columns=data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, shuffle=False)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaler = scaler.transform(X_train)
clf_skaler = LogisticRegression()
clf_skaler.fit(X_train_scaler, y_train)
weight_sorted = sorted(zip(clf_skaler.coef_.ravel(), data.feature_names), reverse=True)
weights_scaler = [x[0] for x in weight_sorted]
features_scaler = [x[1] for x in weight_sorted]
df_scaler = pd.DataFrame({'features_scaler':features_scaler, 'weights_scaler':weights_scaler})

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6)) 
df_scaler.plot.barh(x='features_scaler', y='weights_scaler', color='skyblue', legend=False, ax=ax)
plt.title('Feature Weights (Scaled)')
plt.xlabel('Weight')
plt.ylabel('Features')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
for i, v in enumerate(df_scaler['weights_scaler']):
    ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('output.png', dpi=300)
plt.show()

"""
2.Exclude objects corresponding to Class 2 from the training part.
!!Don't scale the signs!!.
Train logistic regression with default hyperparameters.
Choose a feature from the suggested ones, which corresponds to the minimum weight.

"""

clf = LogisticRegression()
clf.fit(X_train, y_train)
clf.coef_
weight_table = sorted(zip(clf.coef_.ravel(), data.feature_names), reverse=True)
weight_table
weights = [x[0] for x in weight_table]
features = [x[1] for x in weight_table]
df = pd.DataFrame({'features':features, 'weights':weights})
ax = df.plot.barh(x='features', y='weights', rot=0,)


"""
3. The problem of binary classification is solved.
The matrix objects features X and answers for objects y are given.
Train logistic regression and predict the class of the x_new object
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X1 = np.array([[1, 1], [0.3, 0.7], [0, 4], [-2, -7], [0, -2], [-1, -1], [-2, 0]])
y1 = np.array([1, 1, 1, 0, 0, 0, 0])
X_new = np.array([[-5, 1]])

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.1)
y_test1
X_test1
logreg = LogisticRegression()
logreg.fit(X_train1, y_train1)
X_new_pred = logreg.predict_proba(X_new)
X_new_pred
score = accuracy_score(X_new_pred, y_test1)
score

"""### Классификация текстов

4. Загрузите файл SMSSpamCollection из UCI (https://archive.ics.uci.edu/ml/machine-learning-databases/00228/). Данные содержат текстовую информацию и бинарное целевое значение (‘spam’, ‘ham’), Пусть в обучающую часть попадут первые 4000 объектов из таблицы, в тестовую часть оставшиеся объекты. Обучите `TfidfVectorizer` с гиперпараметрами по умолчанию на текстах из обучающей части и получите векторное представление для объектов обучающей и тестовой части. Укажите полученное число признаков.

Чтобы загрузить данные, скачайте файл по ссылке. Если вы используете google colab, то пример загрузки данных приведен ниже.
"""
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

path = r"C:\Users\User\Documents\pyton-projects\spider\Машинное обучение\logRegression\spam.csv"
print(path.replace("\\", "/"))

text = pd.read_csv('C:/Users/User/Documents/pyton-projects/spider/Машинное обучение/logRegression/spam.csv', encoding='latin-1')
text.head()
text.columns
text = text.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
text.head()
text = text.rename(columns={'v1':'target', 'v2':'data'})
text.head()
text.shape

def bow(vectorizer, train, test):
    train_bow = vectorizer.fit_transform(train)
    test_bow = vectorizer.transform(test)
    return train_bow, test_bow
    
massege = text['data']
y = text['target']

X_train = massege.iloc[:4000]
X_train.tail()
y_train = y.iloc[:4000]
y_train.shape
X_test = massege.iloc[4000:]
X_test.shape
y_test = y.iloc[4000:]

#X_train, X_test, y_train, y_test = train_test_split(massege, y, test_size=0.28212491, shuffle=False)

X_train.shape
X_test = X_test.reset_index(drop=True)
X_test.shape
X_test.head()
y_test = y_test.reset_index(drop=True)
y_test.shape

def preprocess_text(texts):
    stop_words = set(stopwords.words('english'))
    regex = re.compile('[^a-z A-Z]')
    preprocess_texts = []
    for i in tqdm.tqdm(range(len(texts))):
        text = texts[i].lower()
        text = regex.sub(' ', text)
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        preprocess_texts.append( ' '.join(filtered_sentence))
        
    return preprocess_texts


X_train_preprocess = preprocess_text(X_train)
X_test_preprocess = preprocess_text(X_test)

from nltk.stem.lancaster import LancasterStemmer

def stemming_texts(texts):
    st = LancasterStemmer()
    stem_text = []
    for text in tqdm.tqdm(texts):
        word_tokens = word_tokenize(text)
        stem_text.append(' '.join([st.stem(word) for word in word_tokens]))
    return stem_text

X_train_stemming = stemming_texts(X_train_preprocess)
X_test_stemming = stemming_texts(X_test_preprocess)

vectorizer_tfidf = TfidfVectorizer()

X_train_bow, X_test_bow = bow(vectorizer_tfidf, X_train_preprocess, X_test_preprocess)
X_train_bow.shape
X_test_bow.shape

"""5.  Загрузите файл SMSSpamCollection из UCI (https://archive.ics.uci.edu/ml/machine-learning-databases/00228/). Данные содержат текстовую информацию и бинарное целевое значение (‘spam’, ‘ham’), Пусть в обучающую часть попадут первые 4000 объектов из таблицы, в тестовую часть оставшиеся объекты. Обучите `TfidfVectorizer`, помимо слов входящих в тексты, учитывайте биграммы (используйте гиперпараметр `ngram_range`). Укажите полученное число признаков."""

vectorizer_ngram = TfidfVectorizer(ngram_range=(1, 2))
X_train_ngram, X_test_ngram = bow(vectorizer_ngram, X_train_preprocess, X_test_preprocess)
X_train_ngram.shape
X_test_ngram.shape



"""6. Загрузите файл SMSSpamCollection из UCI (https://archive.ics.uci.edu/ml/machine-learning-databases/00228/). Данные содержат текстовую информацию и бинарное целевое значение (‘spam’, ‘ham’), Пусть в обучающую часть попадут первые 4000 объектов из таблицы, в тестовую часть оставшиеся объекты. Обучите `TfidfVectorizer`, не учитывайте слова, которые встретились меньше 2 раз в обучающей выборке (используйте гиперпараметр `min_df`). Укажите полученное число признаков."""

vectorizer_min_df = TfidfVectorizer(min_df=2)
X_train_df, X_test_df = bow(vectorizer_min_df, X_train_preprocess, X_test_preprocess)
X_train_df.shape
X_test_df.shape

"""7. Загрузите файл SMSSpamCollection из UCI (https://archive.ics.uci.edu/ml/machine-learning-databases/00228/). Данные содержат текстовую информацию и бинарное целевое значение (‘spam’, ‘ham’), Пусть в обучающую часть попадут первые 4000 объектов из таблицы, в тестовую часть оставшиеся объекты. Обучите `TfidfVectorizer` с гиперпараметрами по умолчанию на текстах из обучающей части и получите векторное представление для объектов обучающей и тестовой части. На полученных векторных представлениях обучите логистическую регрессию и оцените долю правильных ответов на тестовой части. Укажите полученное значение доли правильных ответов."""

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train_bow, y_train)
y_redict = clf.predict(X_test_bow)
accuracy = accuracy_score(y_redict, y_test)
print('Logistic accuracy: ', accuracy)
