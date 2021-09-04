from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from small_text.data import SklearnDataSet


# -- Obtenez la dataset twenty_newsgroups_corpu (Source: sklearn)
def get_train_test():
    categories = ['rec.sport.baseball', 'rec.sport.hockey']
    train = fetch_20newsgroups(subset='train', remove=(
        'headers', 'footers', 'quotes'), categories=categories)
    test = fetch_20newsgroups(subset='test', remove=(
        'headers', 'footers', 'quotes'), categories=categories)
    return train, test


# -- Appliquer la méthode TF-IDF
def preprocess_data(train, test):
    vectorizer = TfidfVectorizer(stop_words='english')
    x_train = normalize(vectorizer.fit_transform(train.data))
    x_test = normalize(vectorizer.transform(test.data))
    return SklearnDataSet(x_train, train.target), SklearnDataSet(x_test, test.target)
