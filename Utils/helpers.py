import matplotlib.pyplot as plt
import numpy as np
from texttable import Texttable

from small_text.active_learner import PoolBasedActiveLearner
from small_text.query_strategies import RandomSampling, LeastConfidence, PredictionEntropy
from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.classifiers.factories import SklearnClassifierFactory

from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


random_accuracys = ['Random Sampling']
least_accuracys = ['Least Confidence']
entory_accuracys = ['Entropy']
accuracys = ['Without Active']
clf_templates = ["Strategy / Classifier", "Linear SVC", "K Neighbors",
                 "Decision Tree", "Random Forest", "AdaBoost", "MLP"]
query_strategys = ["RandomSampling", "LeastConfidence",
                   "PredictionEntropy"]
colors = ['blue', 'red', 'green', 'pink', 'black', 'orange']


# -- obtenir le score de test
def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)
    # print('\tTrain accuracy: {:.2f}'.format(f1_score(y_pred, train.y, average='micro')) +
    # '\tTest accuracy: {:.2f}'.format(f1_score(y_pred_test, test.y, average='micro')))
    return f1_score(y_pred_test, test.y, average='micro')


# -- Effectuer l'apprentissage actif
def perform_active_learning(active_learner, train, labeled_indices, test, NB_ITERATIONS, NB_QUERY):
    accuracy = []
    for i in range(NB_ITERATIONS):
        q_indices = active_learner.query(num_samples=NB_QUERY)
        y = train.y[q_indices]
        active_learner.update(y)
        labeled_indices = np.concatenate([q_indices, labeled_indices])
        accuracy.append(evaluate(active_learner, train[labeled_indices], test))
        # print('Iteration {:d} ({} échantillons)'.format(i+1, len(labeled_indices)))
    return accuracy


# -- Effectuer l'apprentissage normale
def perform_learning(active_learner, train, labeled_indices, test):
    accuracy = []
    accuracy.append(evaluate(active_learner, train[labeled_indices], test))
    return accuracy


# -- Obtenir les étiquettes initiales
def initialize_active_learner(active_learner, y_train):
    indices_pos_label = np.where(y_train == 1)[0]
    indices_neg_label = np.where(y_train == 0)[0]
    x_indices_initial = np.concatenate([np.random.choice(indices_pos_label, 10, replace=False),
                                        np.random.choice(indices_neg_label, 10, replace=False)])
    x_indices_initial = x_indices_initial.astype(int)
    y_initial = [y_train[i] for i in x_indices_initial]
    active_learner.initialize_data(x_indices_initial, y_initial)
    return x_indices_initial


# -- Appliquer l'apprentissage actif avec le classificateur LinearSVC
def apply_ConfidenceEnhancedLinearSVC(train, test, NB_ITERATIONS, NB_QUERY):
    print(f'\nClassifier: Linear SVC')
    clf_template = ConfidenceEnhancedLinearSVC()
    clf_factory = SklearnClassifierFactory(clf_template)

    active_learner = PoolBasedActiveLearner(
        clf_factory, RandomSampling(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(
        active_learner=active_learner, train=train, labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    random_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Random Sampling\t {round(max(acc),3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, LeastConfidence(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    least_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Least Confidence\t {round(max(acc),3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    entory_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Entropy\t\t {round(max(acc),3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_learning(active_learner=active_learner, train=train,
                           labeled_indices=labeled_indices, test=test)
    accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: without Active\t {round(max(acc),3)}')


# -- Appliquer l'apprentissage actif avec le classificateur KNeighbors
def apply_KNeighbors(train, test, NB_ITERATIONS, NB_QUERY):
    print(f'\nClassifier: K Neighbors ')
    clf_template = KNeighborsClassifier()
    clf_factory = SklearnClassifierFactory(clf_template)

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    random_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Random Sampling\t {round(max(acc),3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, LeastConfidence(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    least_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Least Confidence\t {round(max(acc),3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    entory_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Entropy\t\t {round(max(acc),3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_learning(active_learner=active_learner, train=train,
                           labeled_indices=labeled_indices, test=test)
    accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: without Active\t {round(max(acc),3)}')


# -- Appliquer l'apprentissage actif avec le classificateur DecisionTree
def apply_DecisionTree(train, test, NB_ITERATIONS, NB_QUERY):
    print(f'\nClassifier: Decision Tree')
    clf_template = DecisionTreeClassifier()
    clf_factory = SklearnClassifierFactory(clf_template)

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    random_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Random Sampling\t {round(max(acc), 3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, LeastConfidence(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    least_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Least Confidence\t {round(max(acc), 3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    entory_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Entropy\t\t {round(max(acc), 3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_learning(active_learner=active_learner, train=train,
                           labeled_indices=labeled_indices, test=test)
    accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: without Active\t {round(max(acc),3)}')


# -- Appliquer l'apprentissage actif avec le classificateur RandomForest
def apply_RandomForest(train, test, NB_ITERATIONS, NB_QUERY):
    print(f'\nClassifier: Random Forest')
    clf_template = RandomForestClassifier()
    clf_factory = SklearnClassifierFactory(clf_template)

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    random_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Random Sampling\t {round(max(acc), 3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, LeastConfidence(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    least_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Least Confidence\t {round(max(acc), 3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    entory_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Entropy\t\t {round(max(acc), 3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_learning(active_learner=active_learner, train=train,
                           labeled_indices=labeled_indices, test=test)
    accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: without Active\t {round(max(acc),3)}')


# -- Appliquer l'apprentissage actif avec le classificateur AdaBoost
def apply_AdaBoost(train, test, NB_ITERATIONS, NB_QUERY):
    print(f'\nClassifier: AdaBoost')
    clf_template = AdaBoostClassifier()
    clf_factory = SklearnClassifierFactory(clf_template)

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    random_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Random Sampling\t {round(max(acc), 3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, LeastConfidence(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    least_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Least Confidence\t {round(max(acc), 3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    entory_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Entropy\t\t {round(max(acc), 3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_learning(active_learner=active_learner, train=train,
                           labeled_indices=labeled_indices, test=test)
    accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: without Active\t {round(max(acc),3)}')


# -- Appliquer l'apprentissage actif avec le classificateur MLP
def apply_MLP(train, test, NB_ITERATIONS, NB_QUERY):
    print(f'\nClassifier: MLP')
    clf_template = MLPClassifier(max_iter=3000)
    clf_factory = SklearnClassifierFactory(clf_template)

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    random_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Random Sampling\t {round(max(acc), 3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, LeastConfidence(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    least_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Least Confidence\t {round(max(acc), 3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_active_learning(active_learner=active_learner, train=train,
                                  labeled_indices=labeled_indices, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
    entory_accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: Entropy\t\t {round(max(acc), 3)}')

    active_learner = PoolBasedActiveLearner(
        clf_factory, PredictionEntropy(), train)
    labeled_indices = initialize_active_learner(active_learner, train.y)
    acc = perform_learning(active_learner=active_learner, train=train,
                           labeled_indices=labeled_indices, test=test)
    accuracys.append(round(max(acc), 3))
    print(f'\tStrategy: without Active\t {round(max(acc),3)}')


# -- Rendre la matrice des scores
def t_table():
    t = Texttable()
    t.add_rows(rows=[clf_templates,
                     random_accuracys,
                     least_accuracys,
                     entory_accuracys,
                     accuracys])
    print(t.draw())


# -- Rendre le graphique des scores
def plot(NB_ITERATIONS):
    x = clf_templates[1:]
    plt.xlabel("Classifieur")
    plt.ylabel('Précision')
    plt.plot(x, random_accuracys[1:], color=colors[0], alpha=0.6,
             label="Random", linestyle='-', marker='o')
    plt.plot(x, least_accuracys[1:], color=colors[1], alpha=0.6,
             label='Least Conf', linestyle='-', marker='o')
    plt.plot(x, entory_accuracys[1:], color=colors[1], alpha=0.6,
             label='Entropy', linestyle='-', marker='o')
    plt.title('Comparaison entre les différents stratégies | ' +
              str(NB_ITERATIONS) + ' itération')
    plt.grid()
    plt.legend()
    plt.show()
