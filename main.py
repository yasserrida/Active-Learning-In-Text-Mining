import sys
import warnings
import logging

from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from Utils.data import get_train_test, preprocess_data
from Utils.helpers import apply_ConfidenceEnhancedLinearSVC, apply_KNeighbors, apply_DecisionTree, apply_RandomForest, apply_AdaBoost, apply_MLP, t_table, plot

if not sys.warnoptions:
    warnings.simplefilter("ignore")


NB_ITERATIONS = 10  # Nombre d'itérations
NB_QUERY = 10  # Nombre des etiquetes par itération


if __name__ == '__main__':
    logging.getLogger('small_text').setLevel(logging.ERROR)
    logging.getLogger('sklearn').setLevel(logging.ERROR)
    print(f"Nombre d'itérations : {NB_ITERATIONS}")
    print(f"Nombre des etiquetes par itération : {NB_QUERY}")

    text_train, text_test = get_train_test()
    train, test = preprocess_data(text_train, text_test)

    try:
        # ------------------------------------------ Classifier 1 ------------------------------------------------------
        apply_ConfidenceEnhancedLinearSVC(
            train=train, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
        # ------------------------------------------ Classifier 2 ------------------------------------------------------
        apply_KNeighbors(train=train, test=test,
                         NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
        # ------------------------------------------ Classifier 3 ------------------------------------------------------
        apply_DecisionTree(train=train, test=test,
                           NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
        # ------------------------------------------ Classifier 4 ------------------------------------------------------
        apply_RandomForest(train=train, test=test,
                           NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
        # ------------------------------------------ Classifier 5 ------------------------------------------------------
        apply_AdaBoost(train=train, test=test,
                       NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
        # ------------------------------------------ Classifier 6 ------------------------------------------------------
        apply_MLP(train=train, test=test,
                  NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
        # ----------------------------------------- Table and plot -----------------------------------------------------
        t_table()
        plot(NB_ITERATIONS)
    except PoolExhaustedException:
        print("Erreur! Il ne reste pas assez d'échantillons pour traiter la requête")
    except EmptyPoolException:
        print("Erreur! Il ne reste plus d'échantillons. (Le pool non étiqueté est vide)")
