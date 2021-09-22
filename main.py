import sys
import warnings
import logging

from small_text.query_strategies import PoolExhaustedException, EmptyPoolException
from Utils.data import get_train_test, preprocess_data
import Utils.helpers as helper

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
        helper.apply_ConfidenceEnhancedLinearSVC(
            train=train, test=test, NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
        # ------------------------------------------ Classifier 2 ------------------------------------------------------
        helper.apply_KNeighbors(train=train, test=test,
                                NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
        # ------------------------------------------ Classifier 3 ------------------------------------------------------
        helper.apply_DecisionTree(train=train, test=test,
                                  NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
        # ------------------------------------------ Classifier 4 ------------------------------------------------------
        helper.apply_RandomForest(train=train, test=test,
                                  NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
        # ------------------------------------------ Classifier 5 ------------------------------------------------------
        helper.apply_AdaBoost(train=train, test=test,
                              NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
        # ------------------------------------------ Classifier 6 ------------------------------------------------------
        helper.apply_MLP(train=train, test=test,
                         NB_ITERATIONS=NB_ITERATIONS, NB_QUERY=NB_QUERY)
        # ----------------------------------------- Table and plot -----------------------------------------------------
        helper.t_table()
        helper.plot(NB_ITERATIONS)
    except PoolExhaustedException:
        print("Erreur! Il ne reste pas assez d'échantillons pour traiter la requête.")
    except EmptyPoolException:
        print("Erreur! Il ne reste plus d'échantillons. (Le pool non étiqueté est vide).")
