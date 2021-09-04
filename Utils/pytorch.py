import logging
import torch
import sys
import warnings
import gc
import numpy as np

from small_text.active_learner import PoolBasedActiveLearner
from small_text.initialization import random_initialization_balanced
from small_text.integrations.transformers import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory, KimCNNFactory
from small_text.query_strategies import PoolExhaustedException, EmptyPoolException, RandomSampling
from small_text.integrations.transformers.datasets import TransformersDataset
from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset

from Utils.data import preprocess_data, get_train_test
from Utils.shared import evaluate
from torchtext.legacy import datasets
from torchtext.legacy import data
from transformers import AutoTokenizer

if not sys.warnoptions:
    warnings.simplefilter("ignore")


NB_ITERATIONS = 10
NB_QUERY = 20
TRANSFORMER_MODEL = TransformerModelArguments('distilroberta-base')
query_strategy = RandomSampling()


def preprocess_data(tokenizer, data, labels, max_length=500):
    data_out = []
    for i, doc in enumerate(data):
        encoded_dict = tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            return_attention_mask=True,
            return_tensors='pt',
            truncation='longest_first'
        )
        data_out.append(
            (encoded_dict['input_ids'], encoded_dict['attention_mask'], labels[i]))
    return TransformersDataset(data_out)


def get_train_test_imdb():
    text_field = data.Field(lower=True, include_lengths=True, batch_first=True)
    label_field = data.Field(sequential=False, dtype=torch.int)
    train, test = datasets.IMDB.splits(text_field, label_field)

    text_field.build_vocab(train, min_freq=1)
    label_field.build_vocab(train)

    train_tc = _dataset_to_text_classification_dataset(train)
    test_tc = _dataset_to_text_classification_dataset(test)
    return train_tc, test_tc


def _dataset_to_text_classification_dataset(dataset):
    assert dataset.fields['text'].vocab.itos[0] == '<unk>'
    assert dataset.fields['text'].vocab.itos[1] == '<pad>'
    unk_token_idx = 1
    vocab = dataset.fields['text'].vocab
    labels = list(set(dataset.fields['label'].vocab.itos))
    labels = np.array(labels)
    data = [(torch.LongTensor([vocab.stoi[token] if token in vocab.stoi else unk_token_idx
                               for token in example.text]),
             dataset.fields['label'].vocab.stoi[example.label])
            for example in dataset.examples]
    return PytorchTextClassificationDataset(data=data, vocab=vocab, target_labels=labels)


def perform_active_learning(active_learner, train, labeled_indices, test):
    accuracy = []
    for i in range(NB_ITERATIONS):
        q_indices = active_learner.query(num_samples=NB_QUERY)
        y = train.y[q_indices]
        active_learner.update(y)
        labeled_indices = np.concatenate([q_indices, labeled_indices])
        print('Iteration #{:d} ({} samples)'.format(i, len(labeled_indices)))
        acc = evaluate(active_learner, train[labeled_indices], test)
        accuracy.append(acc)
    return accuracy


def initialize_active_learner(active_learner, y_train):
    x_indices_initial = random_initialization_balanced(y_train)
    y_initial = np.array([y_train[i] for i in x_indices_initial])
    num_classes = len(np.unique(y_train))
    active_learner.initialize_data(x_indices_initial, y_initial, num_classes)
    return x_indices_initial


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    logging.getLogger('small_text').setLevel(logging.INFO)
    logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)

    train, test = get_train_test()
    classifier_kwargs = dict({'device': 'cpu'})
    clf_factory = TransformerBasedClassificationFactory(
        TRANSFORMER_MODEL, kwargs=classifier_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        TRANSFORMER_MODEL.model, cache_dir='.cache/')
    x_train = preprocess_data(tokenizer, train.data, train.target)
    y_train = train.target

    x_test = preprocess_data(tokenizer, test.data, test.target)
    y_test = test.target

    active_learner = PoolBasedActiveLearner(
        clf_factory, query_strategy, x_train)
    labeled_indices = initialize_active_learner(active_learner, y_train)

    try:
        print(perform_active_learning(active_learner, x_train,
                                      labeled_indices, x_test))
    except PoolExhaustedException as e:
        print('Error! Not enough samples left to handle the query.')
    except EmptyPoolException as e:
        print('Error! No more samples left. (Unlabeled pool is empty)')
