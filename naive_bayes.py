from typing import List, Dict
from klass import Klass
from nb_predict import NaiveBayesPredictionContext, predict_class, create_prior_probability_dict
"""
Author: Sanjeeb  
"""


def generate_ngrams(word_to_convert: str, size_of_n: int) -> List[str]|None:
    """
    Method to convert the passed @word_to_convert to a list of string ngrams.
    :param word_to_convert: the word to convert to list of ngrams.
    :param size_of_n: the length of the ngrams.
    :return: a list containing all ngrams that can be created from the passed @word_to_convert
    """
    index: int = 0
    ngrams: List[str] = []

    if word_to_convert is None or len(word_to_convert) == 0:
        return None

    while (index + size_of_n) <= len(word_to_convert):
        end_index: int = index + size_of_n
        n_gram: str = word_to_convert[index: end_index]
        index += 1
        ngrams.append(n_gram)

    return ngrams


def process_training_data(train_klasses: List[str], train_texts: List[str], n_gram_number: int,
                          klass_dict: Dict[str, Klass]) -> None:
    """
    Method to process the training data.
    :param train_klasses: list of klasses read from the training data file.
    :param train_texts: list of texts read from the training data file.
    :param n_gram_number: the ngram number. It could be 2 (bigram) or 3 (trigram).
    :param klass_dict: the dictionary containing a mapping of the klasses found in the training data.
    :return: None.
    """
    if train_texts is None or train_klasses is None or len(train_texts) == 0 or len(train_klasses) == 0:
        raise Exception("The training texts or the training klasses list is empty.")

    index_klass: int = 0
    index_text: int = 0
    while index_klass < len(train_klasses) and index_text < len(train_texts):
        text_current: str = train_texts[index_text]
        klass_current: str = train_klasses[index_klass]
        if text_current is not None and klass_current is not None and len(text_current) > 0 and len(klass_current) > 0:
            text_current = text_current.lower()
            list_of_n_grams = generate_ngrams(text_current, n_gram_number)
            if list_of_n_grams is not None and len(list_of_n_grams) > 0:
                if klass_current in klass_dict:
                    klass_object: Klass = klass_dict[klass_current]
                    klass_object.add_ngrams(list_of_n_grams)
                    klass_object.new_training_text_processed()
                else:
                    klass_object: Klass = Klass(klass_current, n_gram_number)
                    klass_object.add_ngrams(list_of_n_grams)
                    klass_object.new_training_text_processed()
                    klass_dict[klass_current] = klass_object

            index_klass += 1
            index_text += 1


def build_vocabulary_of_all_classes(klasses_dict: Dict[str, Klass]) -> set[str]:
    """
    Method to build a single vocabulary set by combining the vocabularies of all klasses.
    :param klasses_dict: dictionary containing all the klass objects
    :return: set containing all vocabularies
    """
    assert klasses_dict is not None
    all_vocab: set[str] = set()
    total_count: int = 0
    total_count2: int = 0
    total_training: int = 0
    for klass_str, klass_value in klasses_dict.items():
        total_count2 += klass_value.get_total_ngrams()
        total_training += klass_value.get_number_of_training_texts()
        if klass_value is not None and klass_value.get_ngrams_dictionary() is not None:
            for vocab_str, vocab_count in klass_value.get_ngrams_dictionary().items():
                total_count += vocab_count
                all_vocab.add(vocab_str)

    return all_vocab


def append_start_end_positions_info_to_train_texts(train_texts: List[str]) -> List[str]:
    """
    Method to append the start and end positions info to the training texts.
    :param train_texts: the texts which will be used for training data.
    :return: new list containing the passed texts with word start and end positions info.
    """
    assert train_texts is not None
    updated_list: List[str] = []
    for text in train_texts:
        updated_list.append("<" + text + ">")

    return updated_list


class NaiveBayes:
    """
    Class that provides an implementation for the Naive Bayes algorithm to find the klass of a data.
    """

    # This dictionary stores all the klasses (and various information related to those klasses)
    # found in the training data.

    def __init__(self):
        """
        Method to initialize NaiveBayes class objects.
        """
        self._klasses_dict: Dict[str, Klass] = {}
        self._ngrams_number = 0
        self._total_texts_processed: int = 0
        self._enable_start_end_positions_info: bool = False
        self._vocabulary_of_all_classes: set[str] = set()

    def train(self, train_klasses: List[str], train_texts: List[str], n_grams_number: int) -> None:
        """
        Method to train the algorithm with the training data. After the training ends then this class can predict the
        classes of other (test) data based on the training data.
        :param train_klasses: list of klasses read from the training data.
        :param train_texts: list of texts read from the training data.
        :param n_grams_number: the ngram number. It could be 2 (bigram), 3 (trigram), or more.
        """
        self._klasses_dict = {}
        self._ngrams_number = n_grams_number
        if self._enable_start_end_positions_info:
            train_texts = append_start_end_positions_info_to_train_texts(train_texts)

        process_training_data(train_klasses, train_texts, n_grams_number, self._klasses_dict)
        self._total_texts_processed = min(len(train_klasses), len(train_texts))
        self._vocabulary_of_all_classes = build_vocabulary_of_all_classes(self._klasses_dict)

    def set_enable_start_end_positions_info(self):
        self._enable_start_end_positions_info = True

    def predict(self, test_texts: List[str]) -> List[str]:
        """
        Method to predict the classes of a list of test text data after learning from the training data.
        :param test_texts: the list of test strings whose classes need to be predicted.
        :return: the predicted classes of the test data.
        """
        results: List[str] = []
        nb_context: NaiveBayesPredictionContext = self.get_naive_bayes_context()
        for test_word in test_texts:
            if self._enable_start_end_positions_info:
                test_word = "<" + test_word + ">"
            max_probability_klass: Klass = predict_class(generate_ngrams(test_word, self._ngrams_number), nb_context)
            if max_probability_klass is None:
                results.append("None")
            else:
                results.append(max_probability_klass.get_class_label())

        return results

    def get_naive_bayes_context(self) -> NaiveBayesPredictionContext:
        prior_prob_dictionary: Dict[str, float] = create_prior_probability_dict(self._klasses_dict,
                                                                                self._total_texts_processed)
        highest_prob_class_label: str = max(prior_prob_dictionary.items(), key=lambda x: x[1])[0]
        total_vocabulary_size: int = len(self._vocabulary_of_all_classes)
        return (NaiveBayesPredictionContext(
            all_vocabulary=self._vocabulary_of_all_classes,
            klasses_dict=self._klasses_dict,
            total_documents_processed=self._total_texts_processed,
            ngram_number=self._ngrams_number,
            prior_probability_dict=prior_prob_dictionary,
            highest_prior_prob_klass=self._klasses_dict[highest_prob_class_label],
            total_vocabulary_size=total_vocabulary_size))
