from typing import List, Dict
"""
Author: Sanjeeb 
"""


class Klass:
    """
    Class related to a machine learning Klass(or class) that stores various information related to the language class
    in the training data.
    """

    def __init__(self, class_label: str, n_grams_number: int):
        """
        Initialization method for the Klass class.
        :param class_label: the label of this current klass.
        :param n_grams_number: the length of each ngram. It could be 2 (bigram) or 3 (trigram).
        """
        self._ngrams_dictionary: Dict[str, int] = {}
        self._number_of_total_ngrams: int = 0
        self._class_label: str = class_label
        self._ngram_number: int = n_grams_number
        self._number_of_training_texts_classified_as_class: int = 0

    def get_total_ngrams(self) -> int:
        """
        Method to get the total number of ngrams found in the training data that are related to this klass.
        :return: number of ngrams found in training data related to this klass.
        """
        return self._number_of_total_ngrams

    def get_tokens_count(self):
        """
        Method to get the total number of tokens stored in this class.
        :return: the total number of tokens stored in this class.
        """
        return sum(self._ngrams_dictionary.values())

    def get_vocabulary_size(self) -> int:
        """
        Method to get the vocabulary size of this klass. The vocabulary size of this klass is the number of unique ngrams
        found in the training data.
        :return: the vocabulary of this klass as found in the training data.
        """
        if self._ngrams_dictionary is None:
            return 0
        else:
            return len(self._ngrams_dictionary)

    def find_count(self, ngram_to_check) -> int:
        """
        Method to get the number of times the passed ngram was encountered in the training data.
        :param ngram_to_check: the ngram whose count is to be returned.
        :return: the count of the ngram, or the number of times the passed ngram was encountered in the training data.
        """
        if ngram_to_check in self._ngrams_dictionary:
            return self._ngrams_dictionary[ngram_to_check]
        else:
            return 0

    def add_ngrams(self, list_of_ngrams: List[str]) -> None:
        """
        Method to add a list of ngrams related to this class (found in the training data). These list of ngrams will be
        stored in this klass and will later be used in probability calculations.
        :param list_of_ngrams: the list of ngrams to add to this klass.
        :return: None.
        """
        if list_of_ngrams is None or len(list_of_ngrams) == 0:
            return None
        for ngram in list_of_ngrams:
            if ngram in self._ngrams_dictionary:
                self._ngrams_dictionary[ngram] = self._ngrams_dictionary[ngram] + 1
            else:
                self._ngrams_dictionary[ngram] = 1
            # Total number of ngrams increased by 1 after just adding 1 ngram
            self._number_of_total_ngrams += 1

    def get_number_of_training_texts(self) -> int:
        """
        Method to get the number of training texts that were classified as being of this class.
        :return: the number of training texts which were classified as being of this class.
        """
        return self._number_of_training_texts_classified_as_class

    def new_training_text_processed(self) -> None:
        """
        Method to add the number of training texts that were classified as being of this class.
        :return: None
        """
        self._number_of_training_texts_classified_as_class += 1

    def get_class_label(self):
        """
        Method to get the class label
        :return: the current class label
        """
        return self._class_label

    def get_ngrams_dictionary(self):
        return self._ngrams_dictionary
