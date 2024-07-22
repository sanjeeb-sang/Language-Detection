from typing import List
from naive_bayes import NaiveBayes

"""
Author: Sanjeeb
"""
NGRAMS_NUMBER: int = 3


# Tokenizes text into character n-grams; applies case folding
def tokenize(text, n=3):
    text = text.lower()
    return [text[i:i + n] for i in range(len(text) - (n - 1))]


# A most-frequent class baseline
class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True,
                          key=lambda x: klass_freqs[x])[0]

    def classify(self, test_instance):
        return self.mfc


def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # Method will be one of 'baseline', 'lr', 'nb', or 'nbse'
    method = sys.argv[1]

    # Getting the file name of the train docs, train_classes, and test docs
    train_texts_filename = sys.argv[2]
    train_klasses_filename = sys.argv[3]
    test_texts_filename = sys.argv[4]

    # Reading all lines in the train docs, train classes, and test docs
    train_texts = [x.strip() for x in open(train_texts_filename, encoding='utf8')]
    train_klasses = [x.strip() for x in open(train_klasses_filename, encoding='utf8')]
    test_texts = [x.strip() for x in open(test_texts_filename, encoding='utf8')]

    results: List[str] = []

    if method == 'baseline':
        classifier = Baseline(train_klasses)
        results = [classifier.classify(x) for x in test_texts]

    elif method == 'lr':
        # Use sklearn's implementation of logistic regression
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        count_vectorizer = CountVectorizer(analyzer=tokenize)
        train_counts = count_vectorizer.fit_transform(train_texts)
        lr = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                penalty='l2',
                                max_iter=1000,
                                random_state=0)
        clf = lr.fit(train_counts, train_klasses)
        test_counts = count_vectorizer.transform(test_texts)
        results = clf.predict(test_counts)

    elif method == 'nb':
        # Creating an instance of the NaiveBayes classifier.
        naive_bayes = NaiveBayes()
        # Training the Naive Bayes classifier on the training data.
        naive_bayes.train(train_klasses, train_texts, NGRAMS_NUMBER)
        # Predicting the classes of the test texts.
        results = naive_bayes.predict(test_texts)

    elif method == 'nbse':
        # Creating an instance of the NaiveBayes classifier and enabling the use_start_and_end_position_info flag
        # in the NaiveBayes class.
        naive_bayes_with_se = NaiveBayes()
        naive_bayes_with_se.set_enable_start_end_positions_info()
        # Training the Naive Bayes with Start and End of Word classifier on the training data.
        naive_bayes_with_se.train(train_klasses, train_texts, NGRAMS_NUMBER)
        # Predicting the classes of the test texts.
        results = naive_bayes_with_se.predict(test_texts)

    else:
        print("The method provided must be 'baseline', 'lr', 'nb', or 'nbse'. You provided " + method +
              " which is not valid.")

    # Printing the predicted results.
    for r in results:
        print(r)


if __name__ == '__main__':
    main()
