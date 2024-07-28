# Language-Detection
This is a Python Machine Learning project that uses Naive Bayes and Logistic Regression from scikit-learn to predict the language of words.

Both of these models use ngrams, including 2-grams, 3-grams, and 4-grams, to make better predictions.


# Language-Detection

## Overview
Language-Detection is a Python-based machine learning project that aims to identify the language of given text inputs. The project leverages two primary machine learning algorithms: Naive Bayes and Logistic Regression, using the `scikit-learn` library. To enhance the accuracy of predictions, the models utilize n-grams (specifically 2-grams, 3-grams, and 4-grams).

## Features
- **Naive Bayes Classifier**: A probabilistic model that applies Bayes' theorem with strong (naive) independence assumptions.
- **Naive Bayes with Smoothing**: An enhancement of the basic Naive Bayes classifier that includes smoothing to handle zero-frequency issues.
- **Logistic Regression**: A statistical method for binary classification, extended to multi-class classification through techniques such as one-vs-rest or softmax regression.

## N-Gram Model
The project employs n-grams to capture context and enhance prediction accuracy. An n-gram is a contiguous sequence of 'n' items from a given sample of text. The models in this project specifically utilize:
- **2-grams**: Pairs of consecutive words/characters.
- **3-grams**: Triplets of consecutive words/characters.
- **4-grams**: Quartets of consecutive words/characters.

## Installation

### Prerequisites
- Python 3.6 or higher
- `scikit-learn`
- `numpy`
- `pandas`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/sanjeeb-sang/Language-Detection
   cd Language-detection
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Data Preparation**: Ensure your data is in a suitable format with separate files for text data and another file for labels (languages). More information in the **Input or Data Files**

2. **Training the Model**: You can train the model and generate output classes for test values using the provided script:
   Naive Bayes
   ```bash
   python classify.py nb path_to_training_text path_to_training_classes path_to_test_dataset
   ```
    Logictic Regression
   ```bash
   python classify.py lr path_to_training_text path_to_training_classes path_to_test_dataset
   ```
    Naive Bayes with Smoothing
   ```bash
   python classify.py nbse path_to_training_text path_to_training_classes path_to_test_dataset
   ```

## Advanced Usage
You can directly import NaiveBayes() class from [naive_bayes.py](naive_bayes.py) and use it for custom classification, similar to the python script shown below, which is from [classify.py](classify.py) file.
The script below shows training and test datasets being loaded from files provided in command line.
```python
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

 ```
Here, the training and test datasets are used to create NaiveBayes models and make predictions.
```python
    if method == 'nb':
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
```
## Input or Data Files
1. **Train Texts Files - train_text_file** : File that contains words in many languages. One word per line. This file will be used for training.
2. **Train Class Files - train_class_file** : File that contains the language of the word provided in the train_text_file. One word per line. For example, the language in line 39 in train_class_file is the actual language of the 39th word in train_text_file.
3. **Test Text Files - test_text_file** : File that contains words in many languages. One word per line. This file will be used for making predictions and testing the model.
   
## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any feature requests or bug reports.

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- This project uses the [scikit-learn](https://scikit-learn.org/stable/) library.
- Special thanks to all contributors and users who help improve the project.
