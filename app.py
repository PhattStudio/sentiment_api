#app.py

from flask import Flask, request, jsonify
import nltk,codecs
import re
from pythainlp.tokenize import word_tokenize

review_features = []

# get all individual words in a list of reviews --------------------------------
def get_words_in_reviews(all_reviews):
  all_words = []
  for (words, sentiment) in all_reviews:
    all_words.extend(words)
  return all_words


# get word features in a list of words -----------------------------------------
def get_word_features(list_of_words):
  list_of_words = nltk.FreqDist(list_of_words)
  word_features = list_of_words.keys()
  return word_features


# extract features -------------------------------------------------------------
def extract_features(document):
  document_words = set(document)
  features = {}
  for word in review_features:
    features['contains(%s)' % word] = (word in document_words)
  return features


# removes punctuation that's paired with a word in a list ----------------------
# def remove_punctuation(list_of_words):
#   rm_punct = []
#   for w in list_of_words:
#     rm_punct.append(re.sub('([a-z]+)[?:!.,;"]*',r'\1',w))
#   return rm_punct


def initialize():
  # open example positive and negative reviews ---------------------------------
  neu_reviews_file = codecs.open('keyword/neutral.txt', 'r', "utf-8")
  pos_reviews_file = codecs.open('keyword/pos.txt', 'r', "utf-8")
  neg_reviews_file = codecs.open('keyword/neg.txt', 'r', "utf-8")

  neu_reviews = []
  for each_review in neu_reviews_file:
    each_review = ' '.join(word_tokenize(each_review))
    if each_review.endswith('\n'):
      each_review = each_review[:-1]
    if not each_review == '':
      neu_reviews.append([each_review, 'neutral'])

  # store positive reviews into a list -----------------------------------------
  pos_reviews = []
  for each_review in pos_reviews_file:
    each_review = ' '.join(word_tokenize(each_review))
    if each_review.endswith('\n'):
      each_review = each_review[:-1]
    if not each_review == '':
      pos_reviews.append([each_review, 'positive'])

  # store negative reviews into a list -----------------------------------------
  neg_reviews = []
  for each_review in neg_reviews_file:
    each_review = ' '.join(word_tokenize(each_review))
    if each_review.endswith('\n'):
      each_review = each_review[:-1]
    if not each_review == '':
      neg_reviews.append([each_review, 'negative'])

  # remove words whose length is < 3 and combine both lists --------------------
  all_reviews = []
  for (review, sentiment) in pos_reviews + neg_reviews + neu_reviews:
    reviews_filtered = [w.lower() for w in word_tokenize(review) if len(w) >= 3]
    all_reviews.append((reviews_filtered, sentiment))

  # get feature set-------------------------------------------------------------
  global review_features
  review_features = get_word_features(get_words_in_reviews(all_reviews))
  # review_features = remove_punctuation(review_features)

  # get training set -----------------------------------------------------------
  training_set = nltk.classify.apply_features(extract_features, all_reviews)
  classifier = nltk.NaiveBayesClassifier.train(training_set)
  return classifier


app = Flask(__name__) #create the Flask app

@app.route('/sentiment', methods=['POST']) #GET requests will be blocked
def json_example():
    req_data = request.get_json()
    message = req_data['message']
    removeSpecialChars = message.translate({ord(c): "" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})

    classifier = initialize()
    tokenize = word_tokenize(removeSpecialChars)
    label = classifier.classify(extract_features(word_tokenize(removeSpecialChars)))

    return jsonify({'sentiment_label': label,
                    'word_tokenize': tokenize})

if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000