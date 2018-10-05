#app.py

from flask import Flask, request, jsonify
import nltk,codecs
# import re
from pythainlp.tokenize import word_tokenize

features = []
def get_words_in_reviews(all_reviews):
  all_words = []
  for (words, sentiment) in all_reviews:
    all_words.extend(words)
  return all_words

def get_word_features(list_of_words):
  list_of_words = nltk.FreqDist(list_of_words)
  word_features = list_of_words.keys()
  return word_features

def extract_features(document):
  document_words = set(document)
  features = {}
  for word in features:
    features['contains(%s)' % word] = (word in document_words)
  return features

def train():
  pos_reviews_file = codecs.open('keyword/pos.txt', 'r', "utf-8")
  neg_reviews_file = codecs.open('keyword/neg.txt', 'r', "utf-8")

  pos_reviews = []
  for each_review in pos_reviews_file:
    each_review = ' '.join(word_tokenize(each_review)) # ตัดคำ
    if each_review.endswith('\n'):
      each_review = each_review[:-1]
    if not each_review == '':
      pos_reviews.append([each_review,'positive'])  # แท็ก positive
  #เก็บ negative ให้เป็น list -----------------------------------------

  neg_reviews = []
  for each_review in neg_reviews_file:
    each_review = ' '.join(word_tokenize(each_review))
    if each_review.endswith('\n'):
      each_review = each_review[:-1]
    if not each_review == '':
      neg_reviews.append([each_review,'negative']) # แท็ก negative

  all_reviews = []
  for (review, sentiment) in pos_reviews + neg_reviews:
    reviews_filtered = [w.lower() for w in word_tokenize(review)]
  all_reviews.append((reviews_filtered, sentiment))
  global features
  features = get_word_features(get_words_in_reviews(all_reviews))
  training_set = nltk.classify.apply_features(extract_features, all_reviews)
  classifier = nltk.NaiveBayesClassifier.train(training_set) # ทำการ train
  return classifier


app = Flask(__name__) #create the Flask app

@app.route('/sentiment', methods=['POST']) #GET requests will be blocked
def json_example():
    req_data = request.get_json()

    message = req_data['message']
    classifier = train()
    tokenize = word_tokenize(message)
    label = classifier.classify(extract_features(word_tokenize(message)))

    return jsonify({'sentiment_label': label,
                    'word_tokenize': tokenize})

if __name__ == '__main__':
    app.run(debug=True, port=5000) #run app in debug mode on port 5000