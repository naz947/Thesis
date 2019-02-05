# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 10:58:24 2018

@author: naz2hi
"""
import pandas as pd
import os
import nltk
import string
import re
import warnings
import numpy as np
from matplotlib import reload
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from gensim.models import Word2Vec
import gzip
import gensim
import logging
from itertools import islice
import multiprocessing
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter