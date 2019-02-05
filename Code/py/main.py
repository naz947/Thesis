# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:41:54 2018

@author: naz2hi
"""
from functions import *

train_data=pd.read_excel("U:/JIRA/JIRAExport/train_data.xls")
train_data=train_data.dropna()
train_data=replace_strings(train_data,'Component/s')
train_data=preprocess(train_data,['Description','Summary'])
#test_data=replace_strings(test_data,'Component/s')
#comp1=list(data['Component/s'].unique())
comp2=list(train_data['Component/s'].unique())

Comp=[]
for each in comp2:
    if(each.find(',')!=-1):
        Comp.append(each.split(",")[-1])
    else:
        Comp.append(each) 
Comp=comp2


train_data['Component/s']=train_data['Component/s'].replace(Comp,range(len(Comp)))

train_data=train_data.sample(frac=1.0, replace=True)

train=train_data.sample(frac=0.90,random_state=120)
test=train_data.drop(train.index)

"""
Model 1
"""

#"""
#Summary
#"""
#
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(train['Summary'])
#
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#clf = MultinomialNB().fit(X_train_tfidf, train['Component/s'])
#print(clf)
#
#text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
#
#text_clf = text_clf.fit(train['Summary'], train['Component/s'])
#
#predicted = text_clf.predict(test['Summary'])
#np.mean(predicted == test['Component/s'])
#
#err.err(test['Component/s'],predicted)
#
#"""
#Description
#"""
#
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(train['Description'])
#
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#clf = MultinomialNB().fit(X_train_tfidf, train['Component/s'])
#print(clf)
#
#text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
#
#text_clf = text_clf.fit(train['Description'], train['Component/s'])
#
#predicted = text_clf.predict(test['Description'])
#np.mean(predicted == test['Component/s'])
#
#err.err(test['Component/s'],predicted)


"""
Model 2
"""


#col=['Issue Key','Summary','Component/s']
#df=train_data[col]
#
#df['component_id'] = df['Component/s'].factorize()[0]
#category_id_df = df[['Component/s', 'component_id']].drop_duplicates().sort_values('component_id')
#category_to_id = dict(category_id_df.values)
#id_to_category = dict(category_id_df[['component_id', 'Component/s']].values)
#
#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
#
#features = tfidf.fit_transform(df.Summary).toarray()
#labels = df.component_id
#
#N = 2
#for Product, category_id in sorted(category_to_id.items()):
#  features_chi2 = chi2(features, labels == category_id)
#  indices = np.argsort(features_chi2[0])
#  feature_names = np.array(tfidf.get_feature_names())[indices]
#  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#  print("# '{}':".format(Product))
#  print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
#  print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))
#  
#models = [
#    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
#    LinearSVC(),
#    MultinomialNB(),
#    LogisticRegression(random_state=0),
#]
#CV = 5
#cv_df = pd.DataFrame(index=range(CV * len(models)))
#entries = []
#for model in models:
#    model_name = model.__class__.__name__
#    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
#    for fold_idx, accuracy in enumerate(accuracies):
#        entries.append((model_name, fold_idx, accuracy))
#cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
#print(cv_df.groupby('model_name').accuracy.mean())
#
#sns.boxplot(x='model_name', y='accuracy', data=cv_df)
#sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
#              size=9, jitter=True, edgecolor="gray", linewidth=2)
#plt.show()
#
#model = LinearSVC()
#
#X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.1, random_state=0)
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#
#print(np.mean(y_test==y_pred))
#
#err.err(y_test,y_pred)


"""
Linear Regression
"""
#check again
#train=train_data.sample(frac=0.90,random_state=120)
#test=train_data.drop(train.index)
#train.head()
#
#train_clean_sentences = []
#for line_number in range(len(train)):
#    line = train.iloc[line_number]['Description']
#    cleaned = clean(line)
#    cleaned = ' '.join(cleaned)
#    train_clean_sentences.append(cleaned)
#
#
#vectorizer = TfidfVectorizer(stop_words='english')
#X = vectorizer.fit_transform(train_clean_sentences)
#
#Accuracy=[]
#for i in range(1,50):
#    modelknn = KNeighborsClassifier(n_neighbors=i)
#    modelknn.fit(X,train['Component/s'])
#    predicted_labels_knn = modelknn.predict(test)
#    Accuracy.append(np.mean(test['Component/s']==predicted_labels_knn))
#    print("For neigbours :",i,"Accuracy is :",np.mean(test['Component/s']==predicted_labels_knn))
