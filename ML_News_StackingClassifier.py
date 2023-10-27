# packages
import numpy as np
import pickle
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score

TRAIN_STATUS = False

try:
	with open('model/ML_StackingClassifier.pkl', 'rb') as SC2:
		model_loaded = pickle.load(SC2)

	with open('model/ML_StackingClassifier_Tfid.pkl', 'rb') as Tfid2:
		tfid_loaded = pickle.load(Tfid2)

	with open('model/ML_StackingClassifier_Categories.pkl', 'rb') as cat2:
		categories_loaded = pickle.load(cat2)



	text = [input("Insert a news here: ")]
	text_transformed = tfid_loaded.transform(text)
	text_predicted = model_loaded.predict(text_transformed)
	category_index = text_predicted[0]
	category = categories_loaded[category_index]
	print("\n\nThe category for this news was defined as: ", category)

except:
	TRAIN_STATUS = True
	print("\n\nWe couldn't predict your news due to the model file doesn't exist.\
		\n\nReasons: \
		\n1 - An accuracy of less than 0.95 \
		\n2 - First time running the model")
	print("\n\n>>> We are training the model!")


if TRAIN_STATUS:
	# loading data
	news = load_files('data', encoding = 'utf-8', decode_error = 'replace')
	x = news.data
	y = news.target
	categories = news.target_names

	# seperating data for training and test
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=93)

	# vectorizing
	stopWords = list(set(stopwords.words('english')))
	vectorizer = TfidfVectorizer(norm=None, stop_words=stopWords, max_features=1000, decode_error="ignore")
	x_train_transformed = vectorizer.fit_transform(x_train)
	x_test_transformed = vectorizer.transform(x_test)

	# defining the base models
	base_models = [("RF", RandomForestClassifier(n_estimators=1000, random_state=42)), ("NB", MultinomialNB())]

	# training model
	stackingClassifier = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(multi_class="multinomial", random_state=30, max_iter=1000))
	stackingModel = stackingClassifier.fit(x_train_transformed, y_train)

	# getting accuracy
	accuracy = stackingModel.score(x_test_transformed, y_test)

	if accuracy >= 0.96:
		print("saving files")
		with open('model/ML_StackingClassifier.pkl', 'wb') as SC:
			pickle.dump(stackingModel, SC)

		with open('model/ML_StackingClassifier_Tfid.pkl', 'wb') as Tfid:
			pickle.dump(vectorizer, Tfid)

		with open('model/ML_StackingClassifier_Categories.pkl', 'wb') as cat:
			pickle.dump(categories, cat)

