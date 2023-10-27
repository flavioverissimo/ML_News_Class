# packages
import numpy as np
import joblib
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

paths = {
	"model": 'model/ML_ClassificationNews.pkl',
	"tfidf": 'model/ML_ClassificationNews_Tfidf.pkl',
	"categories":'model/ML_ClassificationNews_Categories.pkl'
}


TRAIN_STATUS = True

try:
	model_loaded = joblib.load(paths["model"])
	Tfidf_loaded = joblib.load(paths["tfidf"])
	categories_loaded = joblib.load(paths["categories"])

	news = [input("Put it here (just one): ")]
	news_transformed = Tfidf_loaded.transform(news)
	news_predicted = model_loaded.predict(news_transformed)
	category_index = news_predicted[0]
	print("\n\n>>> The news was defined as: ", categories_loaded[category_index])

except:
	TRAIN_STATUS = True
	print("\n\nWe couldn't predict your news due to the model file doesn't exist.\
			\n\nReasons: \
			\n1 - An accuracy of less than 0.95 \
			\n2 - First time running the model")
	print("\n\n>>> We are training the model!")


if TRAIN_STATUS:
	# loading files and data separating
	news = load_files('data', encoding = 'utf-8', decode_error = 'replace')
	x = news.data
	y = news.target
	categories_names = news.target_names

	# separating between train and test
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=75)


	# creating vetorizer
	stopWords = list(set(stopwords.words('english')))
	vect = TfidfVectorizer(norm=None, stop_words=stopWords, max_features=1000, decode_error="ignore")
	x_train_transformed = vect.fit_transform(x_train)
	x_test_transformed = vect.transform(x_test)


	# defining models
	logisticRegression = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=30, max_iter=1000)
	multinomialNB = MultinomialNB()
	randomForest = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=1)


	# defining voting classifier
	votingClassifier = VotingClassifier(estimators=[("LR", logisticRegression), ("NB", multinomialNB), ("RF", randomForest)], voting="soft")
	votingModel = votingClassifier.fit(x_train_transformed, y_train)
	votingPredictions = votingModel.predict(x_test_transformed)

	# defining accuracy
	modelAccuracity = accuracy_score(y_test, votingPredictions)

	print("\n>>> Training finished. Model accuracy: ", modelAccuracity)

	if modelAccuracity >= 0.95:
		print("\nAs the accuracy is greater than or equal to 0.95, the model was saved!")
		joblib.dump(votingModel, paths["model"])
		joblib.dump(vect, paths["tfidf"])
		joblib.dump(categories_names, paths["categories"])

