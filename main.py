import os
import re
import string
import pandas as pd
import kagglehub
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#download NLTK resources
nltk.download('stopwords')

#initialize stop words and stemmer
stopWords = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def loadDataset():
    path = kagglehub.dataset_download("avnbluefox/avn-phishing-email-classification-dataset")

    fileName = "AVN_Basic.csv"
    filePath = os.path.join(path, fileName)

    if not os.path.exists(filePath):
        raise FileNotFoundError(f"{fileName} not found in dataset folder")

    df = pd.read_csv(filePath)

    '''print("\nFirst 5 rows of dataset:")
    print(df.head())

    print("\nColumn names:")
    print(df.columns)'''

    return df

def removeHtml(text):
    return re.sub(r'<.*?>', ' ', text)

def cleanText(text):
    #makes sure text is string
    text = str(text)

    #remove HTML tags
    text = removeHtml(text)

    #lowercase
    text = text.lower()

    #remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    #remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    #tokenize by splitting
    words = text.split()

    #remove stop words and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stopWords]

    return " ".join(words)

def preprocessData(df):
    #column names
    df = df.dropna(subset=['subject', 'body', 'label']).copy()

    #combine subject and body
    df['text'] = df['subject'].astype(str) + " " + df['body'].astype(str)

    #clean text
    df['text'] = df['text'].apply(cleanText)

    x = df['text']
    y = df['label']

    return x, y

def splitData(x, y):
    #70% train, 15% validation, 15% test
    xTrain, xTemp, yTrain, yTemp = train_test_split(
        x, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    xVal, xTest, yVal, yTest = train_test_split(
        xTemp, yTemp,
        test_size=0.50,
        random_state=42,
        stratify=yTemp
    )

    print("\nData split sizes:")
    print("Training set:", len(xTrain))
    print("Validation set:", len(xVal))
    print("Testing set:", len(xTest))

    return xTrain, xVal, xTest, yTrain, yVal, yTest

def extractFeatures(xTrain, xVal, xTest):
    tfidf = TfidfVectorizer(max_features=5000)

    xTrainTfidf = tfidf.fit_transform(xTrain)
    xValTfidf = tfidf.transform(xVal)
    xTestTfidf = tfidf.transform(xTest)

    return xTrainTfidf, xValTfidf, xTestTfidf, tfidf

def evaluateModel(modelName, yTrue, yPred):
    print("\n" + "=" * 50)
    print(modelName)
    print("-" * 50)
    print("Confusion Matrix:")
    print(confusion_matrix(yTrue, yPred))
    print("\nAccuracy:")
    print(round(accuracy_score(yTrue, yPred), 3))
    print("\nClassification Report:")
    print(classification_report(yTrue, yPred))

def knn(xTrainTfidf, yTrain, xTestTfidf, yTest, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(xTrainTfidf, yTrain)
    yPred = model.predict(xTestTfidf)
    evaluateModel(f"KNN (k={k})", yTest, yPred)
    return model, yPred

def naiveBayes(xTrainTfidf, yTrain, xTestTfidf, yTest):
    model = MultinomialNB()
    model.fit(xTrainTfidf, yTrain)
    yPred = model.predict(xTestTfidf)
    evaluateModel("Naive Bayes", yTest, yPred)
    return model, yPred

def logisticRegression(xTrainTfidf, yTrain, xTestTfidf, yTest):
    model = LogisticRegression(max_iter=1000)
    model.fit(xTrainTfidf, yTrain)
    yPred = model.predict(xTestTfidf)
    evaluateModel("Logistic Regression", yTest, yPred)
    return model, yPred

def optimizeLogisticRegression(xTrainTfidf, yTrain, xValTfidf, yVal):
    bestC = None
    bestAccuracy = 0
    for c in [.01, .1, 1, 10, 100]:
        model = LogisticRegression(max_iter=1000, C = c)
        model.fit(xTrainTfidf, yTrain)

        yPred = model.predict(xValTfidf)
        accuracy = accuracy_score(yVal, yPred)

        print(f"\nC : {c}, Accuracy: {accuracy:.3f}")
        evaluateModel("Logistic Regression", yVal, yPred)
        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestC = c

    print("\nBest C:", bestC)
    return bestC

def main():
    df = loadDataset()
    x, y = preprocessData(df)
    k = 5
    xTrain, xVal, xTest, yTrain, yVal, yTest = splitData(x, y)
    xTrainTfidf, xValTfidf, xTestTfidf, tfidf = extractFeatures(xTrain, xVal, xTest)

    '''knn(xTrainTfidf, yTrain, xTestTfidf, yTest, k)
    naiveBayes(xTrainTfidf, yTrain, xTestTfidf, yTest)
    logisticRegression(xTrainTfidf, yTrain, xTestTfidf, yTest)'''
    optimizeLogisticRegression(xTrainTfidf, yTrain, xValTfidf, yVal)

if __name__ == "__main__":
    main()