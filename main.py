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
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#Download NLTK resources
nltk.download('stopwords')

#Initialize stop words and stemmer
stopWords = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

#Load phishing email dataset from Kaggle
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

#Load phishing email dataset from Kaggle
def removeHtml(text):
    #Replace anything inside <> with a space
    return re.sub(r'<.*?>', ' ', text)

#Clean and preprocess email text
def cleanText(text):
    #Makes sure text is string
    text = str(text)

    #Remove HTML tags
    text = removeHtml(text)

    #Lowercase
    text = text.lower()

    #Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    #Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    #Tokenize by splitting
    words = text.split()

    #Remove stop words and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stopWords]

    return " ".join(words)

#Prepare dataset for machine learning
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

#Split dataset into train, validation, and test sets
def splitData(x, y):
    #70% train, 15% validation, 15% test
    xTrain, xTemp, yTrain, yTemp = train_test_split(
        x, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    #Split temporary set into: 15% validation, 15% testing
    xVal, xTest, yVal, yTest = train_test_split(
        xTemp, yTemp,
        test_size=0.50,
        random_state=42,
        stratify=yTemp
    )

    '''print("\nData split sizes:")
    print("Training set:", len(xTrain))
    print("Validation set:", len(xVal))
    print("Testing set:", len(xTest))'''

    return xTrain, xVal, xTest, yTrain, yVal, yTest

#Convert text into numerical tf-idf features
def extractFeatures(xTrain, xVal, xTest):
    #Create tf-idf vectorizer
    # max_features limits vocabulary size
    # ngram_range=(1,2) uses unigrams and bigrams
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    #Fit vectorizer on training data
    xTrainTfidf = tfidf.fit_transform(xTrain)
    #Transform validation and testing data
    xValTfidf = tfidf.transform(xVal)
    xTestTfidf = tfidf.transform(xTest)

    return xTrainTfidf, xValTfidf, xTestTfidf, tfidf

#Display evaluation metrics for a model
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

#Plot confusion matrix heatmap
def plotConfusionMatrix(yTrue, yPred, modelName):
    cm = confusion_matrix(yTrue, yPred)

    plt.figure(figsize=(6,5))

    #Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Legitimate', 'Phishing', 'Garbage'],
        yticklabels=['Legitimate', 'Phishing', 'Garbage']
    )

    plt.title(f'{modelName} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.show()

#KNN model
def knn(xTrainTfidf, yTrain, xTestTfidf, yTest, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(xTrainTfidf, yTrain)
    yPred = model.predict(xTestTfidf)
    evaluateModel(f"KNN (k={k})", yTest, yPred)
    return model, yPred

#Naive Bayes model
def naiveBayes(xTrainTfidf, yTrain, xTestTfidf, yTest):
    model = MultinomialNB()
    model.fit(xTrainTfidf, yTrain)
    yPred = model.predict(xTestTfidf)
    evaluateModel("Naive Bayes", yTest, yPred)
    return model, yPred

#Logistic Regression model
def logisticRegression(xTrainTfidf, yTrain, xTestTfidf, yTest):
    model = LogisticRegression(max_iter=1000)
    model.fit(xTrainTfidf, yTrain)
    yPred = model.predict(xTestTfidf)
    evaluateModel("Logistic Regression", yTest, yPred)
    return model, yPred

#Optimize Logistic Regression model
def optimizeLogisticRegression(xTrainTfidf, yTrain, xValTfidf, yVal):
    #Parameter combinations to test
    paramGrid = {
        #Regularization strength
        "C": [0.01, 0.1, 1, 10, 100],
        #Optimization algorithm
        "solver": ["lbfgs"],
        #Handle class imbalance
        "class_weight": [None, "balanced"]
    }
    #Perform grid search with 5-fold cross validation
    grid = GridSearchCV(
        #Base model
        LogisticRegression(max_iter=2000),
        #Parameter combinations
        paramGrid,
        #5-fold cross validation
        cv=5,
        #Optimize for highest accuracy
        scoring="accuracy",
        #Use all cpu cores
        n_jobs=-1
    )
    #Train and evaluate all parameter combinations
    grid.fit(xTrainTfidf, yTrain)

    '''print("\nBest Parameters:")
    print(grid.best_params_)

    print("\nBest Cross Validation Accuracy:")
    print(round(grid.best_score_, 3))'''
    #Retrieve best-performing model
    bestModel = grid.best_estimator_

    '''yPred = bestModel.predict(xValTfidf)
    evaluateModel("Optimized Logistic Regression Validation", yVal, yPred)'''

    return bestModel

def main():
    df = loadDataset()
    x, y = preprocessData(df)
    k = 5
    #Split dataset
    xTrain, xVal, xTest, yTrain, yVal, yTest = splitData(x, y)
    #Convert text into tf-idf features
    xTrainTfidf, xValTfidf, xTestTfidf, tfidf = extractFeatures(xTrain, xVal, xTest)

    #Train and evaluate KNN
    knnModel, knnPred = knn(xTrainTfidf, yTrain, xTestTfidf, yTest, k)
    #Train and evaluate Naive Bayes
    nbModel, nbPred = naiveBayes(xTrainTfidf, yTrain, xTestTfidf, yTest)
    #Train and evaluate Logistic Regression
    logisticRegression(xTrainTfidf, yTrain, xTestTfidf, yTest)

    #Optimize Logistic Regression
    bestLogModel = optimizeLogisticRegression(xTrainTfidf, yTrain, xValTfidf, yVal)

    #Predict testing set using optimized model
    yTestPred = bestLogModel.predict(xTestTfidf)
    #Evaluate optimized Logistic Regression
    evaluateModel("Final Logistic Regression Test", yTest, yTestPred)

    #Display confusion matrix heatmaps
    plotConfusionMatrix(yTest, knnPred, "KNN")
    plotConfusionMatrix(yTest, nbPred, "Naive Bayes")
    plotConfusionMatrix(yTest, yTestPred, "Logistic Regression")

if __name__ == "__main__":
    main()