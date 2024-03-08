# 1A	Design a simple machine learning model to train the training instances and test the same.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Set a random seed for reproducibility
np.random.seed(2)

# Generate random data
x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) / x

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.title("Scatter Plot of Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)


plt.scatter(train_x,train_y)
plt.show()

plt.scatter(test_x,test_y)
plt.show()


# Create and visualize a polynomial regression model for training data
degree = 4  # Adjust the polynomial degree as needed
train_model = np.poly1d(np.polyfit(train_x, train_y, degree))
myline = np.linspace(0, 6, 200)

plt.figure(figsize=(8, 6))
plt.scatter(train_x, train_y)
plt.plot(myline, train_model(myline))
plt.title("Polynomial Regression Model (Training Data)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Calculate and print the R-squared score for the training data
r2_train = r2_score(train_y, train_model(train_x))
print("R-squared score for training data:", r2_train)

# Create and visualize a polynomial regression model for testing data
test_model = np.poly1d(np.polyfit(test_x, test_y, degree))

plt.figure(figsize=(8, 6))
plt.scatter(test_x, test_y)
plt.plot(myline, test_model(myline))
plt.title("Polynomial Regression Model (Testing Data)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Calculate and print the R-squared score for the testing data
r2_test = r2_score(test_y, test_model(test_x))
print("R-squared score for testing data:", r2_test)

# Make predictions using the model
prediction = test_model(5)
print("Prediction for x = 5:", prediction)


#1B	Implement and demonstrate the FIND-S algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a .CSV file

import csv
num_attributes = 6
a = []
print("\n The Given Training Data Set \n")
with open('abalone_csv.csv', 'r') as csvfile:
 reader = csv.reader(csvfile)
 for row in reader:
  a.append (row)
 print(row)
print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes
print(hypothesis)
for j in range(0,num_attributes):
 hypothesis[j] = a[0][j];
print("\n Find S: Finding a Maximally Specific Hypothesis\n")
for i in range(0,len(a)):
 if a[i][num_attributes]=='yes':
  for j in range(0,num_attributes):
   if a[i][j]!=hypothesis[j]:
    hypothesis[j]='?'
 else :
  hypothesis[j]= a[i][j]
 #print(" For Training instance No:{0} the hypothesis is ".format(i),hypothesis)

print("\n The Maximally Specific Hypothesis for a given Training Examples :\n")
print(hypothesis)


### 2A	Perform Data Loading, Feature selection (Principal Component analysis) and Feature Scoring and Ranking.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Data Loading
data = pd.read_csv("data.csv")  # Replace "data.csv" with your dataset file
X = data.drop('target_column', axis=1)  # Features
y = data['target_column']  # Target variable (if applicable)

# Feature Selection using PCA
pca = PCA()
pca.fit(X)
explained_var_ratio = pca.explained_variance_ratio_

# Plot the explained variance to decide on the number of components to keep
plt.plot(range(1, len(explained_var_ratio) + 1), explained_var_ratio.cumsum())
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.show()

# Choose the number of components that explain most of the variance
num_components = 5  # Adjust as needed

# Transform the data with the selected number of components
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X)

# Feature Scoring and Ranking
loadings = pca.components_
feature_scores = abs(loadings).mean(axis=0)  # Use mean absolute loading values

# Create a DataFrame to display feature scores
feature_scores_df = pd.DataFrame({'Feature': X.columns, 'Score': feature_scores})

# Sort features by score in descending order
feature_scores_df = feature_scores_df.sort_values(by='Score', ascending=False)

# Display the ranked features
print(feature_scores_df)

####   Write a program to implement the naïve Bayesian classifier for a sample training data set stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets.


# import necessary libarities
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# load data from CSV
data = pd.read_csv('tennisdata.csv')
print("THe first 5 values of data is :\n",data.head())

# obtain Train data and Train output
X = data.iloc[:,:-1]
print("\nThe First 5 values of train data is\n",X.head())

y = data.iloc[:,-1]
print("\nThe first 5 values of Train output is\n",y.head())

# Convert then in numbers 
le_outlook = LabelEncoder()
X.Outlook = le_outlook.fit_transform(X.Outlook)

le_Temperature = LabelEncoder()
X.Temperature = le_Temperature.fit_transform(X.Temperature)

le_Humidity = LabelEncoder()
X.Humidity = le_Humidity.fit_transform(X.Humidity)

le_Windy = LabelEncoder()
X.Windy = le_Windy.fit_transform(X.Windy)

print("\nNow the Train data is :\n",X.head())

le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)
print("\nNow the Train output is\n",y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)

classifier = GaussianNB()
classifier.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
print("Accuracy is:",accuracy_score(classifier.predict(X_test),y_test))













##3B	Write a program to implement Decision Tree and Random forest with Prediction, Test Score and Confusion Matrix.

# Import pandas library
import pandas as pd

# Loading dataset
df = pd.read_csv("diabetes_dataset.csv")
df.head()

# Feature variables
x = df.drop(['Outcome'], axis=1)

# Target variable
y = df.Outcome

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create Decision Tree classifier object
model = DecisionTreeClassifier()

# Train Decision Tree Classifier
model.fit(x_train, y_train)

# Predict the response for the test dataset
y_pred = model.predict(x_test)

# Evaluation using Accuracy score
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)

# Evaluation using Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy calculation from Confusion matrix
accuracy = (cm[0, 0] + cm[1, 1]) / sum(sum(cm))
print("Accuracy from Confusion Matrix:", accuracy * 100)

# Evaluation using Classification report
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Checking prediction value
prediction = model.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
print("Prediction for input:", prediction)

# Import modules for Visualizing Decision trees
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualizing Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=x.columns, class_names=['0', '1'], filled=True, rounded=True)
plt.show()

# Create Decision Tree classifier object with entropy and max_depth
model = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifier
model.fit(x_train, y_train)

# Predict the response for the test dataset
y_pred = model.predict(x_test)

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)

# Better Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=x.columns, class_names=['0', '1'], filled=True, rounded=True)
plt.show()




                 
                 
                 



##4A	For a given set of training data examples stored in a .CSV file implement Least Square Regression algorithm.

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Loading dataset
data = pd.read_csv('PropallantAge.csv')
data.head()
data.info()
# Plotting the data
plt.scatter(data['age_of_propellant'],data['shear_strength'])
# Computing X and Y
X = data['age_of_propellant'].values
Y = data['shear_strength'].values
# Mean of variables X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)
 
# Total number of data values
n = len(X)

# Calculating 'm' and 'c'
num = 0
denom = 0
for i in range(n):
  num += (X[i] - mean_x) * (Y[i] - mean_y)
  denom += (X[i] - mean_x) ** 2
m = num / denom
c = mean_y - (m * mean_x)
 
# Printing coefficients
print("Coefficients")
print(m, c)


# Plotting Values and Regression Line
 
maxx_x = np.max(X) + 10
minn_x = np.min(X) - 10
 
# line values for x and y
x = np.linspace(minn_x, maxx_x, 1000)
y = c + m * x
 
# Ploting Regression Line
plt.plot(x, y, color='#58b970', label='Regression Line')
 
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')
 
plt.xlabel('Age of Propellant (in years)')
plt.ylabel('Shear Strength')
plt.legend()
plt.show()


##4B	For a given set of training data examples stored in a .CSV file implement Logistic Regression algorithm.

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import exp
plt.rcParams["figure.figsize"] = (10, 6)

# Download the dataset
# Source of dataset - https://www.kaggle.com/rakeshrau/social-network-ads
# !wget "https://drive.google.com/uc?id=15WAD9_4CpUK6EWmgWVXU8YMnyYLKQvW8&export=download" -O data.csv -q

# Load the data
data = pd.read_csv("data1.csv")
data.head()
# Visualizing the dataset
plt.scatter(data['Age'], data['Purchased'])
plt.show()

# Divide the data to training set and test set
X_train, X_test, y_train, y_test = train_test_split(data['Age'], data['Purchased'], test_size=0.20)
# Creating the logistic regression model

# Helper function to normalize data
def normalize(X):
    return X - X.mean()

# Method to make predictions
def predict(X, b0, b1):
    return np.array([1 / (1 + exp(-1*b0 + -1*b1*x)) for x in X])

# Method to train the model
def logistic_regression(X, Y):

    X = normalize(X)

    # Initializing variables
    b0 = 0
    b1 = 0
    L = 0.001
    epochs = 300

    for epoch in range(epochs):
        y_pred = predict(X, b0, b1)
        D_b0 = -2 * sum((Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b0
        D_b1 = -2 * sum(X * (Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b1
        # Update b0 and b1
        b0 = b0 - L * D_b0
        b1 = b1 - L * D_b1
    
    return b0, b1
# Training the model
b0, b1 = logistic_regression(X_train, y_train)

# Making predictions
X_test_norm = normalize(X_test)
y_pred = predict(X_test_norm, b0, b1)
y_pred = [1 if p >= 0.5 else 0 for p in y_pred]

plt.clf()
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred, c="red")
plt.show()

# The accuracy
accuracy = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test.iloc[i]:
        accuracy += 1
print(f"Accuracy = {accuracy / len(y_pred)}")
# Making predictions using scikit learn
from sklearn.linear_model import LogisticRegression

# Create an instance and fit the model 
lr_model = LogisticRegression()
lr_model.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

# Making predictions
y_pred_sk = lr_model.predict(X_test.values.reshape(-1, 1))

plt.clf()
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred_sk, c="red")
plt.show()

# Accuracy
print(f"Accuracy = {lr_model.score(X_test.values.reshape(-1, 1), y_test.values.reshape(-1, 1))}")



### 5A	Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use an appropriate data set for building the decision tree and apply this knowledge to classify a new sample.


import numpy as np
import math
import csv

def read_data(filename):
    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        headers = next(datareader)
        metadata = []
        traindata = []
        for name in headers:
            metadata.append(name)
        for row in datareader:
            traindata.append(row)

    return (metadata, traindata)

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""
        
    def __str__(self):
        return self.attribute

def subtables(data, col, delete):
    dict = {}
    items, counts = np.unique(data[:, col], return_counts=True)
    
    for x in range(items.shape[0]):
        dict[items[x]] = data[data[:, col] == items[x]]
        if delete:
            dict[items[x]] = np.delete(dict[items[x]], col, 1)
        
    return items, dict

def entropy(S):
    items, counts = np.unique(S, return_counts=True)
    total_size = S.size
    entropies = np.zeros(counts.shape)
    
    for x in range(items.shape[0]):
        ratio = counts[x] / total_size
        entropies[x] = ratio * math.log(ratio, 2)

    return -np.sum(entropies)

def gain_ratio(data, col):
    items, dict = subtables(data, col, delete=False) 
                
    total_size = data.shape[0]
    entropies = np.zeros((items.shape[0], 1))
    intrinsic = np.zeros((items.shape[0], 1))
    
    for x in range(items.shape[0]):
        ratio = dict[items[x]].shape[0] / total_size
        entropies[x] = ratio * entropy(dict[items[x]][:, -1])
        intrinsic[x] = ratio * math.log(ratio, 2)
        
    total_entropy = entropy(data[:, -1])
    iv = -np.sum(intrinsic)
    
    for x in range(entropies.shape[0]):
        total_entropy -= entropies[x]
        
    return total_entropy / iv

def create_node(data, metadata):
    if data.size == 0:
        node = Node("")
        node.answer = "No data"
        return node

    if (np.unique(data[:, -1])).shape[0] == 1:
        node = Node("")
        node.answer = np.unique(data[:, -1])[0]
        return node
        
    gains = np.zeros((data.shape[1] - 1, 1))
    
    for col in range(data.shape[1] - 1):
        gains[col] = gain_ratio(data, col)
        
    split = np.argmax(gains)
    
    node = Node(metadata[split])    
    metadata = np.delete(metadata, split, 0)    
    
    items, dict = subtables(data, split, delete=True)
    
    for x in range(items.shape[0]):
        child = create_node(dict[items[x]], metadata)
        node.children.append((items[x], child))
    
    return node

def empty(size):
    s = ""
    for x in range(size):
        s += "   "
    return s

def print_tree(node, level):
    if node.answer != "":
        print(empty(level), "Answer:", node.answer)
        return
    print(empty(level), "Attribute:", node.attribute)
    for value, n in node.children:
        print(empty(level + 1), f"Value: {value}")
        print_tree(n, level + 2)

metadata, traindata = read_data("tennisdata.csv")
data = np.array(traindata)
node = create_node(data, metadata)
print_tree(node, 0)




### 5B	Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set.

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

dataset=load_iris()
#print(dataset)
X_train,X_test,y_train,y_test=train_test_split(dataset["data"],dataset["target"],random_state=0)

kn=KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train,y_train)

for i in range(len(X_test)):
    x=X_test[i]
    x_new=np.array([x])
    prediction=kn.predict(x_new)
   # print("TARGET=",y_test[i],dataset["target_names"][y_test[i]],"PREDICTED=",prediction,dataset["target_names"][prediction])
print(kn.score(X_test,y_test))


##  6A	Implement the different Distance methods (Euclidean) with Prediction, Test Score and Confusion Matrix.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate sample data (you can replace this with your actual data)
X, y = np.random.rand(100, 2), np.random.randint(0, 2, 100)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function for different distance methods
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def cosine_similarity(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

# Define a function for making predictions using K-Nearest Neighbors
def predict(X_train, y_train, x_test, k, distance_fn):
    distances = []

    for i in range(len(X_train)):
        dist = distance_fn(X_train[i], x_test)
        distances.append((dist, y_train[i]))

    distances = sorted(distances)[:k]

    labels = [d[1] for d in distances]
    return max(set(labels), key=labels.count)

# Define a function for calculating test score and confusion matrix
def evaluate(X_train, y_train, X_test, y_test, k, distance_fn):
    y_pred = []

    for x in X_test:
        y_pred.append(predict(X_train, y_train, x, k, distance_fn))

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm

# Example usage
k = 5
distance_fn = euclidean_distance

accuracy, cm = evaluate(X_train, y_train, X_test, y_test, k, distance_fn)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:')
print(cm)


### 6B	Implement the classification model using clustering for the following techniques with K means clustering with Prediction, Test Score and Confusion Matrix.


# read in the iris data
from sklearn.datasets import load_iris
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target

# Import the class
from sklearn.linear_model import LogisticRegression
# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response values for the observations in X
logreg.predict(X)

# Store the predicted response values
y_pred = logreg.predict(X)

# Check how many Predictions were generated
len(y_pred)

# Computer classification accuracy for the logistic regression model
from sklearn import metrics

print(metrics.accuracy_score(y, y_pred))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))

# print the shapes of X and y
# X is our features matrix with 150 x 4 dimensions
print(X.shape)
# y is our response vector with 150 x 1 dimension
print(y.shape)

# STEP 1: split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

# print the shapes of the new X objects
print(X_train.shape)
print(X_test.shape)

# print the shapes of the new y objects
print(y_train.shape)
print(y_test.shape)

# STEP 2: train the model on the training set
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# STEP 3: make predictions on the testing set
y_pred = logreg.predict(X_test)

# Computer actual response values (y_test) with predicted response values (y_pred)
print(metrics.accuracy_score(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

# try k-1 through k-25 and record testing accuracy
k_range = range(1, 26)

# we can create python dictionary using [] or dict()
scores = []

# we use a loop through the range 1 to 26
# we append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)

# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# plot the relationship between k and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Testing Accuracy')

# instantiate the model with the best-known parameters
knn = KNeighborsClassifier(n_neighbors=11)

# train the model with X and y (not X_train and y_train)
knn.fit(X, y)

# make a prediction for an out-of-sample observation
knn.predict([[3, 5, 4, 3]])



















#### 7A	Implement the classification model using clustering for the following techniques with hierarchical clustering with Prediction, Test Score and Confusion Matrix.



import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
dataset = pd.read_csv('abalone_csv.csv')
X = dataset.iloc[:,[3,4]].values



dendrogram = sch.dendrogram(sch.linkage(X,method = "ward"))
plt.title('Dendrogram')
plt.xlabel('Gender')
plt.ylabel('Euclidean distances')
plt.show()


hc = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean',linkage = 'ward')
y_hc=hc.fit_predict(X)
print("Prediction Values :",y_hc)
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s=100, c='red',label = 'Cluster 1')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s=100, c='blue',label = 'Cluster 2')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s=100, c='green',label = 'Cluster 3')

dendrogram  = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.title('Dendrogram')
plt.xlabel('Gender')
plt.ylabel('Euclidean distances')
plt.show()


hc = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(X)
print("Prediction Values : ",y_hc)
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s=100, c='red',label = 'Cluster 1')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s=100, c='blue',label = 'Cluster 2')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s=100, c='green',label = 'Cluster 3')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s=100, c='cyan',label = 'Cluster 4')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s=100, c='magenta',label = 'Cluster 5')
plt.title('Clusters of Customers (Hierarchical Clustering Model)')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.show()


### 7B	Implement the Rule based method and test the same.

import re

def rule_based_classifier(text):
    # Define rules and corresponding categories
    rules = [
        (r'\b(?:buy|purchase)\b', 'Buy'),
        (r'\b(?:sell|sale)\b', 'Sell'),
        (r'\b(?:rent|lease)\b', 'Rent'),
        (r'\b(?:job|hiring|employment)\b', 'Job'),
        (r'\b(?:lost|found)\b', 'Lost and Found'),
        (r'\b(?:help)\b', 'Help'),
        (r'\b(?:event)\b', 'Event'),
        (r'\b(?:news)\b', 'News')
    ]
    
    for pattern, category in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return category
    
    return 'Other'

# Example usage
text = "I want to buy a new phone"
category = rule_based_classifier(text)
print(f"Category: {category}")


### 8A	Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using standard Heart Disease Data Set.


import numpy as np
import pandas as pd
import csv
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

heartDataset = pd.read_csv('heart1.csv')
heartDataset = heartDataset.replace('?',np.nan)
heartDataset.head()
model = BayesianModel([('age','trestbps'),('age','fbs'),
('gender','trestbps'),('exang','trestbps'),('trestbps','heartdisease'),('fbs','heartdisease'),('heartdisease','restecg'), ('heartdisease','thalach'),('heartdisease','chol')])
print('Learning CPD using Maximum likelihood estimator')
model.fit(heartDataset, estimator=MaximumLikelihoodEstimator)
print('Inferencing with Bayesian Network')
heart_infer = VariableElimination(model)
print('Probability of Heart Disease given age=29')
q = heart_infer.query(variables=['heartdisease'], evidence={'age':29})
print(q)
print('Probability of Heart Disease given cholestrol=131')
q = heart_infer.query(variables=['heartdisease'], evidence={'chol':131})
print(q) 



### 8B	 Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs.


import numpy as np
import matplotlib.pyplot as plt

def local_regression(x0, X, Y, tau):
    x0 = [1, x0]   
    X = [[1, i] for i in X]
    X = np.asarray(X)
    xw = (X.T) * np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau))
    beta = np.linalg.pinv(xw @ X) @ xw @ Y @ x0  
    return beta    

def draw(tau):
    prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
    plt.plot(X, Y, 'o', color='black')
    plt.plot(domain, prediction, color='red')
    plt.show()

X = np.linspace(-3, 3, num=1000)
domain = X
Y = np.log(np.abs(X ** 2 - 1) + .5)

draw(10)
draw(0.1)
draw(0.01)
draw(0.001)




### 9A	Build an Artificial Neural Network by implementing the Back propagation algorithm and test the same using appropriate data sets.

import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)     # X = (hours sleeping, hours studying)
y = np.array(([92], [86], [89]), dtype=float)           # y = score on test

# scale units
X = X/np.amax(X, axis=0)        # maximum of X array
y = y/100                       # max test score is 100

class Neural_Network(object):
    def __init__(self):
                            # Parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
                             # Weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)        # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)       # (3x1) weight matrix from hidden to output layer

    def forward(self, X):
                             #forward propagation through our network
        self.z = np.dot(X, self.W1)               # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z)            # activation function
        self.z3 = np.dot(self.z2, self.W2)        # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3)                 # final activation function
        return o 

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))     # activation function 

    def sigmoidPrime(self, s):
        return s * (1 - s)          # derivative of sigmoid
    
    def backward(self, X, y, o):
                                    # backward propgate through the network
        self.o_error = y - o        # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to 
        self.z2_error = self.o_delta.dot(self.W2.T)    # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error
        self.W1 += X.T.dot(self.z2_delta)       # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

NN = Neural_Network()
print ("\nInput: \n" + str(X))
print ("\nActual Output: \n" + str(y)) 
print ("\nPredicted Output: \n" + str(NN.forward(X)))
print ("\nLoss: \n" + str(np.mean(np.square(y - NN.forward(X)))))     # mean sum squared loss)
NN.train(X, y)




### 9B	Assuming a set of documents that need to be classified, use the naïve Bayesian Classifier model to perform this task.


import pandas as pd
msg = pd.read_csv('document.csv', names=['message', 'label'])
print("Total Instances of Dataset: ", msg.shape[0])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

X = msg.message
y = msg.labelnum
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
from sklearn.feature_extraction.text import CountVectorizer

count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)

df = pd.DataFrame(Xtrain_dm.toarray(),columns=count_v.get_feature_names_out())
print(df[0:5])

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)
pred = clf.predict(Xtest_dm)

for doc, p in zip(Xtrain, pred):
    p = 'pos' if p == 1 else 'neg'
    print("%s -> %s" % (doc, p))

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
print('Accuracy Metrics: \n')
print('Accuracy: ', accuracy_score(ytest, pred))
print('Recall: ', recall_score(ytest, pred))
print('Precision: ', precision_score(ytest, pred))
print('Confusion Matrix: \n', confusion_matrix(ytest, pred))



##  10A

import numpy as np
import math
import csv

def read_data(filename):
    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        headers = next(datareader)
        metadata = []
        traindata = []
        for name in headers:
            metadata.append(name)
        for row in datareader:
            traindata.append(row)

    return (metadata, traindata)

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""
        
    def __str__(self):
        return self.attribute

def subtables(data, col, delete):
    dict = {}
    items = np.unique(data[:, col])
    count = np.zeros((items.shape[0], 1), dtype=np.int32)    
    
    for x in range(items.shape[0]):
        for y in range(data.shape[0]):
            if data[y, col] == items[x]:
                count[x] += 1
                
    for x in range(items.shape[0]):
        dict[items[x]] = np.empty((int(count[x]), data.shape[1]), dtype="|S32")
        pos = 0
        for y in range(data.shape[0]):
            if data[y, col] == items[x]:
                dict[items[x]][pos] = data[y]
                pos += 1       
        if delete:
            dict[items[x]] = np.delete(dict[items[x]], col, 1)
        
    return items, dict

def entropy(S):
    items = np.unique(S)

    if items.size == 1:
        return 0
    
    counts = np.zeros((items.shape[0], 1))
    sums = 0
    
    for x in range(items.shape[0]):
        counts[x] = sum(S == items[x]) / (S.size * 1.0)

    for count in counts:
        sums += -1 * count * math.log(count, 2)
    return sums

def gain_ratio(data, col):
    items, dict = subtables(data, col, delete=False) 
                
    total_size = data.shape[0]
    entropies = np.zeros((items.shape[0], 1))
    intrinsic = np.zeros((items.shape[0], 1))
    
    for x in range(items.shape[0]):
        ratio = dict[items[x]].shape[0]/(total_size * 1.0)
        entropies[x] = ratio * entropy(dict[items[x]][:, -1])
        intrinsic[x] = ratio * math.log(ratio, 2)
        
    total_entropy = entropy(data[:, -1])
    iv = -1 * sum(intrinsic)
    
    for x in range(entropies.shape[0]):
        total_entropy -= entropies[x]
        
    return total_entropy / iv

def create_node(data, metadata):
    if (np.unique(data[:, -1])).shape[0] == 1:
        node = Node("")
        node.answer = np.unique(data[:, -1])[0]
        return node
        
    gains = np.zeros((data.shape[1] - 1, 1))
    
    for col in range(data.shape[1] - 1):
        gains[col] = gain_ratio(data, col)
        
    split = np.argmax(gains)
    
    node = Node(metadata[split])    
    metadata = np.delete(metadata, split, 0)    
    
    items, dict = subtables(data, split, delete=True)
    
    for x in range(items.shape[0]):
        child = create_node(dict[items[x]], metadata)
        node.children.append((items[x], child))
    
    return node

def empty(size):
    s = ""
    for x in range(size):
        s += "   "
    return s

def print_tree(node, level):
    if node.answer != "":
        print(empty(level), node.answer)
        return
    print(empty(level), node.attribute)
    for value, n in node.children:
        print(empty(level + 1), value)
        print_tree(n, level + 2)

metadata, traindata = read_data("tennisdata1.csv")
data = np.array(traindata)
node = create_node(data, metadata)
print_tree(node, 0)




## 10B

import numpy as np

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)     # X = (hours sleeping, hours studying)
y = np.array(([92], [86], [89]), dtype=float)           # y = score on test

# scale units
X = X/np.amax(X, axis=0)        # maximum of X array
y = y/100                       # max test score is 100

class Neural_Network(object):
    def __init__(self):
                            # Parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
                             # Weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)        # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)       # (3x1) weight matrix from hidden to output layer

    def forward(self, X):
                             #forward propagation through our network
        self.z = np.dot(X, self.W1)               # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z)            # activation function
        self.z3 = np.dot(self.z2, self.W2)        # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3)                 # final activation function
        return o 

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))     # activation function 

    def sigmoidPrime(self, s):
        return s * (1 - s)          # derivative of sigmoid
    
    def backward(self, X, y, o):
                                    # backward propgate through the network
        self.o_error = y - o        # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to 
        self.z2_error = self.o_delta.dot(self.W2.T)    # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error
        self.W1 += X.T.dot(self.z2_delta)       # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

NN = Neural_Network()
print ("\nInput: \n" + str(X))
print ("\nActual Output: \n" + str(y)) 
print ("\nPredicted Output: \n" + str(NN.forward(X)))
print ("\nLoss: \n" + str(np.mean(np.square(y - NN.forward(X)))))     # mean sum squared loss)
NN.train(X, y)


## b. Perform Text pre-processing, Text clustering, classification with Prediction, Test Score and Confusion Matrix


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset

# csv_file = 'Restaurant_Reviews.tsv'
csv_file = 'Restaurant_Reviews.tsv'

dataset = pd.read_csv(csv_file, delimiter='\t', quoting=3)

# Text preprocessing using Natural Language Toolkit (nltk)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):    
    review = re.sub('[^a-zA-Z]', '', dataset['Review'][i]) # Remove non-alphabetic characters
    review = review.lower()  # Convert to lowercase
    review = review.split() # Tokenize the words
    ps = PorterStemmer() # Stemming and removing stopwords
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Join the words to form the processed review
    review = ' '.join(review)
    corpus.append(review)

# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=100)

# Fitting Naive Bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = classifier.predict(X_test)

# Model Accuracy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)



### b. Implement and demonstrate the FIND-S algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training



import csv
hypo = ['%','%','%','%','%','%'];

#with open('trainingdata.csv') as csv_file:
with open('trainingData.csv') as csv_file:
    readcsv = csv.reader(csv_file, delimiter=',')
    print(readcsv)
    
    data = []
    print("\nThe given training examples are:")
    for row in readcsv:
        print(row)
        if row[len(row)-1].upper() == "YES":
            data.append(row)
print("\nThe positive examples are:");
for x in data:
    print(x);
print("\n");
TotalExamples = len(data);
i=0;
j=0;
k=0;
print("The steps of the Find-s algorithm are :\n",hypo);
list = [];
p=0;
d=len(data[p])-1;
for j in range(d):
    list.append(data[i][j]);
hypo=list;
i=1;
for i in range(TotalExamples):
    for k in range(d):
        if hypo[k]!=data[i][k]:
            hypo[k]='?';
            k=k+1;        
        else:
            hypo[k];
    print(hypo);
i=i+1;
print("\nThe maximally specific Find-s hypothesis for the given training examples is :");
list=[];
for i in range(d):
    list.append(hypo[i]);
print(list);









    




