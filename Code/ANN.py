import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

#le = preprocessing.LabelEncoder()

#load the dataset into python
dataset = pd.read_csv('CMP3751M_CMP9772M_ML_Assignment 2-dataset-nuclear_plants_final.csv') 


#split the dataset into input and output
x = dataset.iloc[:, 1:13] #Features within the dataset assigned to x
y = dataset.select_dtypes(include=[object]) #The categorical variables in the dataset assigned to y


#Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10) #splits train data to 90% and test data to 10%

#feature scaling to normalise the traing data
scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) #normalise input train data
x_test = scaler.transform(x_test) #normalise input test data

#training and making predictions
mlp = MLPClassifier(hidden_layer_sizes=(500), activation = 'logistic', max_iter=2000)
mlp.fit(x_train, y_train.values.ravel())

predictions = mlp.predict(x_test)

#Evaluating the Algorithm
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

from sklearn.model_selection import cross_val_score

#Model Selection
neuron_sample = [50, 500, 1000] #creates a list so that I can test different hidden layers
for i in range(0, len(neuron_sample)):
    mlp = MLPClassifier(hidden_layer_sizes=(neuron_sample[i]), activation = 'logistic', max_iter=1000) #set the classifier to specified layers
    
    scores = cross_val_score(mlp, x, y, cv=10) #performs CV to 10 folds
    print(scores) #outputs an array of CV results
    print("Accuracy: %0.2f " % (scores.mean())) #outputs the accuracy of the Cross Validation
