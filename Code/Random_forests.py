import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#load the dataset into python
dataset = pd.read_csv('CMP3751M_CMP9772M_ML_Assignment 2-dataset-nuclear_plants_final.csv') 

#split the dataset into input and output
x = dataset.iloc[:, 1:13] #Features within the dataset assigned to x
y = dataset.select_dtypes(include=[object]) #The categorical variables in the dataset assigned to y

#Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10) #Splits data into 90% train and 10% test

#Create a gaussian classifier
clf = RandomForestClassifier(n_estimators=1000, min_samples_leaf=50)

#Train the model using the training sets 
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

#Check the accuracy of the classifier
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

treeSamples = [10, 50, 100, 1000, 5000] #list to hold the amount of trees to test to the classifier


for i in range(0, len(treeSamples)): #for loop that will run through the classifier and output accuracy based on treeSample value
    clf = RandomForestClassifier(n_estimators=treeSamples[i], min_samples_leaf=5) #Creates the classifier
    clf.fit(x_train, y_train) #trains the classifier
    y_pred = clf.predict(x_test) #predicts the value based on the classifier
    
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred)) #outputs the accuracy to the user
    
from sklearn.model_selection import cross_val_score    
    
#Model selection
treeSample = [20, 500, 10000]
for i in range(0, len(treeSample)): #creates a list so that I can test different hidden layers
    clf = RandomForestClassifier(n_estimators=treeSample[i]) #sets classifier to specific number of Trees
    
    scores = cross_val_score(clf, x, y, cv=10) #Performs CV to 10 folds
    print(scores) #outputs an array of CV results
    print("Accuracy: %0.2f " % (scores.mean())) #Outputs the accuracy of the CV