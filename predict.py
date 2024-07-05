import pandas as pd
import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

data = pd.read_csv("heartDisease_2020_sampling.csv")


print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)


print(data.head())


input("\n Press Enter to continue.\n")


data = util.labelEncoder(data, ["HeartDisease", "GenHealth"])


print("\nHere is a preview of the dataset after label encoding. \n")

print(data.head())
input("\nPress Enter to continue.\n")


data = util.oneHotEncoder(data, ["Race"])

print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)
print(data.head())


input("\nPress Enter to continue.\n")




from sklearn.model_selection import train_test_split
X = data.drop("HeartDisease", axis = 1)
y = data["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y)


print(X_train.head())


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 3, class = "balanced")
clf = clf.fit(X_train, y_train)







test_predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, test_predictions, label = [1, 0])





from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_predictions)










