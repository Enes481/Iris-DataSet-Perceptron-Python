__author__ = "İsmail Enes Tığlı"

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

path = kagglehub.dataset_download("uciml/iris")

print("Path to dataset files:", path)
data = pd.read_csv("C:/Users/W11/.cache/kagglehub/datasets/uciml/iris/versions/2/Iris.csv")
data.head(5)

print("\n")

data = data.drop("Id", axis=1)
data.info()

print("\n")

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
rfc = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix for RandomForestClassifier\n')
print(cm)

scores = cross_val_score(rfc, X_train, y_train, cv=5)
print("Cross-Validation Scores: \n\n", scores)

y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)

# make visualization
colours = {'Iris-setosa': 'orange', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}

data_frame = pd.DataFrame(data=x_train, columns=["sepal length", "sepal width", "petal length", "petal width"])

for type, colour in colours.items():
    grup = data_frame[y_train == type]
    plt.scatter(grup['petal length'], grup['petal width'], c=colour, label=type)

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal width and height')

# Show the corresponding types of colors
plt.legend(loc='best')
plt.show()
