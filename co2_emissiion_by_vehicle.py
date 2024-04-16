import pandas as pd

df=pd.read_csv("C:\\Users\\jrsar\\OneDrive\\Desktop\\CO2-Emission-Prediction-main\\co2 Emissions.csv")


# print(df.head(1))
# X=df.drop(columns="")
print(df.shape)


df.drop(['Make','Model','Vehicle Class','Fuel Consumption City (L/100 km)','Fuel Consumption Hwy (L/100 km)','Transmission','Fuel Consumption Comb (mpg)'],inplace=True,axis=1)
# print(df.head())


df= df.drop(columns="Fuel Type")
print(df.head())
X=df.drop(columns="CO2 Emissions(g/km)")
print(X.head())
y=df["CO2 Emissions(g/km)"]
print(y.head())


from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(X_train, y_train)
predictions = rf_classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)



# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
predictions = linear_reg.predict(X_test)
predictions_rounded = [round(pred) for pred in predictions]
predictions_rounded = [int(pred) for pred in predictions_rounded]
accuracy_lr = accuracy_score(y_test, predictions_rounded)
print("Accuracy:", accuracy_lr)


# Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
predictions_nb = nb_classifier.predict(X_test)
accuracy_nb = accuracy_score(y_test, predictions_nb)
print("Naive Bayes Accuracy:", accuracy_nb)


from sklearn.metrics import r2_score

r2_rf = r2_score(y_test, predictions)
r2_nb = r2_score(y_test, predictions_nb)
r2_lr = r2_score(y_test, predictions_rounded)

from tabulate import tabulate

scores_data = [
    ["Random Forest", accuracy, r2_rf],
    ["Naive Bayes", accuracy_nb, r2_nb],
    ["Linear Regression", accuracy_lr, r2_lr]
]

print(tabulate(scores_data, headers=["Model", "Accuracy", "R2 Score"], tablefmt="pretty"))

import pickle
pickle.dump(rf_classifier,open("model.pkl","wb"))
