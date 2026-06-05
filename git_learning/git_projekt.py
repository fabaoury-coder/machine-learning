from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

df = pd.read_csv(
 "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
 )
# Datenvisualisierung

print(df.head())
print(df.info())
print(df.describe())

# 3. Erstelle Histogramme für alle numerischen Spalten.
#df.hist()



# Features als Eingabedaten
X = df[["employment_duration",
        "installment_rate",
        "personal_status_sex",
        "other_debtors"
        ]]
X=pd.get_dummies(X)


# Features als Eingabedaten
y = df["credit_risk"]
print("Target Vektor :",y.shape, type(y))

print(f"Features  :",X.shape, type(X))
X_train, X_test, y_train, y_test = train_test_split(
    X,                    # *arrays
    y,                    # *arrays
    test_size=0.3,        # 30 % Testdaten
    random_state=42,      # reproduzierbarer Zufall
    #shuffle=True,         # Daten mischen
)
print("X_train:", X_train)
print("X_test :", X_test)
print("y_train:", y_train)
print("y_test :", y_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metriken
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mean_absolute_percentage_error_= mean_absolute_percentage_error(y_test, y_pred)
root_mean_squared_error_= root_mean_squared_error(y_test, y_pred)
print("Mean Absolute Error:", mean_absolute_error)
print("Mean Squared Error:", mse)
print("R2:", r2)
print("Mean Absolute Percentage Error:", mean_absolute_percentage_error)
print("Root Mean Squared Error:", root_mean_squared_error)


# git init
# git add .
# git commit -m "first commit"
# git remote add origin https://github.com/TON_USER/TON_REPO.git
# git branch -M main
# git push -u origin main