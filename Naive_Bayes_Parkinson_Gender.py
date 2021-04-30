import sklearn
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(
    "../input/parkinsons-disease-speech-signal-features/pd_speech_features.csv"
)
data.head(21)

data.info()
sns.heatmap(data.corr())

data.describe()

data["gender"].value_counts() / 3

X_train, X_test, y_train, y_test = train_test_split(data, data["gender"], test_size=0.2)

len(X_train)

len(X_test)

len(data)

model = GaussianNB()

model.fit(X_train, y_train)

model.score(X_test, y_test)

X_test

y_test

values = y_test[:40]
values = values.reset_index(drop=True)
values = values.to_numpy()
values

predictions = model.predict(X_test[:40])
predictions

res_array = values - predictions
res_false = np.count_nonzero(res_array == 1) + np.count_nonzero(
    res_array == -1
)  # Sum of False Values
res_true = np.count_nonzero(res_array == 0)  # True values

print("Number of True Predictions: ", res_true)
print("Number of False Predictions: ", res_false)

plt.figure(figsize=(8, 8))
exp_vals = [res_true, res_false]
exp_labels = ["True", "False"]
plt.pie(exp_vals, labels=exp_labels, autopct="%0.00f%%", shadow="true")
plt.axis("equal")
