import pandas as  pd

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
embarked_dummies = pd.get_dummies(train_df['Embarked'], prefix='Embarked')
train_df = pd.concat([train_df, embarked_dummies], axis=1)
train_df.drop(['Embarked'], axis=1, inplace=True)

y = train_df['Survived']
x = train_df.drop('Survived', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(x, y, test_size=0.2, random_state=19)

# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
#
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, Y_train)
# pred = model.predict(X_val)
# accuracy = accuracy_score(Y_val, pred)
#
# print("Accuracy Score:", accuracy*100, "%")

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

# print(classification_report(Y_val, pred))
# print(confusion_matrix(Y_val, pred))

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])
test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
embarked_dummies = pd.get_dummies(test_df['Embarked'], prefix='Embarked')
test_df = pd.concat([test_df, embarked_dummies], axis=1)
test_df.drop(['Embarked'], axis=1, inplace=True)

test_passenger_ids = test_df['PassengerId']
test_df.drop(['PassengerId'], axis=1, inplace=True)

# test_pred = model.predict(test_df)

# submission_df = pd.DataFrame({
#     'PassengerId': test_passenger_ids,
#     'Survived': test_pred
# })
#
# submission_df.to_csv('submission.csv', index=False)
# print("✅ submission.csv is ready!")

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
#
# model_rf = RandomForestClassifier(n_estimators=77, random_state=19)
# model_rf.fit(X_train, Y_train)
# pred = model_rf.predict(X_val)
#
# accuracy = accuracy_score(Y_val, pred)
#
# print("Accuracy score of random forest model:", accuracy)
#
# test_pred = model_rf.predict(test_df)
#
# submission_df = pd.DataFrame({
#     'PassengerId': test_passenger_ids,
#     'Survived': test_pred
# })
#
# submission_df.to_csv('submission2.csv', index=False)
# print("Submission by rf  model is ready!")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

model_svm = KNeighborsClassifier(n_neighbors=10)
model_svm.fit(X_train, Y_train)
model_pred = model_svm.predict(X_val)
model_svm_accuracy = accuracy_score(Y_val, model_pred)

print("Accuracy of SVM model:", model_svm_accuracy)