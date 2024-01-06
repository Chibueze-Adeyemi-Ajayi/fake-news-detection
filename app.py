from NLP.fake_news_detection import FakeNewsDectector as fks

data = {
    "fake": "dataset/fake.csv",
    "true": "dataset/true.csv"
}

fks = fks(data)

fake_csv = fks.loadNews(False)
fake_csv["type"] = 1 #true news [label]
true_csv = fks.loadNews(True)
true_csv["type"] = 0 #fake news [label]

df = fks.mergeData([fake_csv, true_csv])
df["vector"] = df["text"].apply(lambda text: fks.convertTextToVec(text))

X_train, X_test, y_train, y_test = fks.MLCategorizer(df, "vector", "type")

X_train = fks.stackVector(X_train)
# print(X_train)

knnClf = fks.knnClassifier(X_train, y_train)

X_test = fks.stackVector(X_test)
y_pred = fks.predictTextWithKNN(X_test)
print(fks.analizeTextWithKNN(y_pred, y_test))

# clf = fks.classifier(X_train, y_train)
# metrics = fks.analizeReport(X_test, y_test)

# print(metrics)

while True:
    text = input("Type the news to classify: ")
    vec = fks.convertTextToVec(text)
    arr = fks.stackVector(vec)
    prediction = fks.predictTextWithKNN([arr])
    print(prediction),