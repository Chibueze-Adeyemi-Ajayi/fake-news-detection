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

X_train, X_test, y_train, y_test = fks.MLCategorizer(df)

clf = fks.classifier(X_train, y_train)
metrics = fks.analizeReport(X_test, y_test)

print(metrics)

while True:
    text = input("Input fake news subect to detect: ")
    print(fks.predictText(text))