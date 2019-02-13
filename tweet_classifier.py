def load_data():
    data = []
    data_labels = []
    with open("./pos_tweets.txt") as f:
        for i in f: 
            data.append(i) 
            data_labels.append('pos')

    with open("./neg_tweets.txt") as f:
        for i in f: 
            data.append(i)
            data_labels.append('neg')

    return data, data_labels

def transform_to_features(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase = False,
    )
    features = vectorizer.fit_transform(
        data
    )
    features_nd = features.toarray()
    return features_nd

def train_then_build_model(data_labels, features_nd, data):
    
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test  = train_test_split(
        features_nd,
        data_labels,
        train_size=0.80, 
        random_state=1234)
        
    from sklearn.linear_model import LogisticRegression
    log_model = LogisticRegression()

    log_model = log_model.fit(X=X_train, y=y_train)
    y_pred = log_model.predict(X_test)
    
    for x in (zip(y_pred[:10], X_test.index.values.tolist()[:10])):
        print(x[0], ':', data[x[1]])
    from sklearn.metrics import accuracy_score
    print("Accuracy={}".format(accuracy_score(data_labels.ix[X_test.index.values.tolist()], y_pred)))

def process():
    data, data_labels = load_data()
    
    features_nd = transform_to_features(data)
    import pandas as pd
    train_then_build_model(pd.Series(data_labels), pd.DataFrame(features_nd), data)


process()
