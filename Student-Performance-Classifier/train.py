import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def main(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['performance'])
    y = df['performance']

    le = LabelEncoder()
    y = le.fit_transform(y)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=0))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    param_grid = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10, 20, None],
        'rf__min_samples_split': [2, 5]
    }

    gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    os.makedirs('model', exist_ok=True)
    joblib.dump((best, le), 'model/model.joblib')

    preds = best.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f'best_params: {gs.best_params_}')
    print(f'accuracy: {acc:.4f}')
    print(classification_report(y_test, preds))
    print('model saved to model/model.joblib')

if __name__ == '__main__':
    main('data/student_data.csv')
