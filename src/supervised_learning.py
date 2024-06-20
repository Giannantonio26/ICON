import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


def returnBestHyperparametres(dataset, differentialColumn):
    # Separate the features (X) from the target variable (y)
    X = dataset.drop(columns=[differentialColumn])
    y = dataset[differentialColumn]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create transformers for preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create pipelines for each model
    models = {
        'LinearRegression': {
            'model': Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', LinearRegression())]),
            'params': {
                'regressor__fit_intercept': [True, False],
                'regressor__copy_X': [True, False],
            }
        },

        'RandomForestRegressor': {
            'model': Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', RandomForestRegressor(random_state=42))]),
            'params': {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20, 30],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            }
        },
        'DecisionTreeRegressor': {
            'model': Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', DecisionTreeRegressor(random_state=42))]),
            'params': {
                'regressor__max_depth': [None, 10, 20, 30],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4],
                'regressor__splitter': ['best', 'random']
            }
        }
    }

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    best_params = {}
    for model_name, model_info in models.items():
        grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1, error_score='raise')
        try:
            grid_search.fit(X_train, y_train)
            best_params[model_name] = grid_search.best_params_
            best_model = grid_search.best_estimator_
            
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            best_params[model_name]['mse'] = mse
        except Exception as e:
            print(f"Error for model {model_name}: {e}")
            best_params[model_name] = None

    return best_params

def print_hyperparameters_table(best_params):
    data = []
    for model, params in best_params.items():
        if params is not None:
            data.append({
                'Model': model,
                'Hyperparameters': params,
                'MSE': params.pop('mse', None)
            })
    
    df = pd.DataFrame(data)
    pd.set_option('display.max_colwidth', None)
    print(df)


#Funzione che visualizza i grafici delle metriche per ogni modello
def visualizeMetricsGraphs(model):
    models = list(model.keys())

    # Creazione di un array numpy per ogni metrica
    accuracy = np.array([model[clf]['accuracy_list'] for clf in models])
    precision = np.array([model[clf]['precision_list'] for clf in models])
    recall = np.array([model[clf]['recall_list'] for clf in models])
    f1 = np.array([model[clf]['f1'] for clf in models])

    # Calcolo delle medie per ogni modello e metrica
    mean_accuracy = np.mean(accuracy, axis=1)
    mean_precision = np.mean(precision, axis=1)
    mean_recall = np.mean(recall, axis=1)
    mean_f1 = np.mean(f1, axis=1)

    # Creazione del grafico a barre
    bar_width = 0.2
    index = np.arange(len(models))
    plt.bar(index, mean_accuracy, bar_width, label='Accuracy')
    plt.bar(index + bar_width, mean_precision, bar_width, label='Precision')
    plt.bar(index + 2 * bar_width, mean_recall, bar_width, label='Recall')
    plt.bar(index + 3 * bar_width, mean_f1, bar_width, label='F1')
    # Aggiunta di etichette e legenda
    plt.xlabel('Modelli')
    plt.ylabel('Punteggi medi')
    plt.title('Punteggio medio per ogni modello')
    plt.xticks(index + 1.5 * bar_width, models)
    plt.legend()

    # Visualizzazione del grafico
    plt.show()
