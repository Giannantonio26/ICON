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
from tabulate import tabulate
from sklearn.model_selection import RepeatedKFold, learning_curve, train_test_split, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error


def getBestHyperparametres(dataset, differentialColumn):
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
            for param, value in params.items():
                if value==None:
                    value = "None"
                if param != 'mse':
                    data.append({
                        'Modello': model,
                        'Parametro': param,
                        'Valore': value
                    })
    
    df = pd.DataFrame(data)
    pd.set_option('display.max_colwidth', None)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))



# Funzione che esegue il training del modello mediante cross validation
def trainModelKFold(dataSet, differentialColumn):
    model = {
        'LinearRegression': {
            'mae_list': [],
            'mse_list': [],
        },
        'RandomForestRegressor': {
            'mae_list': [],
            'mse_list': [],
        },
        'DecisionTreeRegressor': {
            'mae_list': [],
            'mse_list': [],
        }
    }
    
    bestParameters = getBestHyperparametres(dataSet, differentialColumn)
    print("\033[94m" + str(bestParameters) + "\033[0m")
    
    categorical_cols = dataSet.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = dataSet.select_dtypes(exclude=['object']).columns.tolist()
    numeric_cols.remove(differentialColumn)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    X = dataSet.drop(differentialColumn, axis=1)
    y = dataSet[differentialColumn]

    dtc = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(
            splitter='best',
            max_depth=bestParameters['DecisionTreeRegressor']['regressor__max_depth'],
            min_samples_split=bestParameters['DecisionTreeRegressor']['regressor__min_samples_split'],
            min_samples_leaf=bestParameters['DecisionTreeRegressor']['regressor__min_samples_leaf'],
            random_state=42
        ))
    ])

    rfc = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=bestParameters['RandomForestRegressor']['regressor__n_estimators'],
            max_depth=bestParameters['RandomForestRegressor']['regressor__max_depth'],
            min_samples_split=bestParameters['RandomForestRegressor']['regressor__min_samples_split'],
            min_samples_leaf=bestParameters['RandomForestRegressor']['regressor__min_samples_leaf'],
            random_state=42
        ))
    ])

    reg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression(
            fit_intercept=bestParameters['LinearRegression']['regressor__fit_intercept'],
            copy_X=bestParameters['LinearRegression']['regressor__copy_X']
        ))
    ])
    
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    
    scoring_metrics = {
        'mae': make_scorer(mean_absolute_error),
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
    }
    
    results_dtc = {metric: cross_val_score(dtc, X, y, scoring=scorer, cv=cv) for metric, scorer in scoring_metrics.items()}
    results_rfc = {metric: cross_val_score(rfc, X, y, scoring=scorer, cv=cv) for metric, scorer in scoring_metrics.items()}
    results_reg = {metric: cross_val_score(reg, X, y, scoring=scorer, cv=cv) for metric, scorer in scoring_metrics.items()}
    
    model['LinearRegression']['mae_list'] = results_reg['mae']
    model['LinearRegression']['mse_list'] = -results_reg['mse']
    model['DecisionTreeRegressor']['mae_list'] = results_dtc['mae']
    model['DecisionTreeRegressor']['mse_list'] = -results_dtc['mse']
    model['RandomForestRegressor']['mae_list'] = results_rfc['mae']
    model['RandomForestRegressor']['mse_list'] = -results_rfc['mse']
    
    #plot_learning_curves(dtc, X, y, differentialColumn, 'DecisionTree')
    #plot_learning_curves(rfc, X, y, differentialColumn, 'RandomForest')
    #plot_learning_curves(reg, X, y, differentialColumn, 'LinearRegression')
    visualizeMetricsGraphs(model)
    
    return model


#def plot_learning_curves(model, X, y, differentialColumn, model_name):


#Funzione che visualizza i grafici delle metriche per ogni modello
def visualizeMetricsGraphs(model):
    # Estraiamo le metriche dai risultati del modello
    metrics = ['mae', 'mse']
    models = ['LinearRegression', 'DecisionTreeRegressor', 'RandomForestRegressor']
    
    # Prepariamo i dati per il plot
    mae_scores = {
        'LinearRegression': model['LinearRegression']['mae_list'],
        'DecisionTreeRegressor': model['DecisionTreeRegressor']['mae_list'],
        'RandomForestRegressor': model['RandomForestRegressor']['mae_list']
    }
    
    mse_scores = {
        'LinearRegression': model['LinearRegression']['mse_list'],
        'DecisionTreeRegressor': model['DecisionTreeRegressor']['mse_list'],
        'RandomForestRegressor': model['RandomForestRegressor']['mse_list']
    }
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    fig.suptitle('Performance Metrics Comparison')

    for i, metric in enumerate(metrics):
        for j, model_name in enumerate(models):
            scores = mae_scores[model_name] if metric == 'mae' else mse_scores[model_name]
            ax = axes[i, j]
            ax.boxplot(scores, vert=False)
            ax.set_title(f'{model_name} - {metric.upper()}')
            ax.set_yticklabels([])
    
    plt.tight_layout()
    plt.show()
