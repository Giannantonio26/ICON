import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
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
        'Lasso': {
            'model': Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', Lasso())]),
            'params': {
                'regressor__alpha': [0.1, 1.0, 10.0],
                'regressor__fit_intercept': [True, False],
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


# Funzione per calcolare deviazione standard e varianza
def calculate_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_std = np.std(train_preds)
    test_std = np.std(test_preds)
    
    train_var = np.var(train_preds)
    test_var = np.var(test_preds)
    
    return {
        "Train_Std": train_std,
        "Test_Std": test_std,
        "Train_Var": train_var,
        "Test_Var": test_var
    }

# Funzione che esegue il training del modello mediante cross validation
def trainModelKFold(dataSet, differentialColumn):
    model = {
        'LinearRegression': {
            'mae_list': [],
            'mse_list': [],
        },
        'Lasso': {
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

    lasso = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Lasso(
            alpha=bestParameters['Lasso']['regressor__alpha'],
            fit_intercept=bestParameters['Lasso']['regressor__fit_intercept']
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
    results_lasso = {metric: cross_val_score(lasso, X, y, scoring=scorer, cv=cv) for metric, scorer in scoring_metrics.items()}

    model['LinearRegression']['mae_list'] = results_reg['mae']
    model['LinearRegression']['mse_list'] = -results_reg['mse']
    model['DecisionTreeRegressor']['mae_list'] = results_dtc['mae']
    model['DecisionTreeRegressor']['mse_list'] = -results_dtc['mse']
    model['RandomForestRegressor']['mae_list'] = results_rfc['mae']
    model['RandomForestRegressor']['mse_list'] = -results_rfc['mse']
    model['Lasso']['mae_list'] = results_lasso['mae']
    model['Lasso']['mse_list'] = results_lasso['mse']

    #plot_learning_curves(dtc, X, y, 'DecisionTree')
    #plot_learning_curves(rfc, X, y, 'RandomForest')
    #plot_learning_curves(reg, X, y, 'LinearRegression')
    plot_learning_curves(lasso, X, y, 'LinearRegression con regolarizzatore L1')
    visualizeMetricsGraphs(model)

    return model


def plot_learning_curves(model, X, y, model_name):
    # Generiamo i dati della curva di apprendimento
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='neg_mean_squared_error', 
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)
    
    # Calcola gli errori su addestramento e test
    train_errors = 1 - train_scores
    test_errors = 1 - test_scores


    # Calcoliamo la media e la deviazione standard degli errori
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    train_scores_var = np.var(train_errors, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    test_scores_var = np.var(test_errors, axis=1)

    
    # Stampa deviazione standard e varianza
    print(
        f"\033[95m{model_name} - Train Error Std: {train_scores_std[-1]}, Test Error Std: {test_scores_std[-1]}, Train Error Var: {train_scores_var[-1]}, Test Error Var: {test_scores_var[-1]}\033[0m")



    # Tracciamo le curve di apprendimento
    plt.figure(figsize=(12, 6))
    plt.title(f"Learning Curves ({model_name})")
    plt.xlabel("Training examples")
    plt.ylabel("Error")
    plt.grid()
    
    # Plot dei punteggi di addestramento
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training error")   
    # Plot dei punteggi di test
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test error")   
    plt.legend(loc="best")
    plt.show()



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

