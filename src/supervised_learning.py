import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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
        'Ridge': {
            'model': Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', Ridge())]),
            'params': {
                'regressor__alpha': [0.1, 1.0, 10.0],
                'regressor__fit_intercept': [True, False],
                'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                'regressor__max_iter': [1000, 5000, 10000]
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
