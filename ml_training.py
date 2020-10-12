import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV

def grid_search(X, target, estimator, param_grid, scoring, cv):
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv,
                                        scoring=scoring, n_jobs=-1, verbose=1)
    tmp = X.copy()
    grid = grid.fit(tmp, target)

    result = pd.DataFrame(grid.cv_results_).sort_values(by='mean_test_score', ascending=False).reset_index()

    del result['params']

    times = [col for col in result.columns if col.endswith('_time')]
    params = [col for col in result.columns if col.startswith('param_')]

    result = result[params + ['mean_test_score', 'std_test_score'] + times]
    return result, grid.best_params_, grid



def save_ml_model(model, filename = None, path = None):

    model_filename = '' + filename + '.pkl'

    if path != None:
        model_filename = path + model_filename

    joblib.dump(model, model_filename)


def load_ml_model(filename, path = None):

    model_filename = ''+filename+'.pkl'

    if path != None:
        model_filename = path + model_filename

    return joblib.load(model_filename)


