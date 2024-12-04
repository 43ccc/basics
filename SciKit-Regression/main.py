from util import get_data
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os

RANDOM_SEED = 0

def evaluate_model(model, model_name, image_folder='./evaluation_images'):
    # Folder for figure
    os.makedirs(image_folder, exist_ok=True)

    # Load data, train model
    in_train, in_test, out_train, out_test = get_data(random_seed=RANDOM_SEED)
    model.fit(in_train, out_train)
    pred = model.predict(in_test)

    # Quantitative
    mse = mean_squared_error(out_test, pred)
    corr = np.corrcoef(out_test, pred)[0,1]

    # Qualitative
    plt.scatter(out_test, pred, label=f'MSE: {mse}\nCorr: {corr}')

    # Make plot pretty and add quantitative into legend
    x = np.linspace(*plt.xlim()) # Points for 45 degree line
    plt.plot(x, x, linestyle='--', color='k', lw=2)
    plt.xlabel('Target Value')
    plt.ylabel('Predicted Value')
    plt.title(f'Performance of {model_name}')
    plt.legend()
    plt.savefig(os.path.join(image_folder, model_name))
    plt.clf()

    return mse, corr

def plot_models(mse, corr, image_folder='./evaluation_images'):
    plt.bar(mse.keys(), mse.values())
    plt.ylabel('MSE')
    plt.title('MSE of different Regression models, lower is better')
    plt.savefig(os.path.join(image_folder, 'mse_results'))
    plt.clf()

    plt.bar(corr.keys(), corr.values())
    plt.ylabel('Corr')
    plt.title('Correlation Coefficient of different Regression models, higher is better')
    plt.savefig(os.path.join(image_folder, 'corr_results'))


def main():
    mse = {}
    corr = {}

    model = LinearRegression(n_jobs=-1)
    mse['LinearRegression'], corr['LinearRegression'] = evaluate_model(model, 'LinearRegression')

    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, max_features=None, min_samples_leaf=1, random_state=RANDOM_SEED)
    mse['RandomForest'], corr['RandomForest'] = evaluate_model(model, 'RandomForest')

    model = AdaBoostRegressor(n_estimators=50, random_state=RANDOM_SEED, learning_rate=0.5)
    mse['AdaBoost'], corr['AdaBoost'] = evaluate_model(model, 'AdaBoost')

    plot_models(mse, corr)


if __name__ == '__main__':
    main()