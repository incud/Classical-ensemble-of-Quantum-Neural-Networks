import os
import pathlib
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import click
import numpy as np
from qnn import Qnn


@click.group()
def main():
    pass


@main.command()
@click.option('--dataset', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
@click.option('--mode', type=click.Choice(['jax', 'ibm']), required=True)
@click.option('--ibm-device', type=str, required=False)
@click.option('--ibm-token', type=str, required=False)
@click.option('--varform', type=click.Choice(['hardware_efficient', 'tfim', 'ltfim']), required=True)
@click.option('--layers', type=int, required=True)
@click.option('--seed', type=int, required=True)
def experiment(dataset, mode, ibm_device, ibm_token, varform, layers, seed):

    pathlib.Path(f'executions/{mode}').mkdir(exist_ok=True)
    pathlib.Path(f'executions/{mode}/{varform}').mkdir(exist_ok=True)
    pathlib.Path(f'executions/{mode}/{varform}/{layers}').mkdir(exist_ok=True)
    dataset = pathlib.Path(dataset)
    dataset_name = dataset.name
    working_dir = pathlib.Path(f'executions/{mode}/{varform}/{layers}/{dataset_name}')
    working_dir.mkdir(exist_ok=True)
    np.random.seed(seed)

    # split training and testing dataset
    X = np.load(dataset / "X.npy")
    y = np.load(dataset / "y.npy")
    if (working_dir / "y_test.npy").exists():
        print(f"The directory {working_dir} already exists and the dataset are already generated")
        X_train = np.load(working_dir / "X_train.npy")
        X_test = np.load(working_dir / "X_test.npy")
        y_train = np.load(working_dir / "y_train.npy")
        y_test = np.load(working_dir / "y_test.npy")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=seed * 2)
        np.save(working_dir / "X_train.npy", X_train)
        np.save(working_dir / "X_test.npy", X_test)
        np.save(working_dir / "y_train.npy", y_train)
        np.save(working_dir / "y_test.npy", y_test)

    # base estimator
    np.random.seed(seed * 3)
    base_estimator = Qnn(var_form=varform, layers=layers, backend=mode, ibm_device=ibm_device, ibm_token=ibm_token)

    # bagging 1
    np.random.seed(seed * 3 + 1)
    ensemble = BaggingRegressor(base_estimator=base_estimator, n_estimators=10, max_features=0.5, max_samples=0.2, random_state=seed)
    bag_dir = working_dir / "bagging_feature05_sample02"
    bag_dir.mkdir()
    if any(pathlib.Path(bag_dir).iterdir()):
        print(f"The directory {bag_dir} is not empty, skipping such an experiment")
    else:
        evaluate_bagging_predictor(ensemble, X_train, X_test, y_train, y_test, bag_dir)

    # bagging 2
    np.random.seed(seed * 3 + 2)
    ensemble = BaggingRegressor(base_estimator=base_estimator, n_estimators=10, max_features=0.5, max_samples=1.0, random_state=seed)
    bag_dir = working_dir / "bagging_feature05_sample10"
    bag_dir.mkdir()
    if any(pathlib.Path(bag_dir).iterdir()):
        print(f"The directory {bag_dir} is not empty, skipping such an experiment")
    else:
        evaluate_bagging_predictor(ensemble, X_train, X_test, y_train, y_test, bag_dir)

    # bagging 3
    np.random.seed(seed * 3 + 3)
    ensemble = BaggingRegressor(base_estimator=base_estimator, n_estimators=10, max_features=1.0, max_samples=0.2, random_state=seed)
    bag_dir = working_dir / "bagging_feature10_sample02"
    bag_dir.mkdir()
    if any(pathlib.Path(bag_dir).iterdir()):
        print(f"The directory {bag_dir} is not empty, skipping such an experiment")
    else:
        evaluate_bagging_predictor(ensemble, X_train, X_test, y_train, y_test, bag_dir)

    # bagging 4
    np.random.seed(seed * 3 + 4)
    ensemble = BaggingRegressor(base_estimator=base_estimator, n_estimators=10, max_features=1.0, max_samples=1.0, random_state=seed)
    bag_dir = working_dir / "bagging_feature10_sample10"
    bag_dir.mkdir()
    if any(pathlib.Path(bag_dir).iterdir()):
        print(f"The directory {bag_dir} is not empty, skipping such an experiment")
    else:
        evaluate_bagging_predictor(ensemble, X_train, X_test, y_train, y_test, bag_dir)

    # adaboost
    np.random.seed(seed * 3 + 5)
    ensemble = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=10, random_state=seed*2, loss='square')
    ada_dir = working_dir / "adaboost"
    ada_dir.mkdir()
    if any(pathlib.Path(ada_dir).iterdir()):
        print(f"The directory {ada_dir} is not empty, skipping such an experiment")
    else:
        evaluate_adaboost_predictor(ensemble, X_train, X_test, y_train, y_test, ada_dir)


def evaluate_bagging_predictor(ensemble, X_train, X_test, y_train, y_test, bag_working_dir):
    # train model
    ensemble.fit(X_train, y_train)
    # predict
    y_predict = ensemble.predict(X_test)
    np.save(bag_working_dir / "y_predict.npy", y_predict)
    # save intermediate prediction
    i = 0
    for estimator, feature_list in zip(ensemble.estimators_, ensemble.estimators_features_):
        theta_comp = estimator.get_thetas()
        np.save(bag_working_dir / f"thetas_{i}.npy", theta_comp)
        y_predict_comp = estimator.predict(X_test[:, feature_list])
        np.save(bag_working_dir / f"y_predict_{i}.npy", y_predict_comp)
        i = i + 1


def evaluate_adaboost_predictor(ensemble, X_train, X_test, y_train, y_test, ada_working_dir):
    # train model
    ensemble.fit(X_train, y_train)
    # predict
    y_predict = ensemble.predict(X_test)
    np.save(ada_working_dir / "Y_predict.npy", y_predict)
    # save intermediate prediction
    i = 0
    for estimator in ensemble.estimators_:
        theta_comp = estimator.get_thetas()
        np.save(ada_working_dir / f"thetas_{i}.npy", theta_comp)
        y_predict_comp = estimator.predict(X_test)
        np.save(ada_working_dir / f"y_predict_{i}.npy", y_predict_comp)
        i = i + 1


if __name__ == '__main__':
    main()
