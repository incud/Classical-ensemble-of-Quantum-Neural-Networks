import os
import pathlib
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import click
import numpy as np
from qnn import Qnn
from pickle import dump
import time

@click.group()
def main():
    pass


@main.command()
@click.option('--dataset', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
@click.option('--dataset-type', type=str, required=True)
@click.option('--mode', type=click.Choice(['jax', 'ibm']), required=True)
@click.option('--ibm-device', type=str, required=False)
@click.option('--ibm-token', type=str, required=False)
@click.option('--varform', type=click.Choice(['hardware_efficient', 'tfim', 'ltfim']), required=True)
@click.option('--layers', type=int, required=True)
@click.option('--seed', type=int, required=True)
def experiment(dataset, dataset_type, mode, ibm_device, ibm_token, varform, layers, seed):

    pathlib.Path(f'executions/{mode}').mkdir(exist_ok=True)
    pathlib.Path(f'executions/{mode}/{varform}').mkdir(exist_ok=True)
    pathlib.Path(f'executions/{mode}/{varform}/{layers}').mkdir(exist_ok=True)
    dataset = pathlib.Path(dataset)
    dataset_name = dataset.name
    working_dir = pathlib.Path(f'executions/{mode}/{varform}/{layers}/{dataset_type}/{dataset_name}')
    working_dir.mkdir(exist_ok=True)
    np.random.seed(seed)

    # split training and testing dataset
    X = np.load(dataset / "X.npy")
    y = np.load(dataset / "y_no_noise.npy")
    scaler = MinMaxScaler(feature_range=(-1, 1))                                          
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
        
    # scale y
    y_train = scaler.fit_transform(y_train.reshape(-1,1)).reshape(-1,)
    y_test = scaler.transform(y_test.reshape(-1,1)).reshape(-1,)
    dump(scaler, open(working_dir / 'scaler.pkl', 'wb'))
    
    # Plot 3D data
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(X_test[:,0], X_test[:,1], y_test)
    # plt.show()
    
    # base estimator
    #np.random.seed(seed * 3)
    base_estimator = Qnn(var_form=varform, layers=layers, backend=mode, seed=seed, ibm_device=ibm_device, ibm_token=ibm_token)
    
    # full_model
    full_model_dir = working_dir / "full_model"
    os.makedirs(full_model_dir,  0o755,  exist_ok=True)
    if any(pathlib.Path(full_model_dir).iterdir()):
        print(f"The directory {full_model_dir} is not empty, skipping this experiment")
    else:
        evaluate_full_model_predictor(base_estimator, X_train, X_test, y_train, y_test, full_model_dir)

    # bagging 1
    #np.random.seed(seed * 3 + 1)
    n_estimators=10
    max_features=0.5
    max_samples=0.2
    bag_dir = working_dir / "bagging_feature05_sample02"
    os.makedirs(bag_dir,  0o755,  exist_ok=True)
    if any(pathlib.Path(bag_dir).iterdir()):
        print(f"The directory {bag_dir} is not empty, skipping this experiment")
    else:
        evaluate_bagging_predictor(base_estimator, n_estimators, max_features, max_samples, X_train, X_test, y_train, y_test, bag_dir)

    # bagging 2
    #np.random.seed(seed * 3 + 2)
    n_estimators=10
    max_features=0.5
    max_samples=1.0
    bag_dir = working_dir / "bagging_feature05_sample10"
    os.makedirs(bag_dir,  0o755,  exist_ok=True)
    if any(pathlib.Path(bag_dir).iterdir()):
        print(f"The directory {bag_dir} is not empty, skipping this experiment")
    else:
        evaluate_bagging_predictor(base_estimator, n_estimators, max_features, max_samples, X_train, X_test, y_train, y_test, bag_dir)

    # bagging 3
    #np.random.seed(seed * 3 + 3)
    n_estimators=10
    max_features=1.0
    max_samples=0.2
    bag_dir = working_dir / "bagging_feature10_sample02"
    os.makedirs(bag_dir,  0o755,  exist_ok=True)
    if any(pathlib.Path(bag_dir).iterdir()):
        print(f"The directory {bag_dir} is not empty, skipping this experiment")
    else:
        evaluate_bagging_predictor(base_estimator, n_estimators, max_features, max_samples, X_train, X_test, y_train, y_test, bag_dir)


    # # bagging 4
    # #np.random.seed(seed * 3 + 4)   
    # n_estimators=10
    # max_features=1.0
    # max_samples=1.0
    # bag_dir = working_dir / "bagging_feature10_sample10"
    # os.makedirs(bag_dir,  0o755,  exist_ok=True)
    # if any(pathlib.Path(bag_dir).iterdir()):
        # print(f"The directory {bag_dir} is not empty, skipping this experiment")
    # else:
        # evaluate_bagging_predictor(base_estimator, n_estimators, max_features, max_samples, X_train, X_test, y_train, y_test, bag_dir)


    # adaboost
    #np.random.seed(seed * 3 + 5)
    n_estimators=10
    ada_dir = working_dir / "adaboost"
    os.makedirs(ada_dir,  0o755,  exist_ok=True)
    if any(pathlib.Path(ada_dir).iterdir()):
        print(f"The directory {ada_dir} is not empty, skipping this experiment")
    else:
        evaluate_adaboost_predictor(base_estimator, n_estimators, X_train, X_test, y_train, y_test, ada_dir)


def evaluate_full_model_predictor(base_estimator, X_train, X_test, y_train, y_test, full_model_working_dir):
    # num runs
    runs = 10
    for i in range(runs):
        # seed
        seed = i
        base_estimator.random_state = i
        # train model
        start_time_tr = time.time() 
        base_estimator.fit(X_train, y_train)
        end_time_tr = time.time()-start_time_tr
        save_dir = full_model_working_dir 
        os.makedirs(save_dir,  0o755,  exist_ok=True)
        np.save(str(save_dir) + f"/time_training_{i}.npy",end_time_tr)
        # save parameters
        thetas = base_estimator.get_thetas()
        np.save(str(save_dir) + f"/thetas_{i}.npy", thetas)
        # predict
        start_time_ts = time.time() 
        y_predict = base_estimator.predict(X_test)
        end_time_ts = time.time()-start_time_ts
        np.save(str(save_dir) + f"/time_test_{i}.npy",end_time_ts)
        # save predictions
        np.save(str(save_dir) + f"/y_predict_{i}.npy", y_predict)
        # Plot 3D data
        if i == 0:
            import matplotlib.pyplot as plt
            # plt.scatter(X_test[:,0], y_test, label='Original points')
            # plt.scatter(X_test[:,0], y_predict, c='y', label='Predicted points')
            # plt.legend()
            # plt.savefig(str(save_dir) + f"/full_model_pred.png")
            # plt.close('all')
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            print(X_test.shape)
            ax.scatter(X_test[:,0], X_test[:,1], y_test)
            ax.scatter(X_test[:,0], X_test[:,1], y_predict)
            plt.show()

def evaluate_bagging_predictor(base_estimator, n_estimators, max_features, max_samples, X_train, X_test, y_train, y_test, bag_working_dir):
    # num runs
    runs = 10
    for i in range(runs):
        # seed
        seed = i
        ensemble = BaggingRegressor(base_estimator=base_estimator, n_estimators=n_estimators, max_features=max_features, max_samples=max_samples, random_state=seed)
        # train model
        start_time_tr = time.time() 
        ensemble.fit(X_train, y_train)
        end_time_tr = time.time()-start_time_tr
        save_dir = bag_working_dir / f"ensemble_model"
        os.makedirs(save_dir,  0o755,  exist_ok=True)
        np.save(str(save_dir) + f"/time_training_{i}.npy",end_time_tr)
        # predict
        start_time_ts = time.time() 
        y_predict = ensemble.predict(X_test)
        end_time_ts = time.time()-start_time_ts
        np.save(str(save_dir) + f"/time_test_{i}.npy",end_time_ts)
        # save predictions
        np.save(str(save_dir) + f"/y_predict_{i}.npy", y_predict)
        
        import matplotlib.pyplot as plt
        plt.scatter(X_test[:,0], y_test, label='Original points')
        plt.scatter(X_test[:,0], y_predict, c='y', label='Predicted points')
        plt.legend()
        plt.savefig(str(save_dir) + f"/bagging_model_pred.png")
        plt.close('all')
        
        # save intermediate prediction
        j = 0
        for estimator, feature_list in zip(ensemble.estimators_, ensemble.estimators_features_):
            theta_comp = estimator.get_thetas()
            save_dir = bag_working_dir / f"estimator_{j}"
            os.makedirs(save_dir,  0o755,  exist_ok=True)
            np.save(str(save_dir) + f"/thetas_{i}.npy", theta_comp)
            start_time_ts = time.time() 
            y_predict_comp = estimator.predict(X_test[:, feature_list])
            end_time_ts = time.time()-start_time_ts
            np.save(str(save_dir) + f"/time_test_{i}.npy",end_time_ts)
            np.save(str(save_dir) + f"/y_predict_{i}.npy", y_predict_comp)
            j = j + 1

def evaluate_adaboost_predictor(base_estimator, n_estimators, X_train, X_test, y_train, y_test, ada_working_dir):
    # num runs
    runs = 10
    for i in range(runs):
        # seed
        seed = i
        ensemble = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=n_estimators, random_state=seed, loss='square')
        # train model
        start_time_tr = time.time() 
        ensemble.fit(X_train, y_train)
        end_time_tr = time.time()-start_time_tr
        save_dir = ada_working_dir / f"ensemble_model"
        os.makedirs(save_dir,  0o755,  exist_ok=True)
        np.save(str(save_dir) + f"/time_training_{i}.npy",end_time_tr)
        # predict
        start_time_ts = time.time() 
        y_predict = ensemble.predict(X_test)
        end_time_ts = time.time()-start_time_ts
        np.save(str(save_dir) + f"/time_test_{i}.npy",end_time_ts)
        # save predictions
        np.save(str(save_dir) + f"/y_predict_{i}.npy", y_predict)
        # save intermediate prediction
        j = 0
        for estimator in ensemble.estimators_:
            print(len(ensemble.estimators_))
            theta_comp = estimator.get_thetas()
            save_dir = ada_working_dir / f"estimator_{j}"
            os.makedirs(save_dir,  0o755,  exist_ok=True)
            np.save(str(save_dir) + f"/thetas_{i}.npy", theta_comp)
            start_time_ts = time.time()
            y_predict_comp = estimator.predict(X_test)
            end_time_ts = time.time()-start_time_ts
            np.save(str(save_dir) + f"/time_test_{i}.npy",end_time_ts)
            np.save(str(save_dir) + f"/y_predict_{i}.npy", y_predict_comp)
            j = j + 1
           

if __name__ == '__main__':
    main()
