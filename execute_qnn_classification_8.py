import os
import pathlib
# from sklearn.ensemble import BaggingRegressor
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import click
import numpy as np
from pickle import dump
from functions_classifier import create_circuit, evaluate_bagging_predictor, evaluate_full_model_predictor, \
    evaluate_adaboost_predictor
import optax
import jax
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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
    os.makedirs(f'executions/{mode}/{varform}/{layers}', 0o755, exist_ok=True)
    pathlib.Path(f'executions/{mode}/{varform}/{layers}').mkdir(exist_ok=True)
    os.makedirs(f'executions/{mode}/{varform}/{layers}/{dataset_type}', 0o755, exist_ok=True)
    dataset = pathlib.Path(dataset)
    dataset_name = dataset.name
    working_dir = pathlib.Path(f'executions/{mode}/{varform}/{layers}/{dataset_type}/{dataset_name}')
    working_dir.mkdir(exist_ok=True)
    np.random.seed(seed)

    # split training and testing dataset
    X = np.load(dataset / "X.npy")
    y = np.load(dataset / "y.npy")

    scalerX = MinMaxScaler(feature_range=(-1, 1))
    scaler = OneHotEncoder(sparse_output=False)
    if (working_dir / "y_test.npy").exists():
        print(f"The directory {working_dir} already exists and the dataset are already generated")
        X_train = np.load(working_dir / "X_train.npy")
        X_test = np.load(working_dir / "X_test.npy")
        y_train = np.load(working_dir / "y_train.npy")
        y_test = np.load(working_dir / "y_test.npy")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify = y, random_state=seed * 2)
        np.save(working_dir / "X_train.npy", X_train)
        np.save(working_dir / "X_test.npy", X_test)
        np.save(working_dir / "y_train.npy", y_train)
        np.save(working_dir / "y_test.npy", y_test)

    # scale X
    X_train = scalerX.fit_transform(X_train)
    X_test = scalerX.transform(X_test)
    dump(scalerX, open(working_dir / 'scalerX.pkl', 'wb'))
    
    # scale y
    y_train = scaler.fit_transform(y_train.reshape(-1,1))
    y_test = scaler.transform(y_test.reshape(-1,1))
    dump(scaler, open(working_dir / 'scaler.pkl', 'wb'))

    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # print(X_test.shape)
    pca = PCA(n_components=3)
    X_test_pca = pca.fit_transform(X_test)
    # X_test_pca=X_test
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2],
               cmap=plt.cm.Paired, c=y_test)
    target_names=['setosa','versicolor','virginica']
    for k in range(3):
        ax.scatter(X_test_pca[y_test==k, 0], X_test_pca[y_test==k, 1], X_test_pca[y_test==k, 2],
                   label=target_names[k])
    ax.set_title("First three P.C.")
    ax.set_xlabel("P.C. 1")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("P.C. 2")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("P.C. 3")
    ax.w_zaxis.set_ticklabels([])
    plt.legend(numpoints=1)
    
    plt.show()
    ax.view_init(3, -83)
    plt.close('all')
    
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # print(X_test.shape)
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)
    for i in range(len(X_test_pca)):
        print(X_test_pca[i],y_test[i])
    # X_test_pca=X_test
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], y_test,
               cmap=plt.cm.Paired, c=y_test)
    for k in range(3):
        ax.scatter(X_test_pca[y_test==k, 0], X_test_pca[y_test==k, 1], y_test[y_test==k],
                   label=target_names[k])
    ax.set_title("First two P.C.")
    ax.set_xlabel("P.C. 1")

    ax.set_ylabel("P.C. 2")

    ax.set_zlabel("class")

    plt.legend(numpoints=1)
    
    plt.show()
    ax.view_init(3, -83)
    plt.close('all')
    '''
    
    # number of qubits
    n_qubits = X.shape[1]
    print(f"Using {n_qubits} qubits")

    # backend
    backend = mode

    # quantum circuit
    qnn_tmp = create_circuit(n_qubits, backend, layers, varform,ibm_device, ibm_token)

    # apply vmap on x (first circuit param)
    qnn_batched = jax.vmap(qnn_tmp, (0, None))

    # Jit for faster execution
    qnn = jax.jit(qnn_batched)

    # optimizer
    optimizer = optax.adam(learning_rate=0.1)

    # training options
    runs = 10
    epochs = 150

    # full_model
    full_model_dir = working_dir / "full_model_8"
    os.makedirs(full_model_dir, 0o755, exist_ok=True)
    # if any(pathlib.Path(full_model_dir).iterdir()):
    #     print(f"The directory {full_model_dir} is not empty, skipping this experiment")
    # else:
    evaluate_full_model_predictor(qnn, optimizer, n_qubits, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, full_model_dir)

    # bagging 1
    n_estimators=8
    max_features=0.3
    max_samples=0.2
    bag_dir = working_dir / "bagging_feature03_sample02_8"
    os.makedirs(bag_dir,  0o755,  exist_ok=True)
    # number of qubits bagging
    n_qubits_bag=max(1,int(max_features*X_train.shape[1]))
    # quantum circuit
    qnn_tmp_bag = create_circuit(n_qubits_bag,backend,layers,varform,ibm_device, ibm_token)
    # apply vmap on x (first circuit param)
    qnn_batched_bag = jax.vmap(qnn_tmp_bag, (0, None))
    # Jit for faster execution
    qnn_bag_1 = jax.jit(qnn_batched_bag)
    
    # if any(pathlib.Path(bag_dir).iterdir()):
    #     print(f"The directory {bag_dir} is not empty, skipping this experiment")
    # else:
    evaluate_bagging_predictor(qnn_bag_1, n_estimators, max_features, max_samples, optimizer, n_qubits_bag, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, bag_dir)

    # bagging 2
    n_estimators=8
    max_features=0.3
    max_samples=1.0
    bag_dir = working_dir / "bagging_feature03_sample10_8"
    os.makedirs(bag_dir,  0o755,  exist_ok=True)
    # if any(pathlib.Path(bag_dir).iterdir()):
    #     print(f"The directory {bag_dir} is not empty, skipping this experiment")
    # else:
    evaluate_bagging_predictor(qnn_bag_1, n_estimators, max_features, max_samples, optimizer, n_qubits_bag, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, bag_dir)

    # bagging 3
    n_estimators=8
    max_features=0.5
    max_samples=0.2
    bag_dir = working_dir / "bagging_feature05_sample02_8"
    os.makedirs(bag_dir,  0o755,  exist_ok=True)
    # number of qubits bagging
    n_qubits_bag=max(1,int(max_features*X_train.shape[1]))
    # quantum circuit
    qnn_tmp_bag = create_circuit(n_qubits_bag,backend,layers,varform,ibm_device, ibm_token)
    # apply vmap on x (first circuit param)
    qnn_batched_bag = jax.vmap(qnn_tmp_bag, (0, None))
    # Jit for faster execution
    qnn_bag_2 = jax.jit(qnn_batched_bag)
    
    # if any(pathlib.Path(bag_dir).iterdir()):
    #     print(f"The directory {bag_dir} is not empty, skipping this experiment")
    # else:
    evaluate_bagging_predictor(qnn_bag_2, n_estimators, max_features, max_samples, optimizer, n_qubits_bag, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, bag_dir)

    # bagging 4
    n_estimators=8
    max_features=0.5
    max_samples=1.0
    bag_dir = working_dir / "bagging_feature05_sample10_8"
    os.makedirs(bag_dir,  0o755,  exist_ok=True)
    # if any(pathlib.Path(bag_dir).iterdir()):
    #     print(f"The directory {bag_dir} is not empty, skipping this experiment")
    # else:
    evaluate_bagging_predictor(qnn_bag_2, n_estimators, max_features, max_samples, optimizer, n_qubits_bag, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, bag_dir)

    # bagging 5
    n_estimators=8
    max_features=0.8
    max_samples=0.2
    bag_dir = working_dir / "bagging_feature08_sample02_8"
    os.makedirs(bag_dir,  0o755,  exist_ok=True)
    # number of qubits bagging
    n_qubits_bag=max(1,int(max_features*X_train.shape[1]))
    # quantum circuit
    qnn_tmp_bag = create_circuit(n_qubits_bag,backend,layers,varform,ibm_device, ibm_token)
    # apply vmap on x (first circuit param)
    qnn_batched_bag = jax.vmap(qnn_tmp_bag, (0, None))
    # Jit for faster execution
    qnn_bag_3 = jax.jit(qnn_batched_bag)
    
    # if any(pathlib.Path(bag_dir).iterdir()):
    #     print(f"The directory {bag_dir} is not empty, skipping this experiment")
    # else:
    evaluate_bagging_predictor(qnn_bag_3, n_estimators, max_features, max_samples, optimizer, n_qubits_bag, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, bag_dir)

    # bagging 6
    n_estimators=8
    max_features=0.8
    max_samples=1.0
    bag_dir = working_dir / "bagging_feature08_sample10_8"
    os.makedirs(bag_dir,  0o755,  exist_ok=True)
    # if any(pathlib.Path(bag_dir).iterdir()):
    #     print(f"The directory {bag_dir} is not empty, skipping this experiment")
    # else:
    evaluate_bagging_predictor(qnn_bag_3, n_estimators, max_features, max_samples, optimizer, n_qubits_bag, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, bag_dir)

 
    # adaboost
    n_estimators=8
    ada_dir = working_dir / "adaboost_8"
    os.makedirs(ada_dir, 0o755, exist_ok=True)
    # if any(pathlib.Path(ada_dir).iterdir()):
    #     print(f"The directory {ada_dir} is not empty, skipping this experiment")
    # else:
    evaluate_adaboost_predictor(qnn, n_estimators, optimizer, n_qubits, runs, epochs, layers, varform, X_train, X_test,
                                y_train, y_test, ada_dir)
    
    ######### Classical AdaBoost (DecisionTrees)
    # from sklearn.ensemble import AdaBoostClassifier
    # clf = AdaBoostClassifier(n_estimators=10)
    # clf.fit(X_train, np.argmax(y_train,axis=1))
    # y_predict=clf.predict(X_test)
    # from sklearn.metrics import accuracy_score
    # print(f'Accuracy of bagging on test set: {accuracy_score(np.argmax(y_test,axis=1),y_predict)}\n')


if __name__ == '__main__':
    main()
