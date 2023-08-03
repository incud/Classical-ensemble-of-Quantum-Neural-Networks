import pennylane as qml
import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
import os
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from embedding import embedding, ry_embedding, rx_embedding
from ansatz import get_ansatz


IBM_QISKIT_HUB = 'MYHUB'
IBM_QISKIT_GROUP = 'MYGROUP'
IBM_QISKIT_PROJECT = 'MYPROJECT'


def create_circuit(n_qubits,backend,layers,ansatz,ibm_device=None, ibm_token=None):
    if backend == 'jax':
        device = qml.device("default.qubit.jax", wires=n_qubits)
    elif backend == 'ibmq':
        device = qml.device('qiskit.ibmq', wires=n_qubits, backend=ibm_device,
                            ibmqx_token=ibm_token, hub=IBM_QISKIT_HUB,
                            group=IBM_QISKIT_GROUP, project=IBM_QISKIT_PROJECT)
    else:
        raise ValueError(f"Backend {backend} is unknown")
    ansatz, params_per_layer = get_ansatz(ansatz,n_qubits)

    @qml.qnode(device, interface='jax')
    def circuit(x, theta):
        #embedding(x, wires=range(n_qubits))
        ry_embedding(x, wires=range(n_qubits))
        for i in range(layers):
            #ry_embedding(x, wires=range(n_qubits))
            ansatz(theta[i * params_per_layer: (i + 1) * params_per_layer], wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=0)),qml.expval(qml.PauliZ(wires=1)),qml.expval(qml.PauliZ(wires=2))]

    return jax.jit(circuit)#


def get_thetas(params):

        def jnp_to_np(value):
            try:
                value_numpy = np.array(value)
                return value_numpy
            except:
                try:
                    value_numpy = np.array(value.primal)
                    return value_numpy
                except:
                    try:
                        value_numpy = np.array(value.primal.aval)
                        return value_numpy
                    except:
                        raise ValueError(f"Cannot convert to numpy value {value}")

        return jnp_to_np(params)
    
    
def evaluate_bagging_predictor(qnn, n_estimators, max_features, max_samples, optimizer, n_qubits, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, bag_working_dir):
        
    @jax.jit
    def cross_entropy_loss(y_true, y_pred):
        return -jnp.mean(jnp.sum(jnp.log(y_pred) * y_true, axis=1))
    
    @jax.jit
    def calculate_ce_cost(X, y, theta):
        yp = qnn(X, theta)
        yp = jax.nn.softmax(yp)

        cost = cross_entropy_loss(y, yp)

        return cost
    
    @jax.jit
    def optimizer_update(opt_state, params, x, y):
        loss, grads = jax.value_and_grad(lambda theta: calculate_ce_cost(x, y, theta))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    y_test_ohe = y_test.copy() 
    y_test = jnp.argmax(y_test, axis=1)
    
    print('bagging\n')
    for i in range(runs):
        
        print(f'bagging - run {i}\n')
        
        # array to gather estimators' predictions
        predictions = []
        predictions_train = []
        predictions_softmax = []
        predictions_softmax_train = []
                
        for j in range(n_estimators):
            
            print(f'bagging - run {i} - estimator {j}\n')
            
            # seed
            key = jax.random.PRNGKey(i+(j*10))
                
            print(f'Using {n_qubits} qubits for bagging\n')
            random_estimator_samples = jax.random.choice(key, a=X_train.shape[0], shape=(int(max_samples*X_train.shape[0]),), p=max_samples*jnp.ones(X_train.shape[0]))
            X_train_est = X_train[random_estimator_samples,:]
            y_train_est = y_train[random_estimator_samples,:]
            random_estimator_features = jax.random.choice(key, a=X_train_est.shape[1], shape=(max(1,int(max_features*X_train_est.shape[1])),), replace=False, p=max_features*jnp.ones(X_train_est.shape[1]))
            X_train_est = X_train_est[:,random_estimator_features]
        
            # get number of circuit params
            _, params_per_layer = get_ansatz(varform, n_qubits)
            
            # initialize circuit params
            initial_params = jax.random.normal(key, shape=(layers * params_per_layer,))
            params = jnp.copy(initial_params)
    
            # initialize optimizer
            opt_state = optimizer.init(initial_params)
            
            start_time_tr = time.time() 
            ##### fit #####
            for epoch in range(epochs):
                key = jax.random.split(key)[0]
                params, opt_state, cost = optimizer_update(opt_state, params, X_train_est, y_train_est)
                if epoch % 5 == 0:
                    print(f'epoch: {epoch} - cost: {cost}')
            print()
    
            end_time_tr = time.time()-start_time_tr
            print('optimization time: ',end_time_tr)
               
            # save training time
            save_dir = bag_working_dir / f"estimator_{j}"
            os.makedirs(save_dir,  0o755,  exist_ok=True)
            np.save(str(save_dir) + f"/time_training_{i}.npy",end_time_tr)
            
            # save parameters
            thetas = get_thetas(params)
            np.save(str(save_dir) + f"/thetas_{i}.npy", thetas)
            
            ##### predict #####
            start_time_ts = time.time() 
            y_predict = qnn(X_test[:,random_estimator_features], params)
            y_predict = jax.nn.softmax(y_predict)
            y_predict_softmax = y_predict.copy()
            y_predict_train = qnn(X_train[:,random_estimator_features], params)
            y_predict_train = jax.nn.softmax(y_predict_train)
            y_predict_softmax_train = y_predict_train.copy()
            print(f'Error of bagging estimator {j} on test set: {cross_entropy_loss(y_test_ohe,y_predict)}\n')
            y_predict = jnp.argmax(y_predict, axis=1)
            y_predict_train = jnp.argmax(y_predict_train, axis=1)
            end_time_ts = time.time()-start_time_ts
            np.save(str(save_dir) + f"/time_test_{i}.npy",end_time_ts)

            # save predictions
            np.save(str(save_dir) + f"/y_predict_{i}.npy", y_predict)
            print(f'Accuracy of bagging estimator {j} on test set: {accuracy_score(y_test,y_predict)}\n')
            predictions.append(y_predict)
            predictions_train.append(y_predict_train)
            predictions_softmax.append(y_predict_softmax)
            predictions_softmax_train.append(y_predict_softmax_train)
                
        print(f'bagging - run {i} - bagging model {i}\n')
        
        # change save directory
        save_dir = bag_working_dir / "ensemble_model"
        os.makedirs(save_dir,  0o755,  exist_ok=True)
        
        # transform list of predictions into an array
        predictions = np.array(predictions)
        predictions_train = np.array(predictions_train)
        predictions_softmax = np.array(predictions_softmax)
        predictions_softmax_train = np.array(predictions_softmax_train)

        ##### predict #####
        start_time_ts = time.time() 
        
        maj_voting = False
        
        if maj_voting:
            # compute mode (majority voting) of estimators' predictions
            y_predict = jnp.mode(predictions).reshape(-1,1)
            y_predict_train = jnp.mode(predictions_train).reshape(-1,1)
            
            end_time_ts = time.time()-start_time_ts
            np.save(str(save_dir) + f"/time_test_{i}.npy",end_time_ts)
            
            # save predictions
            np.save(str(save_dir) + f"/y_predict_{i}.npy", y_predict)
            np.save(str(save_dir) + f"/y_predict_train_{i}.npy", y_predict_train)
            print(f'Accuracy of bagging on test set: {accuracy_score(y_test,y_predict)}\n')
        
        else:
            # compute average of estimators' predictions
            y_predict = jnp.mean(predictions_softmax,axis=0).reshape(-1,3)
            y_predict_train = jnp.mean(predictions_softmax_train,axis=0).reshape(-1,3)
            print(f'Error of bagging on test set: {cross_entropy_loss(y_test_ohe,y_predict)}\n')
    
            y_predict = jnp.argmax(y_predict, axis=1)
            y_predict_train = jnp.argmax(y_predict_train, axis=1)
            end_time_ts = time.time()-start_time_ts
            np.save(str(save_dir) + f"/time_test_{i}.npy",end_time_ts)
            
            # save predictions
            np.save(str(save_dir) + f"/y_predict_{i}.npy", y_predict)
            np.save(str(save_dir) + f"/y_predict_train_{i}.npy", y_predict_train)
            print(f'Accuracy of bagging on test set: {accuracy_score(y_test,y_predict)}\n')
        
        # Plot 3D data
        # if i == 0:
        #     # plt.scatter(X_test[:,0], y_test, label='Original points')
        #     # plt.scatter(X_test[:,0], y_predict, c='y', label='Predicted points')
        #     # plt.legend()
        #     # plt.savefig(str(save_dir) + f"/full_model_pred.png")
        #     # plt.close('all')
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     pca = PCA(n_components=2)
        #     X_test_pca = pca.fit_transform(X_test)
        #     #X_test_pca=X_test
        #     target_names=['setosa','versicolor','virginica']
        #     ax.scatter(X_test_pca[:,0], X_test_pca[:,1], y_predict, 
        #                cmap=plt.cm.Paired, c=y_predict)
        #     for k in range(3):
        #         ax.scatter(X_test_pca[y_predict==k, 0], X_test_pca[y_predict==k, 1], y_predict[y_predict==k],
        #            label=target_names[k])
        #     ax.set_title("First two P.C.")
        #     ax.set_xlabel("P.C. 1")
    
        #     ax.set_ylabel("P.C. 2")
    
        #     ax.set_zlabel("class")
    
        #     plt.legend(numpoints=1)
        #     plt.show()
        #     plt.savefig(str(save_dir) + "/bagging_model_pred.png")
        #     plt.close('all')

def evaluate_full_model_predictor(qnn, optimizer, n_qubits, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, full_model_working_dir):
    
    @jax.jit
    def cross_entropy_loss(y_true, y_pred):
        return -jnp.mean(jnp.sum(jnp.log(y_pred) * y_true, axis=1))
    
    @jax.jit
    def calculate_ce_cost(X, y, theta):
        yp = qnn(X, theta)
        yp = jax.nn.softmax(yp)

        cost = cross_entropy_loss(y, yp)

        return cost
    
    @jax.jit
    def optimizer_update(opt_state, params, x, y):
        loss, grads = jax.value_and_grad(lambda theta: calculate_ce_cost(x, y, theta))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    y_test_ohe = y_test.copy() 
    y_test = jnp.argmax(y_test, axis=1)

    for i in range(runs):
        
        print(f'full_model - run {i}\n')

        # seed
        key = jax.random.PRNGKey(i)
        
        # get number of circuit params
        _, params_per_layer = get_ansatz(varform, n_qubits)
        
        # initialize circuit params
        initial_params = jax.random.normal(key, shape=(layers * params_per_layer,))
        params = jnp.copy(initial_params)

        # initialize optimizer
        opt_state = optimizer.init(initial_params)
        
        start_time_tr = time.time() 
        ##### fit #####
        for epoch in range(epochs):
            key = jax.random.split(key)[0]
            params, opt_state, cost = optimizer_update(opt_state, params, X_train, y_train)
            if epoch % 5 == 0:
                print(f'epoch: {epoch} - cost: {cost}')
        print()

        end_time_tr = time.time()-start_time_tr
        print('optimization time: ',end_time_tr)
           
        # save training time
        save_dir = full_model_working_dir 
        os.makedirs(save_dir,  0o755,  exist_ok=True)
        np.save(str(save_dir) + f"/time_training_{i}.npy",end_time_tr)
        
        # save parameters
        thetas = get_thetas(params)
        np.save(str(save_dir) + f"/thetas_{i}.npy", thetas)
        
        ##### predict train #####
        y_predict_train = qnn(X_train, params)
        y_predict_train = jax.nn.softmax(y_predict_train)
        y_predict_train = jnp.argmax(y_predict_train, axis=1)
        np.save(str(save_dir) + f"/y_predict_train_{i}.npy", y_predict_train)
        
        ##### predict #####
        start_time_ts = time.time() 
        y_predict = qnn(X_test, params)
        y_predict = jax.nn.softmax(y_predict)
        print(f'cross entropy loss: {cross_entropy_loss(y_test_ohe,y_predict)}')
        y_predict = jnp.argmax(y_predict, axis=1)
        end_time_ts = time.time()-start_time_ts
        np.save(str(save_dir) + f"/time_test_{i}.npy",end_time_ts)
        
        # save predictions
        np.save(str(save_dir) + f"/y_predict_{i}.npy", y_predict)
        print(f'Accuracy of fullmodel on test set: {accuracy_score(y_test,y_predict)}\n')
        # Plot 3D data
        # if i%1 == 0:
        #     # plt.scatter(X_test[:,0], y_test, label='Original points')
        #     # plt.scatter(X_test[:,0], y_predict, c='y', label='Predicted points')
        #     # plt.legend()
        #     # plt.savefig(str(save_dir) + f"/full_model_pred.png")
        #     # plt.close('all')
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     pca = PCA(n_components=2)
        #     X_test_pca = pca.fit_transform(X_test)
        #     #X_test_pca=X_test
        #     target_names=['setosa','versicolor','virginica']
        #     ax.scatter(X_test_pca[:,0], X_test_pca[:,1], y_predict, 
        #                cmap=plt.cm.Paired, c=y_predict)
        #     for k in range(3):
        #         ax.scatter(X_test_pca[y_predict==k, 0], X_test_pca[y_predict==k, 1], y_predict[y_predict==k],
        #            label=target_names[k])
        #     ax.set_title("First two P.C.")
        #     ax.set_xlabel("P.C. 1")

        #     ax.set_ylabel("P.C. 2")

        #     ax.set_zlabel("class")

        #     plt.legend(numpoints=1)
        #     plt.show()
        #     plt.savefig(str(save_dir) + "/full_model_pred.png")
        #     plt.close('all')
            
            # from sklearn.neural_network import MLPClassifier
            # clf = MLPClassifier(hidden_layer_sizes=(5,5), random_state=1, activation='identity', max_iter=500).fit(X_train, y_train)
            # y_predict = jax.nn.softmax(clf.predict(X_test))
            # print(f'cross entropy loss: {cross_entropy_loss(y_test_ohe,y_predict)}')
            # y_predict = jnp.argmax(y_predict, axis=1)
            # print(f'accuracy MLP: {accuracy_score(y_test,y_predict)}')
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(X_test_pca[:,0], X_test_pca[:,1], y_test)
            # ax.scatter(X_test_pca[:,0], X_test_pca[:,1], y_predict)
            # plt.show()
            # ax.view_init(3, -83)
            # plt.close('all')
            
            
            
            
# algorithm for boosting classifier was taken from here: https://dafriedman97.github.io/mlbook/content/c6/s2/boosting.html
def evaluate_adaboost_predictor(qnn, n_estimators, optimizer, n_qubits, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, ada_working_dir):
        
    @jax.jit
    def cross_entropy_loss(y_true, y_pred):
        return -jnp.mean(jnp.sum(jnp.log(y_pred) * y_true, axis=1))
    
    @jax.jit
    def calculate_ce_cost(X, y, theta):
        yp = qnn(X, theta)
        yp = jax.nn.softmax(yp)

        cost = cross_entropy_loss(y, yp)

        return cost
    
    @jax.jit
    def optimizer_update(opt_state, params, x, y):
        loss, grads = jax.value_and_grad(lambda theta: calculate_ce_cost(x, y, theta))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    y_test_ohe = y_test.copy() 
    y_test = jnp.argmax(y_test, axis=1)
    
    y_train_argmax = jnp.argmax(y_train, axis=1)
    
    print('adaboost\n')
    for i in range(runs):
        
        print(f'adaboost - run {i}\n')
        
        # number of training samples
        N = X_train.shape[0]
        
        # number of training classes
        num_classes = len(jnp.unique(y_train_argmax))
        
        # array to gather estimators' predictions
        s = np.empty((n_estimators,len(y_train),num_classes))
        
        # array to gather parameters
        estimators_params = []
        
        # weights applied to training samples
        weights = np.repeat(1/N, N)
                
        for t in range(n_estimators):
            
            print(f'adaboost - run {i} - estimator {t}\n')
            
            # seed
            key = jax.random.PRNGKey(i+(t*10))

            random_estimator_samples = jax.random.choice(key, a=N, shape=(int(X_train.shape[0]),), p=weights)
            X_train_est = X_train[random_estimator_samples,:]
            y_train_est = y_train[random_estimator_samples,:]
            
            # get number of circuit params
            _, params_per_layer = get_ansatz(varform, n_qubits)
            
            # initialize circuit params
            initial_params = jax.random.normal(key, shape=(layers * params_per_layer,))
            params = jnp.copy(initial_params)
    
            # initialize optimizer
            opt_state = optimizer.init(initial_params)
            
            start_time_tr = time.time() 
            ##### fit #####
            for epoch in range(epochs):
                key = jax.random.split(key)[0]
                params, opt_state, cost = optimizer_update(opt_state, params, X_train_est, y_train_est)
                if epoch % 5 == 0:
                    print(f'epoch: {epoch} - cost: {cost}')
            print()
            
            y_predict_train = qnn(X_train, params)
            y_predict_train = jax.nn.softmax(y_predict_train)
            y_predict_softmax = y_predict_train.copy()
            y_predict_train = jnp.argmax(y_predict_train, axis=1)
            
            s[t,:,:] = (num_classes-1)*(jnp.log(y_predict_softmax)-((1/num_classes)*(jnp.sum(jnp.log(y_predict_softmax),axis=1)[:, jnp.newaxis])))
            
            
            if t < n_estimators-1:
                weights = jnp.array([w*jnp.exp(-1*((num_classes-1)/num_classes)*(jnp.dot(y_train[i],jnp.log(y_predict_softmax[i]).T))) for i, w in enumerate(weights)])
    
                weights /= jnp.sum(weights)
            
            end_time_tr = time.time()-start_time_tr
            print('optimization time: ',end_time_tr)
               
            ## Append stuff
            estimators_params.append(params)
            
            # save training time
            save_dir = ada_working_dir / f"estimator_{t}"
            os.makedirs(save_dir,  0o755,  exist_ok=True)
            np.save(str(save_dir) + f"/time_training_{i}.npy",end_time_tr)
            
            # save parameters
            thetas = get_thetas(params)
            np.save(str(save_dir) + f"/thetas_{i}.npy", thetas)

        y_train_hat = jnp.argmax(jnp.sum(s,axis=0),axis=0)
    
        
        print(f'adaboost - run {i} - boosting model {i}\n')
        
        # change save directory
        save_dir = ada_working_dir / "ensemble_model"
        os.makedirs(save_dir,  0o755,  exist_ok=True)
            
        
        ##### predict #####
        start_time_ts = time.time() 
        
        s = np.empty((n_estimators,len(y_test),num_classes))
        for t, params in enumerate(estimators_params):
            y_predict_test = qnn(X_test, params)
            y_predict_test = jax.nn.softmax(y_predict_test)
            y_predict_softmax = y_predict_test.copy()
            
            s[t,:,:] = (num_classes-1)*(jnp.log(y_predict_softmax)-((1/num_classes)*(jnp.sum(jnp.log(y_predict_softmax),axis=1)[:, jnp.newaxis])))
            
        y_predict = jnp.sum(s,axis=0)
        y_predict = jnp.argmax(y_predict,axis=1)
        
        end_time_ts = time.time()-start_time_ts
        np.save(str(save_dir) + f"/time_test_{i}.npy",end_time_ts)
        
        
        ##### predict train #####
        start_time_ts = time.time() 
        
        s = np.empty((n_estimators,len(y_train_argmax),num_classes))
        for t, params in enumerate(estimators_params):
            y_predict_train = qnn(X_train, params)
            y_predict_train = jax.nn.softmax(y_predict_train)
            y_predict_softmax_train = y_predict_train.copy()
            
            s[t,:,:] = (num_classes-1)*(jnp.log(y_predict_softmax_train)-((1/num_classes)*(jnp.sum(jnp.log(y_predict_softmax_train),axis=1)[:, jnp.newaxis])))
            
        y_predict_train = jnp.sum(s,axis=0)
        y_predict_train = jnp.argmax(y_predict_train,axis=1)
        
        
        
        # save predictions
        np.save(str(save_dir) + f"/y_predict_{i}.npy", y_predict)
        np.save(str(save_dir) + f"/y_predict_train_{i}.npy", y_predict_train)
        print(f'Accuracy of boosting on test set: {accuracy_score(y_test,y_predict)}\n')
