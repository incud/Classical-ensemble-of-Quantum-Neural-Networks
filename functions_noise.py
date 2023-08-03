import qiskit
import pennylane as qml
import pennylane_qiskit
import qiskit.providers.aer.noise as noise
from qiskit.providers.fake_provider import FakeQasmSimulator, FakeLimaV2#, FakeMontrealV2
import jax
import jax.numpy as jnp
import optax
import numpy as np
import time
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from embedding import embedding, ry_embedding, rx_embedding
from ansatz import get_ansatz

# import Array and set default backend
# from qiskit_dynamics.array import Array
# Array.set_default_backend('jax')

# from qiskit_dynamics.array import wrap

# jit = wrap(jax.jit, decorator=True)

IBM_QISKIT_HUB = 'ibm-q'
IBM_QISKIT_GROUP = 'open'
IBM_QISKIT_PROJECT = 'main'


def create_circuit(n_qubits,backend,layers,ansatz,ibm_device=None, ibm_token=None):
    #backend = 'ibm'
    if backend == 'jax':
        device = qml.device("default.qubit.jax", wires=n_qubits)
    elif backend == 'ibm':
        fake_backend = FakeLimaV2()
        noise_model = noise.NoiseModel.from_backend(fake_backend)
        device = qml.device('qiskit.aer', wires=n_qubits,  noise_model=noise_model, shots=256)
    elif backend == 'noise':
        device = qml.device("default.mixed", wires=n_qubits, readout_prob=0.005)
        
    else:
        raise ValueError(f"Backend {backend} is unknown")
    ansatz, params_per_layer = get_ansatz(ansatz,n_qubits)

    @qml.qnode(device, interface='jax')
    def circuit(x, theta):
        #print(x.shape,theta.shape)
        print(device)
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
        for i in range(layers):
            ansatz(theta[i * params_per_layer: (i + 1) * params_per_layer], wires=range(n_qubits))
           
        p_bitflip = 1e-3
        #qml.BitFlip(p_bitflip, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(wires=0))
    return jax.jit(circuit)



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
    
    
def evaluate_bagging_predictor(qnn, n_estimators, max_features, max_samples, optimizer, n_qubits, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, bag_working_dir, run=0, load_weights=False, starting_estimator=0):
        
    @jax.jit
    def calculate_mse_cost(X, y, theta):
        yp = qnn(X, theta)
        cost = jnp.mean((yp - y) ** 2)
        return cost
    
    @jax.jit
    def optimizer_update(opt_state, params, x, y):
        loss, grads = jax.value_and_grad(lambda theta: calculate_mse_cost(x, y, theta))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    print('bagging\n')
    for i in range(run,runs):
        
        print(f'bagging {bag_working_dir} - run {i}\n')
        
        # array to gather estimators' predictions
        predictions = []
        predictions_train = []
        bag_working_dir = str(bag_working_dir)
                
        for j in range(starting_estimator,n_estimators):
            if (os.path.isfile(bag_working_dir + f"/estimator_{j}/y_predict_{i}_noise.npy")):
                y_predict = np.load(bag_working_dir + f"/estimator_{j}/y_predict_{i}_noise.npy")
                predictions.append(y_predict)
            else:
                print(f'bagging - run {i} - estimator {j}\n')
                
                # seed
                key = jax.random.PRNGKey(i+(j*10))
                    
                random_estimator_samples = jax.random.choice(key, a=X_train.shape[0], shape=(int(max_samples*X_train.shape[0]),), p=max_samples*jnp.ones(X_train.shape[0]))
                X_train_est = X_train[random_estimator_samples,:]
                y_train_est = y_train[random_estimator_samples]
                random_estimator_features = jax.random.choice(key, a=X_train_est.shape[1], shape=(max(1,int(max_features*X_train_est.shape[1])),), replace=False, p=max_features*jnp.ones(X_train_est.shape[1]))
                X_train_est = X_train_est[:,random_estimator_features]
            
                # get number of circuit params
                _, params_per_layer = get_ansatz(varform, n_qubits)
                
                # initialize circuit params
                initial_params = jax.random.normal(key, shape=(layers * params_per_layer,))
                params = jnp.copy(initial_params)
        
                # initialize optimizer
                opt_state = optimizer.init(initial_params)
                starting_epoch = -1
                if load_weights and os.path.isfile(bag_working_dir + f'/estimator_{j}/saved_params_{i}_{j}.npy'):
                    params = np.load(bag_working_dir + f'/estimator_{j}/saved_params_{i}_{j}.npy')
                    params = jnp.copy(params)
                    with open(bag_working_dir + f'/estimator_{j}/starting_epoch_{i}_{j}.txt', 'r') as f:
                        starting_epoch = int(f.readline())
                        print(starting_epoch,type(starting_epoch))
                    # initialize optimizer
                    with open(bag_working_dir + f'/estimator_{j}/opt_state_{i}_{j}.pickle', 'rb') as handle:
                        opt_state = pickle.load(handle)
                
                start_time_tr = time.time() 
                ##### fit #####
                for epoch in range(starting_epoch+1,epochs):
                    key = jax.random.split(key)[0]
                    params, opt_state, cost = optimizer_update(opt_state, params, X_train_est, y_train_est)
                    if epoch % 5 == 0:
                        print(f'epoch: {epoch} - cost: {cost}')
                        #Fake time save
                        save_dir = bag_working_dir
                        os.makedirs(save_dir,  0o755,  exist_ok=True)
                        np.save(str(save_dir) + f"/fake_time_training_{i}_noise.npy",time.time())
                    
                    #Save epoch
                    save_dir_epoch = bag_working_dir + f"/estimator_{j}"
                    os.makedirs(save_dir_epoch,  0o755,  exist_ok=True)
                    with open(bag_working_dir + f'/estimator_{j}/starting_epoch_{i}_{j}.txt', 'w') as f:
                        f.write('%d' % epoch)
                    #Save params
                    np.save(bag_working_dir + f'/estimator_{j}/saved_params_{i}_{j}.npy',params)
                    with open(bag_working_dir + f'/estimator_{j}/opt_state_{i}_{j}.pickle', 'wb') as handle:
                        pickle.dump(opt_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                print()
        
                end_time_tr = time.time()-start_time_tr
                print('optimization time: ',end_time_tr)
                   
                # save training time
                save_dir = bag_working_dir + f"/estimator_{j}"
                os.makedirs(save_dir,  0o755,  exist_ok=True)
                np.save(str(save_dir) + f"/time_training_{i}_noise.npy",end_time_tr)
                
                # save parameters
                thetas = get_thetas(params)
                np.save(str(save_dir) + f"/thetas_{i}_noise.npy", thetas)
                
                ##### predict #####
                start_time_ts = time.time() 
                y_predict = qnn(X_test[:,random_estimator_features], params)
                y_predict_train = qnn(X_train[:,random_estimator_features], params)
                end_time_ts = time.time()-start_time_ts
                np.save(str(save_dir) + f"/time_test_{i}_noise.npy",end_time_ts)
                predictions.append(y_predict)
                predictions_train.append(y_predict_train)
                
                # save predictions
                np.save(str(save_dir) + f"/y_predict_{i}_noise.npy", y_predict)
                print(f'Error of bagging estimator {j} on test set: {mean_squared_error(y_test,y_predict)}\n')

                
        print(f'bagging - run {i} - bagging model {i}\n')
        
        # change save directory
        save_dir = bag_working_dir + f"/ensemble_model"
        os.makedirs(save_dir,  0o755,  exist_ok=True)
        
        # transform list of predictions into an array
        predictions_train = np.array(predictions_train)
        predictions = np.array(predictions)
        
        ##### predict train #####
        y_predict_train = jnp.mean(predictions_train,axis=0).reshape(-1,1)
        np.save(str(save_dir) + f"/y_predict_train_{i}_noise.npy", y_predict_train)
        
        ##### predict #####
        start_time_ts = time.time() 
        # compute average of estimators' predictions
        y_predict = jnp.mean(predictions,axis=0).reshape(-1,1)
        end_time_ts = time.time()-start_time_ts
        np.save(str(save_dir) + f"/time_test_{i}_noise.npy",end_time_ts)
        
        # save predictions
        np.save(str(save_dir) + f"/y_predict_{i}_noise.npy", y_predict)
        print(f'Error of bagging on test set: {mean_squared_error(y_test,y_predict)}\n')

def evaluate_full_model_predictor(qnn, optimizer, n_qubits, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, full_model_working_dir):   
    @jax.jit
    def calculate_mse_cost(X, y, theta):
        # yps=[]
        # for i in X:
            # yp = qnn(i, theta)
            #print('yp.shape',yp.shape)
            # yps.append(yp)
        # yps=jnp.array(yps)
        # print('yps.shape',yps.shape)
        
        yp = qnn(X,theta)
        cost = jnp.mean((yp - y) ** 2)

        #yps=jnp.array([])
        #for i in range(len(X)):
        #    yp=qnn(X[i],theta)
        #    yps=jnp.append(yps,yp)
        #cost = jnp.mean((yps - y) ** 2)
        return cost
    
    @jax.jit
    def optimizer_update(opt_state, params, x, y):
        loss, grads = jax.value_and_grad(lambda theta: calculate_mse_cost(x, y, theta))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

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
            start_time=time.time()
            params, opt_state, cost = optimizer_update(opt_state, params, X_train, y_train)
            end_time=time.time()
            print('time required',end_time-start_time)
            print(cost)
            if epoch % 5 == 0:
                print(f'epoch: {epoch} - cost: {cost}')
        print()

        end_time_tr = time.time()-start_time_tr
        print('optimization time: ',end_time_tr)
           
        # save training time
        save_dir = full_model_working_dir 
        os.makedirs(save_dir,  0o755,  exist_ok=True)
        np.save(str(save_dir) + f"/time_training_{i}_noise.npy",end_time_tr)
        
        # save parameters
        thetas = get_thetas(params)
        np.save(str(save_dir) + f"/thetas_{i}_noise.npy", thetas)
        
        ##### predict train #####
        y_predict_train = qnn(X_train, params)
        np.save(str(save_dir) + f"/y_predict_train_{i}_noise.npy", y_predict_train)
        
        ##### predict #####
        start_time_ts = time.time() 
        y_predict = qnn(X_test, params)
        end_time_ts = time.time()-start_time_ts
        np.save(str(save_dir) + f"/time_test_{i}_noise.npy",end_time_ts)
        
        # save predictions
        np.save(str(save_dir) + f"/y_predict_{i}_noise.npy", y_predict)
        print(f'Error of fullmodel on test set: {mean_squared_error(y_test,y_predict)}\n')
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
        #     ax.scatter(X_test_pca[:,0], X_test_pca[:,1], y_test)
        #     ax.scatter(X_test_pca[:,0], X_test_pca[:,1], y_predict)
        #     plt.show()
        #     ax.view_init(3, -83)
        #     plt.savefig(str(save_dir) + "/full_model_pred.png")
        #     plt.close('all')
        
        #     from sklearn.linear_model import LinearRegression
        #     reg = LinearRegression().fit(X_train, y_train)
        #     print(f'score:{reg.score(X_test, y_test)}')
        #     print(f'predict reg: {mean_squared_error(y_test,reg.predict(X_test))}')
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     ax.scatter(X_test_pca[:,0], X_test_pca[:,1], y_test)
        #     ax.scatter(X_test_pca[:,0], X_test_pca[:,1], reg.predict(X_test))
        #     plt.show()
        #     ax.view_init(3, -83)
        #     plt.close('all')
            
        #     from sklearn.neural_network import MLPRegressor
        #     reg = MLPRegressor(hidden_layer_sizes=(5,10,5,), random_state=1, activation='identity', max_iter=500).fit(X_train, y_train)
        #     print(f'score:{reg.score(X_test, y_test)}')
        #     print(f'predict MLP: {mean_squared_error(y_test,reg.predict(X_test))}')
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection='3d')
        #     ax.scatter(X_test_pca[:,0], X_test_pca[:,1], y_test)
        #     ax.scatter(X_test_pca[:,0], X_test_pca[:,1], reg.predict(X_test))
        #     plt.show()
        #     ax.view_init(3, -83)
        #     plt.close('all')
            
            
            
            
# algorithm for boosting regressor was taken from here: https://dafriedman97.github.io/mlbook/content/c6/s2/boosting.html
def evaluate_adaboost_predictor(qnn, n_estimators, optimizer, n_qubits, runs, epochs, layers, varform, X_train, X_test, y_train, y_test, ada_working_dir):
        
    @jax.jit
    def calculate_mse_cost(X, y, theta):
        yp = qnn(X, theta)
        cost = jnp.mean((yp - y) ** 2)
        return cost
    
    @jax.jit
    def optimizer_update(opt_state, params, x, y):
        loss, grads = jax.value_and_grad(lambda theta: calculate_mse_cost(x, y, theta))(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    def weighted_median(values, weights):
        sorted_indices = values.argsort()
        values = values[sorted_indices]
        weights = weights[sorted_indices]
        weights_cumulative_sum = weights.cumsum()
        median_weight = np.argmax(weights_cumulative_sum >= sum(weights)/2)
        return values[median_weight]
    
    print('adaboost\n')
    for i in range(runs):
        
        print(f'adaboost - run {i}\n')
        
        # number of training samples
        N = X_train.shape[0]
        
        # array to gather estimators' predictions
        fitted_values = np.empty((N, n_estimators))
        
        # array to gather parameters
        estimators_params = []
        
        # betas
        betas = []
        
        # weights applied to training samples
        weights = np.repeat(1/N, N)
                
        for t in range(n_estimators):
            
            print(f'adaboost - run {i} - estimator {t}\n')
            
            # seed
            key = jax.random.PRNGKey(i+(t*10))
                
            random_estimator_samples = jax.random.choice(key, a=N, shape=(int(X_train.shape[0]),), p=weights)
            X_train_est = X_train[random_estimator_samples,:]
            y_train_est = y_train[random_estimator_samples]
            
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

            estimators_params.append(params)
            y_predict_train = qnn(X_train, params)
            fitted_values[:,t] = y_predict_train
            
            ## Calculate observation errors
            abs_errors_t = np.abs(y_train - y_predict_train)
            D_t = np.max(abs_errors_t)
            L_ts = abs_errors_t/D_t
            
            ## Calculate model error (and possibly break)
            Lbar_t = np.sum(weights*L_ts)
            if Lbar_t >= 0.5:
                n_estimators = t - 1
                fitted_values = fitted_values[:,:t-1]
                estimators_params = estimators_params[:t-1]
                break
            
            ## Calculate and record beta 
            beta_t = Lbar_t/(1 - Lbar_t)
            betas.append(beta_t)
            
            ## Reweight
            Z_t = np.sum(weights*beta_t**(1-L_ts))
            weights *= beta_t**(1-L_ts)/Z_t
            
            
    
            end_time_tr = time.time()-start_time_tr
            print('optimization time: ',end_time_tr)
               
            # save training time
            save_dir = ada_working_dir / f"estimator_{t}"
            os.makedirs(save_dir,  0o755,  exist_ok=True)
            np.save(str(save_dir) + f"/time_training_{i}_noise.npy",end_time_tr)
            
            # save parameters
            thetas = get_thetas(params)
            np.save(str(save_dir) + f"/thetas_{i}_noise.npy", thetas)

    
        ## Get median 
        model_weights = np.log(1/np.array(betas))
        y_train_hat = np.array([weighted_median(fitted_values[n], model_weights) for n in range(N)])
        
            
        print(f'adaboost - run {i} - boosting model {i}\n')
        
        # change save directory
        save_dir = ada_working_dir / "ensemble_model"
        os.makedirs(save_dir,  0o755,  exist_ok=True)
        
        ##### predict train #####
        N_train = len(X_train)
        fitted_values = np.empty((N_train, n_estimators))
        for t, params in enumerate(estimators_params):
            fitted_values[:,t] = qnn(X_train, params)
        y_predict_train = np.array([weighted_median(fitted_values[n], model_weights) for n in range(N_train)]) 
        
        np.save(str(save_dir) + f"/y_predict_train_{i}_noise.npy", y_predict_train)
        
        ##### predict #####
        start_time_ts = time.time() 
        
        N_test = len(X_test)
        fitted_values = np.empty((N_test, n_estimators))
        for t, params in enumerate(estimators_params):
            fitted_values[:,t] = qnn(X_test, params)
        y_predict = np.array([weighted_median(fitted_values[n], model_weights) for n in range(N_test)]) 
        
        end_time_ts = time.time()-start_time_ts
        np.save(str(save_dir) + f"/time_test_{i}_noise.npy",end_time_ts)
        
        # save predictions
        np.save(str(save_dir) + f"/y_predict_{i}_noise.npy", y_predict)
        print(f'Error of adaboost on test set: {mean_squared_error(y_test,y_predict)}\n')

