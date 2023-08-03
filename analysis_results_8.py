import os
import numpy as np
import click
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from joblib import load

import matplotlib.style as style
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns
#sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks

style.use('seaborn-v0_8-paper') #sets the size of the charts (paper/talk/poster)
#plt.rc('font', size=24)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams.update({'lines.markeredgewidth': 1})

@click.group()
def main():
    pass

@main.command()
def run():   
    dataset='/diabete/diabete'
    #for i in range(1, 11):
    #    try:
    #        plot_first_experiment(directory_experiment=f'executions/jax/hardware_efficient/{i}/{dataset}',title=f"Comparison between Full model and Ensembles with {i} layers")
    #    except Exception:
    #        print(f'\n====== ERROR AT: {i} ======\n')
    #        pass
        
    #plot_error_layers_train(directory_experiment='executions/jax/hardware_efficient/',dataset=f'{dataset}',title="Training error of each model in terms of the number of layers")
    plot_error_layers(directory_experiment='executions/jax/hardware_efficient/',dataset=f'{dataset}',title="Error of each model in terms of the number of layers")


    #plot_first_experiment(directory_experiment='executions/jax/hardware_efficient/10/linear/n250_d05_e01_seed1001',title="Prova")
    #plot_first_experiment(directory_experiment='executions/jax/hardware_efficient/1/sin/n250_e01_seed1000',title="Provasin")
    #plot_first_experiment(directory_experiment='executions/jax/hardware_efficient/1/linear/n1000_d02_e01_seed1000',title="Prova1000")
    #plot_first_experiment(directory_experiment='executions/jax/hardware_efficient/1/linear/n250_d10_e01_seed3006',title="Prova_10")
    
@main.command()
def plotvariance():   
    datasets=['/linear/n250_d05_e01_seed1001','/concrete/concrete','/diabete/diabete']
    all_models = ['bagging_feature03_sample02', 'bagging_feature03_sample10', 'bagging_feature05_sample02', 'bagging_feature05_sample10', 'bagging_feature08_sample02', 'bagging_feature08_sample10']#,  'adaboost']
    for dataset in datasets:
        models = ['full_model']
        for i in range(0,len(all_models),2):
            tmp_models = []
            tmp_models.append(models[0])
            tmp_models.append(all_models[i])
            tmp_models.append(all_models[i+1])
            plot_layers_variance(directory_experiment=f'executions/jax/hardware_efficient/',dataset=dataset,models=tmp_models,title=f"Comparison between Full model and Ensembles")
        models.append('adaboost')
        plot_layers_variance(directory_experiment=f'executions/jax/hardware_efficient/',dataset=dataset,models=models,title=f"Comparison between Full model and Ensembles")

    #plot_first_experiment(directory_experiment='executions/jax/hardware_efficient/10/linear/n250_d05_e01_seed1001',title="Prova")
    
def plot_layers_variance(directory_experiment,dataset,models,title):
    plt.title(title)
    print(models)
    x_ticks = models#, 'bagging_feature05_sample02', 'bagging_feature05_sample10', 'bagging_feature08_sample02', 'bagging_feature08_sample10',  'adaboost']
    #markers = ['o', 'P', '8', 'x', 's', 'p']
    plt.xticks(ticks=range(1,11))
    # Plot of different ensemble models + baseline
    for i, model in enumerate(x_ticks):
        if model == 'full_model':
            model_errors_fm, model_std_fm = get_model_errors_std(directory_experiment,dataset,model)
            plt.fill_between(range(1,11),  model_errors_fm - model_std_fm, 
                            model_errors_fm + model_std_fm, 
                            alpha=0.2)
            plt.plot(range(1,11), model_errors_fm, label=f'Full Model')#Avg Loss 
        elif model == 'adaboost':
            model = model + "/ensemble_model"
            model_errors_adaboost, model_std_adaboost = get_model_errors_std(directory_experiment,dataset,model)
            plt.fill_between(range(1,11),  model_errors_adaboost - model_std_adaboost, 
                model_errors_adaboost + model_std_adaboost, 
                alpha=0.2)
            plt.plot(range(1,11), model_errors_adaboost, label=f'AdaBoost')#Avg Loss
        else:
            # Plot of bagging only
            model = model + "/ensemble_model"
            model_errors_bag, model_std_bag = get_model_errors_std(directory_experiment,dataset,model)
            plt.fill_between(range(1,11),  model_errors_bag - model_std_bag, 
                model_errors_bag + model_std_bag, 
                alpha=0.2)
            plt.plot(range(1,11), model_errors_bag, label=f'{models[i]}')#Avg Loss

    plt.xlabel('Layers')
    plt.ylabel('Average Mean Squared Error')   #Change with appropriate loss
    plt.tight_layout()
    plt.legend()#['Training Loss', 'Test Loss']
    save_dir = f"{directory_experiment}/plots_layersVariance/{dataset}/{models[1]}"
    os.makedirs(save_dir,  0o755,  exist_ok=True)
    plt.savefig(save_dir + f"/{title}.png",dpi=600)
    #plt.show()
    plt.close('all')
    
@main.command()
def histograms():   
    datasets=['/linear/n250_d05_e01_seed1001','/concrete/concrete','/diabete/diabete','wine/wine']
    plot_histogram_qubits(directory_experiment='executions/jax/hardware_efficient/',datasets=datasets,title="Number of qubits used in each experiment")
    plot_histogram_parameters(directory_experiment='executions/jax/hardware_efficient/',datasets=datasets,title="Number of parameters used in each experiment")

    #plot_first_experiment(directory_experiment='executions/jax/hardware_efficient/10/linear/n250_d05_e01_seed1001',title="Prova")

def plot_histogram_qubits(directory_experiment,datasets,title="Number of qubits used in each experiment"):
    # plt.title(title)

    x_ticks = ['full_model', 'bagging_feature03_sample02', 'bagging_feature03_sample10', 'bagging_feature05_sample02', 'bagging_feature05_sample10', 'bagging_feature08_sample02', 'bagging_feature08_sample10',  'adaboost']
    experiments = ['Experiment I','Experiment II','Experiment III','Experiment IV']

    x_pos = 3*np.arange(len(x_ticks))

    # Build the plot
    fig, ax = plt.subplots()
    
    qubits_exp_1 = [5,1,1,2,2,4,4,5]
    qubits_exp_2 = [8,2,2,4,4,6,6,8]
    qubits_exp_3 = [10,3,3,5,5,8,8,10]
    qubits_exp_4 = [13,3,3,6,6,10,10,13]

    ax.bar(x_pos, qubits_exp_1, align='center', color='C0', alpha=0.25, ecolor='black', capsize=10, width=0.4)
    ax.bar(x_pos+0.5, qubits_exp_2, align='center', color='C0', alpha=0.5, ecolor='black', capsize=10, width=0.4)
    ax.bar(x_pos+1, qubits_exp_3, align='center', color='C0', alpha=0.75, ecolor='black', capsize=10, width=0.4)
    ax.bar(x_pos+1.5, qubits_exp_4, align='center', color='C0', alpha=1, ecolor='black', capsize=10, width=0.4)
    
    ax.set_ylabel('Number of qubits')
    ax.set_xticks(x_pos+0.75)
    ax.set_xticklabels(x_ticks, rotation=45)

    ax.legend(experiments)

    ax.set_title(title)
    #ax.yaxis.grid(True)
    
    #plt.xticks(ticks=range(len(x_ticks)), labels=x_ticks, rotation=-45)
    plt.tight_layout()
    #plt.grid()
    save_dir = f"{directory_experiment}/plots_histogram_qubits"
    os.makedirs(save_dir,  0o755,  exist_ok=True)
    plt.savefig(save_dir + f"/{title}.png",dpi=600)
    plt.show()
    plt.close('all')
    

def plot_histogram_parameters(directory_experiment,datasets,title="Number of qubits used in each experiment"):
    # plt.title(title)

    x_ticks = ['full_model', 'bagging_feature03_sample02', 'bagging_feature03_sample10', 'bagging_feature05_sample02', 'bagging_feature05_sample10', 'bagging_feature08_sample02', 'bagging_feature08_sample10',  'adaboost']
    experiments = ['Experiment I','Experiment II','Experiment III','Experiment IV']

    x_pos = 3*np.arange(len(x_ticks))

    # Build the plot
    fig, ax = plt.subplots()
    
    qubits_exp_1 = np.multiply(np.array([5,1,1,2,2,4,4,5]),3)
    qubits_exp_2 = np.multiply(np.array([8,2,2,4,4,6,6,8]),3)
    qubits_exp_3 = np.multiply(np.array([10,3,3,5,5,8,8,10]),3)
    qubits_exp_4 = np.multiply(np.array([13,3,3,6,6,10,10,13]),3)

    ax.bar(x_pos, qubits_exp_1, align='center', color='C0', alpha=0.25, ecolor='black', capsize=10, width=0.4)
    ax.bar(x_pos+0.5, qubits_exp_2, align='center', color='C0', alpha=0.5, ecolor='black', capsize=10, width=0.4)
    ax.bar(x_pos+1, qubits_exp_3, align='center', color='C0', alpha=0.75, ecolor='black', capsize=10, width=0.4)
    ax.bar(x_pos+1.5, qubits_exp_4, align='center', color='C0', alpha=1, ecolor='black', capsize=10, width=0.4)
    

    ax.set_ylabel(r'Number of parameters ($\times$ no. of layers $l$)')
    ax.set_xticks(x_pos+0.75)
    ax.set_xticklabels(x_ticks, rotation=45)

    ax.legend(experiments)

    ax.set_title(title)
    #ax.yaxis.grid(True)
    
    #plt.xticks(ticks=range(len(x_ticks)), labels=x_ticks, rotation=-45)
    plt.tight_layout()
    #plt.grid()
    save_dir = f"{directory_experiment}/plots_histogram_parameters"
    os.makedirs(save_dir,  0o755,  exist_ok=True)
    plt.savefig(save_dir + f"/{title}.png",dpi=600)
    plt.show()
    plt.close('all')


    
def get_experiment_data(directory_experiment):
    X_train = np.load(directory_experiment + "/X_train.npy")
    X_test = np.load(directory_experiment + "/X_test.npy")
    y_train = np.load(directory_experiment + "/y_train.npy")
    y_test = np.load(directory_experiment + "/y_test.npy")
    return X_train,y_train,X_test,y_test

def get_experiment_predictions(directory_experiment,model):
    predictions = [np.load(f'{directory_experiment}/{model}/y_predict_{i}.npy') for i in range(10)]
    return np.array(predictions)
    
def get_experiment_predictions_train(directory_experiment,model):
    predictions = [np.load(f'{directory_experiment}/{model}/y_predict_train_{i}.npy') for i in range(10)]
    return np.array(predictions)
    
def get_model_avg_error(directory_experiment,model):
    X_train,y_train,X_test,y_test = get_experiment_data(directory_experiment)
    predictions = get_experiment_predictions(directory_experiment,model)
    scaler = load(f'{directory_experiment}/scaler.pkl')

    errors = [mean_squared_error(scaler.inverse_transform(p.reshape(-1,1)),y_test) for p in predictions]
    
    model_mean = np.average(errors)
    model_std = np.std(errors)
    
    return model_mean, model_std
    
    
def get_model_avg_error_train(directory_experiment,model):
    X_train,y_train,X_test,y_test = get_experiment_data(directory_experiment)
    predictions = get_experiment_predictions_train(directory_experiment,model)
    scaler = load(f'{directory_experiment}/scaler.pkl')

    errors = [mean_squared_error(scaler.inverse_transform(p.reshape(-1,1)),y_train) for p in predictions]
    
    model_mean = np.average(errors)
    model_std = np.std(errors)
    
    return model_mean, model_std
    
    
def get_model_errors(directory_experiment,dataset,model):
    errors = []
    for i in range(1,11):
        try:
            model_mean, model_std = get_model_avg_error(directory_experiment+f'{i}/{dataset}',model)
            errors.append(model_mean)
        except Exception:
            print(f'\n====== ERROR AT: {i} ======\n')
            pass        
    return np.array(errors)
    
    
def get_model_errors_train(directory_experiment,dataset,model):
    errors = []
    for i in range(1,11):
        try:
            model_mean, model_std = get_model_avg_error_train(directory_experiment+f'{i}/{dataset}',model)
            errors.append(model_mean)
        except Exception:
            print(f'\n====== ERROR AT: {i} ======\n')
            pass        
    return np.array(errors)
    
    
def get_model_errors_std(directory_experiment,dataset,model):
    errors = []
    std = []
    for i in range(1,11):
        try:
            model_mean, model_std = get_model_avg_error(directory_experiment+f'{i}/{dataset}',model)
            errors.append(model_mean)
            std.append(model_std)
        except Exception:
            print(f'\n====== ERROR AT: {i} ======\n')
            pass        
    return np.array(errors), np.array(std)
    
    
def get_model_errors_std_train(directory_experiment,dataset,model):
    errors = []
    std = []
    for i in range(1,11):
        try:
            model_mean, model_std = get_model_avg_error_train(directory_experiment+f'{i}/{dataset}',model)
            errors.append(model_mean)
            std.append(model_std)
        except Exception:
            print(f'\n====== ERROR AT: {i} ======\n')
            pass        
    return np.array(errors), np.array(std)



# @main.command()
# @click.option('--directory-experiment', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
# @click.option('--title', type=str, required=True)
def plot_first_experiment(directory_experiment, title='Average MSE and std of the models'):

    plt.title(title)
    #plt.grid()

    x_ticks = ['full_model', 'bagging_feature03_sample02_8', 'bagging_feature03_sample10_8', 'bagging_feature05_sample02_8', 'bagging_feature05_sample10_8', 'bagging_feature08_sample02_8', 'bagging_feature08_sample10_8',  'adaboost_8']

    # Plot of different ensemble models + baseline
    for i, model in enumerate(x_ticks):
        if model == 'full_model':
            model_mean, model_std = get_model_avg_error(directory_experiment,model)
            print('full_model',model_mean,model_std)
            plt.bar(i, model_mean, yerr=model_std, align='center', alpha=0.5, ecolor='black', capsize=10)
            #plt.errorbar([i], y=[model_mean], yerr=[model_std], c='green', elinewidth=5.0)
            #plt.scatter([i], [model_mean], c='green', s=120.0, zorder=100)

        elif model == 'adaboost':
            model = model + "/ensemble_model"
            model_mean, model_std = get_model_avg_error(directory_experiment,model)
            print('adaboost',model_mean,model_std)
            plt.bar(i, model_mean, yerr=model_std, align='center', alpha=0.5, ecolor='black', capsize=10)
            #plt.errorbar([i], y=[model_mean], yerr=[model_std], c='green', elinewidth=5.0)
            #plt.scatter([i], [model_mean], c='green', s=120.0, zorder=100)
        else:
            # Plot of bagging only
            model = model + "/ensemble_model"
            model_mean, model_std = get_model_avg_error(directory_experiment,model)
            print('bagging',model_mean,model_std)       
            plt.bar(i, model_mean, yerr=model_std, align='center', alpha=0.5, ecolor='black', capsize=10)
            #plt.errorbar([i], y=[model_mean], yerr=[model_std], c='green', elinewidth=5.0)
            #plt.scatter([i], [model_mean], c='green', s=120.0, zorder=100)
            
#    plt.ylim((0, 0.5))
    plt.xlim((-0.6, len(x_ticks)-1+0.6))
    plt.ylabel('Avg MSE and std over 10 runs')
    plt.xlabel('ensemble model')
    plt.xticks(ticks=range(len(x_ticks)), labels=x_ticks, rotation=45)
    plt.tight_layout()
    save_dir = f"{directory_experiment}/plots_errorbars"
    os.makedirs(save_dir,  0o755,  exist_ok=True)
    plt.savefig(save_dir + f"/{title}.png",dpi=600)
    plt.close('all')
    
    # Plot of each ensemble model + its estimators
    estimators_ticks = [f'estimator_{i}' for i in range(10)]
    for i, model in enumerate(x_ticks):
        if (model not in ['full_model', 'adaboost']):
            print(model)
            # Plot of bagging + its estimators
            for j in range(10):
                # bagging estimator
                model_est = model + f"/estimator_{j}"
                model_mean, model_std = get_model_avg_error(directory_experiment,model_est)
                print(f'bagging estimator {j}',model_mean,model_std)       
                plt.bar(j, model_mean, yerr=model_std, align='center', alpha=0.5, ecolor='black', capsize=10)
                #plt.errorbar([j], y=[model_mean], yerr=[model_std], c='green', elinewidth=5.0)
                #plt.scatter([j], [model_mean], c='green', s=120.0, zorder=100)
            # bagging model
            model_bag = model + f"/ensemble_model"
            model_mean, model_std = get_model_avg_error(directory_experiment,model_bag)
            print('bagging',model_mean,model_std)       
            plt.bar(j+1, model_mean, yerr=model_std, align='center', alpha=0.5, ecolor='black', capsize=10)
            #plt.errorbar([j+1], y=[model_mean], yerr=[model_std], c='red', elinewidth=5.0)
            #plt.scatter([j+1], [model_mean], c='red', s=120.0, zorder=100)

            #plt.ylim((0, 0.5))
            plt.xlim((-0.6, len(estimators_ticks)+0.6))
            plt.ylabel('Avg MSE and std over 10 runs')
            plt.xlabel(model)
            estimators_ticks_tmp = estimators_ticks.copy()
            estimators_ticks_tmp.append('bagging')
            plt.xticks(ticks=range(len(estimators_ticks_tmp)), labels=estimators_ticks_tmp, rotation=45)
            plt.tight_layout()
            #plt.grid()
            save_dir = f"{directory_experiment}/plots_errorbars"
            os.makedirs(save_dir,  0o755,  exist_ok=True)
            plt.savefig(save_dir + f"/{title}_{model}.png",dpi=600)
            plt.close('all')
        
    # Plot of predictions for each ensemble model + baseline
    X_train,y_train,X_test,y_test = get_experiment_data(directory_experiment)
    for i, model in enumerate(x_ticks):
        if model == 'full_model':
            predictions = get_experiment_predictions(directory_experiment,model)
            scaler = load(open(f'{directory_experiment}/scaler.pkl', 'rb'))
            plt.scatter(X_test[:,0], y_test, label='Original points')
            plt.scatter(X_test[:,0], scaler.inverse_transform((predictions[0]).reshape(-1,1)), c='y', label='Predicted points')
            plt.legend()
            plt.grid()
            plt.savefig(f"{directory_experiment}/{model}/plot_predictions.png",dpi=600)
            plt.close('all')

        elif model == 'adaboost':
            model = model + f"/ensemble_model"
            predictions = get_experiment_predictions(directory_experiment,model)
            scaler = load(open(f'{directory_experiment}/scaler.pkl', 'rb'))
            plt.scatter(X_test[:,0], y_test, label='Original points')
            plt.scatter(X_test[:,0], scaler.inverse_transform((predictions[0]).reshape(-1,1)), c='y', label='Predicted points')
            plt.legend()
            plt.grid()
            plt.savefig(f"{directory_experiment}/{model}/plot_predictions.png",dpi=600)
            plt.close('all')
        else:
            # Plot of bagging only
            model = model + f"/ensemble_model"
            predictions = get_experiment_predictions(directory_experiment,model)
            scaler = load(open(f'{directory_experiment}/scaler.pkl', 'rb'))
            plt.scatter(X_test[:,0], y_test, label='Original points')
            plt.scatter(X_test[:,0], scaler.inverse_transform((predictions[0]).reshape(-1,1)), c='y', label='Predicted points')
            plt.legend()
            plt.grid()
            plt.savefig(f"{directory_experiment}/{model}/plot_predictions.png",dpi=600)
            plt.close('all')
            
            
def plot_error_layers(directory_experiment, dataset, title=f"Error of each model in terms of the number of layers"):

    plt.title(title)

    x_ticks = ['full_model', 'bagging_feature03_sample02_8', 'bagging_feature03_sample10_8', 'bagging_feature05_sample02_8', 'bagging_feature05_sample10_8', 'bagging_feature08_sample02_8', 'bagging_feature08_sample10_8',  'adaboost_8']
    markers = ['o', 'P', '8', 'x', 's', 'p']
    count = 0
    
    # Plot of different ensemble models + baseline
    for i, model in enumerate(x_ticks):
        if model == 'full_model':
            model_errors = get_model_errors(directory_experiment,dataset,model)
            print('full_model',model_errors)
            plt.plot(model_errors, marker="^")#, color='blue')

        elif model == 'adaboost_8':
            model = model + "/ensemble_model"
            model_errors = get_model_errors(directory_experiment,dataset,model)
            print('adaboost',model_errors)
            plt.plot(model_errors, marker="*")#, color='orangered')
        else:
            # Plot of bagging only
            model = model + "/ensemble_model"
            model_errors = get_model_errors(directory_experiment,dataset,model)
            model_errors_even = model_errors[[1,3,5,7,9]]
            print(model)
            for i in range(len(model_errors_even)):
                model_errors_even[i] = round(model_errors_even[i], 1)
                print(model_errors_even[i],'&', end=' ')
            print('\n')
            plt.plot(model_errors, marker=markers[count])
            count += 1
            
#    plt.ylim((0, 0.5))
    plt.ylabel('MSE')
    plt.xlabel('Layers')
    x_ticks = ['FM', 'Bag_0.3_0.2', 'Bag_0.3_1.0', 'Bag_0.5_0.2', 'Bag_0.5_1.0', 'Bag_0.8_0.2', 'Bag_0.8_1.0',  'AdaBoost']
    plt.legend(x_ticks, loc='upper center')
    plt.xticks(ticks=range(len(model_errors)),labels=np.arange(1, len(model_errors)+1,1))

    #plt.xticks(ticks=range(len(x_ticks)), labels=x_ticks, rotation=-45)
    plt.tight_layout()
    #plt.grid()
    save_dir = f"{directory_experiment}/plots_errors_layers/{dataset}"
    os.makedirs(save_dir,  0o755,  exist_ok=True)
    #plt.savefig(save_dir + f"/{title}.png",dpi=600)
    plt.close('all')

    
    
def plot_error_layers_train(directory_experiment, dataset, title=f"Training error of each model in terms of the number of layers"):

    plt.title(title)

    x_ticks = ['full_model', 'bagging_feature03_sample02_8', 'bagging_feature03_sample10_8', 'bagging_feature05_sample02_8', 'bagging_feature05_sample10_8', 'bagging_feature08_sample02_8', 'bagging_feature08_sample10_8',  'adaboost_8']
    markers = ['o', 'P', '8', 'x', 's', 'p']
    count = 0
    
    # Plot of different ensemble models + baseline
    for i, model in enumerate(x_ticks):
        if model == 'full_model':
            model_errors = get_model_errors_train(directory_experiment,dataset,model)
            print('full_model',model_errors)
            plt.plot(model_errors, marker="^")#, color='blue')

        elif model == 'adaboost_8':
            model = model + "/ensemble_model"
            model_errors = get_model_errors_train(directory_experiment,dataset,model)
            print('adaboost',model_errors)
            plt.plot(model_errors, marker="*")#, color='orangered')
        else:
            # Plot of bagging only
            model = model + "/ensemble_model"
            model_errors = get_model_errors_train(directory_experiment,dataset,model)
            print('bagging',model_errors)
            plt.plot(model_errors, marker=markers[count])
            count += 1
            
#    plt.ylim((0, 0.5))
    plt.ylabel('MSE')
    plt.xlabel('Layers')
    x_ticks = ['FM', 'Bag_0.3_0.2', 'Bag_0.3_1.0', 'Bag_0.5_0.2', 'Bag_0.5_1.0', 'Bag_0.8_0.2', 'Bag_0.8_1.0',  'AdaBoost']
    plt.legend(x_ticks, loc='upper center')
    plt.xticks(ticks=range(len(model_errors)),labels=np.arange(1, len(model_errors)+1,1))

    #plt.xticks(ticks=range(len(x_ticks)), labels=x_ticks, rotation=-45)
    plt.tight_layout()
    #plt.grid()
    save_dir = f"{directory_experiment}/plots_errors_layers/{dataset}"
    os.makedirs(save_dir,  0o755,  exist_ok=True)
    #plt.savefig(save_dir + f"/{title}_train.png",dpi=600)
    plt.close('all')
    

if __name__ == '__main__':
    main()