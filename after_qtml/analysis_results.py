import os
import numpy as np
import click
import pathlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from pickle import load

@click.group()
def main():
    pass

@main.command()
def run():
    plot_first_experiment(directory_experiment='executions/jax/hardware_efficient/1/linear/n250_d02_e01_seed1000',title="Prova")
    plot_first_experiment(directory_experiment='executions/jax/hardware_efficient/1/sin/n250_e01_seed1000',title="Provasin")
    #plot_first_experiment(directory_experiment='executions/jax/hardware_efficient/1/linear/n1000_d02_e01_seed1000',title="Prova1000")
    #plot_first_experiment(directory_experiment='executions/jax/hardware_efficient/1/linear/n250_d10_e01_seed3006',title="Prova_10")
    
    
def get_experiment_data(directory_experiment):
    X_train = np.load(directory_experiment + f"/X_train.npy")
    X_test = np.load(directory_experiment + f"/X_test.npy")
    y_train = np.load(directory_experiment + f"/y_train.npy")
    y_test = np.load(directory_experiment + f"/y_test.npy")
    return X_train,y_train,X_test,y_test

def get_experiment_predictions(directory_experiment,model):
    predictions = [np.load(f'{directory_experiment}/{model}/y_predict_{i}.npy') for i in range(10)]
    return np.array(predictions)
    
def get_model_avg_error(directory_experiment,model):
    X_train,y_train,X_test,y_test = get_experiment_data(directory_experiment)
    predictions = get_experiment_predictions(directory_experiment,model)
    
    scaler = load(open(f'{directory_experiment}/scaler.pkl', 'rb'))

    errors = [mean_squared_error(scaler.inverse_transform(p.reshape(-1,1)),y_test) for p in predictions]
    
    model_mean = np.average(errors)
    model_std = np.std(errors)
    
    return model_mean, model_std
    

    
# @main.command()
# @click.option('--directory-experiment', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
# @click.option('--title', type=str, required=True)
def plot_first_experiment(directory_experiment, title='Average MSE and std of the models'):

    plt.title(title)

    x_ticks = ['full_model', 'bagging_feature05_sample02', 'bagging_feature05_sample10', 'bagging_feature10_sample02', 'adaboost']

    # Plot of different ensemble models + baseline
    for i, model in enumerate(x_ticks):
        if model == 'full_model':
            model_mean, model_std = get_model_avg_error(directory_experiment,model)
            print('full_model',model_mean,model_std)
            plt.errorbar([i], y=[model_mean], yerr=[model_std], c='green', elinewidth=5.0)
            plt.scatter([i], [model_mean], c='green', s=120.0, zorder=100)

        elif model == 'adaboost':
            model = model + f"/ensemble_model"
            model_mean, model_std = get_model_avg_error(directory_experiment,model)
            print('adaboost',model_mean,model_std)
            plt.errorbar([i], y=[model_mean], yerr=[model_std], c='green', elinewidth=5.0)
            plt.scatter([i], [model_mean], c='green', s=120.0, zorder=100)
        else:
            # Plot of bagging only
            model = model + f"/ensemble_model"
            model_mean, model_std = get_model_avg_error(directory_experiment,model)
            print('bagging',model_mean,model_std)       
            plt.errorbar([i], y=[model_mean], yerr=[model_std], c='green', elinewidth=5.0)
            plt.scatter([i], [model_mean], c='green', s=120.0, zorder=100)
            
    plt.ylim((0, 0.5))
    plt.xlim((-0.2, 4.2))
    plt.ylabel('Avg MSE and std over 10 runs')
    plt.xlabel('ensemble model')
    plt.xticks(ticks=range(len(x_ticks)), labels=x_ticks, rotation=-45)
    plt.tight_layout()
    save_dir = f"{directory_experiment}/plots_errorbars"
    os.makedirs(save_dir,  0o755,  exist_ok=True)
    plt.savefig(save_dir + f"/{title}.png")
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
                plt.errorbar([j], y=[model_mean], yerr=[model_std], c='green', elinewidth=5.0)
                plt.scatter([j], [model_mean], c='green', s=120.0, zorder=100)
            # bagging model
            model_bag = model + f"/ensemble_model"
            model_mean, model_std = get_model_avg_error(directory_experiment,model_bag)
            print('bagging',model_mean,model_std)       
            plt.errorbar([j+1], y=[model_mean], yerr=[model_std], c='red', elinewidth=5.0)
            plt.scatter([j+1], [model_mean], c='red', s=120.0, zorder=100)

            plt.ylim((0, 0.5))
            plt.xlim((-0.2, 10.2))
            plt.ylabel('Avg MSE and std over 10 runs')
            plt.xlabel(model)
            estimators_ticks_tmp = estimators_ticks.copy()
            estimators_ticks_tmp.append('bagging')
            plt.xticks(ticks=range(len(estimators_ticks_tmp)), labels=estimators_ticks_tmp, rotation=-45)
            plt.tight_layout()
            save_dir = f"{directory_experiment}/plots_errorbars"
            os.makedirs(save_dir,  0o755,  exist_ok=True)
            plt.savefig(save_dir + f"/{title}_{model}.png")
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
            plt.savefig(f"{directory_experiment}/{model}/plot_predictions.png")
            plt.close('all')

        elif model == 'adaboost':
            model = model + f"/ensemble_model"
            predictions = get_experiment_predictions(directory_experiment,model)
            scaler = load(open(f'{directory_experiment}/scaler.pkl', 'rb'))
            plt.scatter(X_test[:,0], y_test, label='Original points')
            plt.scatter(X_test[:,0], scaler.inverse_transform((predictions[0]).reshape(-1,1)), c='y', label='Predicted points')
            plt.legend()
            plt.savefig(f"{directory_experiment}/{model}/plot_predictions.png")
            plt.close('all')
        else:
            # Plot of bagging only
            model = model + f"/ensemble_model"
            predictions = get_experiment_predictions(directory_experiment,model)
            scaler = load(open(f'{directory_experiment}/scaler.pkl', 'rb'))
            plt.scatter(X_test[:,0], y_test, label='Original points')
            plt.scatter(X_test[:,0], scaler.inverse_transform((predictions[0]).reshape(-1,1)), c='y', label='Predicted points')
            plt.legend()
            plt.savefig(f"{directory_experiment}/{model}/plot_predictions.png")
            plt.close('all')


if __name__ == '__main__':
    main()