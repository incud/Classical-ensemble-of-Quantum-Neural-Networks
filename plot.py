import json

import click
import matplotlib.pyplot as plt
import numpy as np


@click.group()
def main():
    pass


def get_mean_std_label(score_vector, index, is_bagging, is_estimator):
    if is_bagging and is_estimator:
        values = np.array([scores['mse_bagging'][index]['bagging_estimators'] for scores in score_vector])
        label = f"bag d={score_vector[0]['max_features'][index]}, N={score_vector[0]['max_samples'][index]}"
        mean = np.average(np.average(values, axis=0))
        std = np.average(np.std(values, axis=0))
    if is_bagging and not is_estimator:
        values = np.array([scores['mse_bagging'][index]['bagging'] for scores in score_vector])
        label = f"bag d={score_vector[0]['max_features'][index]}, N={score_vector[0]['max_samples'][index]}"
        mean = np.average(values, axis=0)
        std = np.std(values, axis=0)
    if not is_bagging and is_estimator:
        mean_vec = [np.average(scores['mse_adaboost']['adaboost_estimators']) for scores in score_vector]
        std_vec = [np.std(scores['mse_adaboost']['adaboost_estimators']) for scores in score_vector]
        mean = np.average(mean_vec)
        std = np.average(std_vec)
        label = f"adaboost"
    if not is_bagging and not is_estimator:
        values = [scores['mse_adaboost']['adaboost'] for scores in score_vector]
        mean = np.average(values, axis=0)
        std = np.std(values, axis=0)
        label = f"adaboost"

    return mean, std, label


def plot_first_experiment(directory_experiment, title, score_vector):
    # label to plots
    x_ticks = []

    # create plot of the current bagging results
    scores = score_vector[0]
    N = len(scores['max_features'])
    for i in range(N):
        max_samples = scores['max_samples'][i]
        max_features = scores['max_features'][i]
        mse = scores['mse_bagging'][i]
        # print(max_samples, max_features, mse)
        plt.scatter([i] * len(mse['bagging_estimators']),
                    mse['bagging_estimators'],
                    color='orange', s=40)
        plt.scatter(i,
                    mse['bagging'],
                    color='red', s=45)
        x_ticks.append(f"bag s={max_samples}, f={max_features}")

    # create plot of the current boosting results
    mse = scores['mse_adaboost']
    plt.scatter([N] * len(mse['adaboost_estimators']),
                mse['adaboost_estimators'],
                color='blue', s=40)
    plt.scatter(N,
                mse['adaboost'],
                color='yellow', s=45)
    x_ticks.append("adaboost")

    plt.title(f'Testing accuracy of dataset {title}')
    plt.ylim((0, 0.5))
    plt.xticks(ticks=range(len(x_ticks)), labels=x_ticks, rotation=-45)
    plt.tight_layout()
    plt.savefig(f"{directory_experiment}/{title}.png")
    plt.close('all')


@main.command()
@click.option('--directory-experiment', type=click.Path(exists=True, dir_okay=True, file_okay=False), required=True)
@click.option('--title', type=str, required=True)
@click.option('--score', type=str, required=True, multiple=True)
def run_plot_first_experiment(directory_experiment, title, score):
    score_vector = []
    for the_name in score:
        score = json.load(open(f"{directory_experiment}/{the_name}.json"))
        score_vector.append(score)

    plt.title(title)

    x_ticks = []
    # bagging
    for i in range(6):
        mean, std, label = get_mean_std_label(score_vector, i, is_bagging=True, is_estimator=True)
        print(f"bagging i={i}, mean={mean}, std={std}")
        plt.scatter([i], [mean], c='green')
        plt.errorbar([i], y=[mean], yerr=[std], c='green')
        mean, std, _ = get_mean_std_label(score_vector, i, is_bagging=True, is_estimator=False)
        print(f"bagging i={i}, mean={mean}, std={std}")
        plt.scatter([i], [mean], c='red')
        plt.errorbar([i], y=[mean], yerr=[std], c='red')
        x_ticks.append(label)
    # boosting
    for i in range(1):
        mean, std, label = get_mean_std_label(score_vector, None, is_bagging=False, is_estimator=True)
        print(f"boosting i={i}, mean={mean}, std={std}")
        plt.scatter([6 + i], [mean], c='green')
        plt.errorbar([6 + i], y=[mean], yerr=[std], c='green')
        mean, std, _ = get_mean_std_label(score_vector, None, is_bagging=False, is_estimator=False)
        print(f"boosting i={i}, mean={mean}, std={std}")
        plt.scatter([6 + i], [mean], c='red')
        plt.errorbar([6 + i], y=[mean], yerr=[std], c='red')
        x_ticks.append(label)

    plt.ylim((0, 0.35))
    plt.xlim((-0.2, 6.2))
    plt.xticks(ticks=range(len(x_ticks)), labels=x_ticks, rotation=-90)
    plt.tight_layout()
    plt.savefig(f"{directory_experiment}/{title}.png")
    plt.close('all')


if __name__ == '__main__':
    main()
