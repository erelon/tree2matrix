import pickle
import matplotlib.pyplot as plt

from scipy.stats import f_oneway, ttest_ind
import numpy as np


def visualize(our_r, tabnet_r, xgboost_r, ax, label):
    # Calculate means and standard deviations
    means = [our_r.mean(), tabnet_r.mean(), xgboost_r.mean()]
    stds = [our_r.std(), tabnet_r.std(), xgboost_r.std()]

    # Perform t-tests
    _, p_value_our_tabnet = ttest_ind(our_r, tabnet_r)
    _, p_value_our_xgboost = ttest_ind(our_r, xgboost_r)
    _, p_value_tabnet_xgboost = ttest_ind(tabnet_r, xgboost_r)

    # Create bar plot with error bars
    labels = ['HERMANN', 'TabNet', 'Xgboost']
    x_pos = range(len(labels))

    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10,
           color=[(0.2, 0.4, 0.6), (0.6, 0.4, 0.8), (0.8, 0.4, 0.6)])

    # Add asterisks for significant p-values
    font_size = 6
    if p_value_our_tabnet <= 0.05 and p_value_our_tabnet > 0.01:
        ax.text(0, means[0] + stds[0] + 0.01, '* (HERMANN vs TabNet)', ha='center', fontsize=font_size)
    if p_value_our_xgboost <= 0.05 and p_value_our_xgboost > 0.01:
        ax.text(0, means[0] + stds[0] + 0.033 + 0.01, '* (HERMANN vs Xgboost)', ha='center', fontsize=font_size)

    if p_value_our_tabnet <= 0.01 and p_value_our_tabnet > 0.001:
        ax.text(0, means[0] + stds[0] + 0.01, '** (HERMANN vs TabNet)', ha='center', fontsize=font_size)
    if p_value_our_xgboost <= 0.01 and p_value_our_xgboost > 0.001:
        ax.text(0, means[0] + stds[0] + 0.033 + 0.01, '** (HERMANN vs Xgboost)', ha='center', fontsize=font_size)

    if p_value_our_tabnet <= 0.001:
        ax.text(0, means[0] + stds[0] + 0.01, '*** (HERMANN vs TabNet)', ha='center', fontsize=font_size)
    if p_value_our_xgboost <= 0.001:
        ax.text(0, means[0] + stds[0] + 0.033 + 0.01, '*** (HERMANN vs Xgboost)', ha='center', fontsize=font_size)

    ax.set_ylabel("AVG of " + label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(f'{label}')
    ax.yaxis.grid(True)
    ax.set_ylim(0, 1)


if __name__ == '__main__':
    with open("results_model.pkl", "rb") as f:
        our_r = pickle.load(f)

    with open("results_tabnet.pkl", "rb") as f:
        tabnet_r = pickle.load(f)

    with open("results_xgboost.pkl", "rb") as f:
        xgboost_r = pickle.load(f)

    stat_acc = f_oneway(our_r[0], tabnet_r[0], xgboost_r[0])  # Accuracy
    stat_auc = f_oneway(our_r[1], tabnet_r[1], xgboost_r[1])  # AUC
    stat_f1 = f_oneway(our_r[2], tabnet_r[2], xgboost_r[2])  # F1

    print("ANOVA test results:")
    print(stat_acc)
    print(stat_auc)
    print(stat_f1)
    print()

    if stat_acc.pvalue < 0.05:
        print("There is a significant difference between the models for accuracy.")
        t_test_acc = ttest_ind(our_r[0], tabnet_r[0])
        print("T-test results: for accuracy, our model vs. tabnet")
        print(t_test_acc)
        print("T-test results: for accuracy, our model vs. xgboost")
        t_test_acc = ttest_ind(our_r[0], xgboost_r[0])
        print(t_test_acc)
        print()

    if stat_auc.pvalue < 0.05:
        print("There is a significant difference between the models for AUC.")
        t_test_auc = ttest_ind(our_r[1], tabnet_r[1])
        print("T-test results: for AUC, our model vs. tabnet")
        print(t_test_auc)
        print("T-test results: for AUC, our model vs. xgboost")
        t_test_auc = ttest_ind(our_r[1], xgboost_r[1])
        print(t_test_auc)
        print()

    if stat_f1.pvalue < 0.05:
        print("There is a significant difference between the models for F1.")
        t_test_f1 = ttest_ind(our_r[2], tabnet_r[2])
        print("T-test results: for F1, our model vs. tabnet")
        print(t_test_f1)
        print("T-test results: for F1, our model vs. xgboost")
        t_test_f1 = ttest_ind(our_r[2], xgboost_r[2])
        print(t_test_f1)

    fig, AX = plt.subplots(1, 3, figsize=(12, 5))
    visualize(np.array(our_r[0]), np.array(tabnet_r[0]), np.array(xgboost_r[0]), AX[0], "Accuracy")
    visualize(np.array(our_r[1]), np.array(tabnet_r[1]), np.array(xgboost_r[1]), AX[1], "AUC")
    visualize(np.array(our_r[2]), np.array(tabnet_r[2]), np.array(xgboost_r[2]), AX[2], "F1")
    # Save the figure and show
    plt.tight_layout()
    plt.savefig("results.png")
