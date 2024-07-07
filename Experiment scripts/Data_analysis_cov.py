import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, rankdata
from sklearn.utils import resample


def calculate_A12_v2(X, Y):
    n1 = len(X)
    n2 = len(Y)
    rank_sum = np.sum(rankdata(np.concatenate((X, Y)))[:n1])
    A12 = (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n2)
    return A12


def calculate_p_value(X, Y):
    stat, p_value = mannwhitneyu(X, Y, alternative='two-sided')
    return p_value


def process_files(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    results = []

    classes = set(df1['TARGET_CLASS']).union(set(df2['TARGET_CLASS']))
    for cls in classes:
        X = df1[df1['TARGET_CLASS'] == cls]['Coverage'].values
        Y = df2[df2['TARGET_CLASS'] == cls]['Coverage'].values

        if len(X) > 0 and len(Y) > 0:
            min_samples = min(len(X), len(Y))
            X_resampled = resample(X, n_samples=min_samples, random_state=42)
            Y_resampled = resample(Y, n_samples=min_samples, random_state=42)

            A12_v2 = calculate_A12_v2(X_resampled, Y_resampled)
            p_value_v2 = calculate_p_value(X_resampled, Y_resampled)
            mean_X = np.mean(X) * 100 if len(X) > 0 else np.nan
            mean_Y = np.mean(Y) * 100 if len(Y) > 0 else np.nan
            std_X = np.std(X) * 100 if len(X) > 0 else np.nan
            std_Y = np.std(Y) * 100 if len(Y) > 0 else np.nan
            results.append((cls, round(A12_v2, 3), format_p_value(p_value_v2), mean_X, mean_Y, std_X, std_Y))
        else:
            results.append((cls, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))

    return results


def format_p_value(p_value):
    if p_value < 0.001:
        return "{:.2e}".format(p_value)
    else:
        return round(p_value, 3)


def main():
    file1 = 'Default_Version.csv'
    file2 = 'RL_Version.csv'
    results = process_files(file1, file2)


    df_results = pd.DataFrame(results, columns=['Class', 'Â12', 'p-value', 'Dynamosa', 'RL-Dynamosa', 'Dynamosa Std',
                                                'RL-Dynamosa Std'])


    df_results['Dynamosa'] = df_results['Dynamosa'].map(lambda x: f"{x:.3f}%" if not pd.isnull(x) else x)
    df_results['RL-Dynamosa'] = df_results['RL-Dynamosa'].map(lambda x: f"{x:.3f}%" if not pd.isnull(x) else x)


    df_results['Dynamosa Std'] = df_results['Dynamosa Std'].map(lambda x: f"{x:.3f}" if not pd.isnull(x) else x)
    df_results['RL-Dynamosa Std'] = df_results['RL-Dynamosa Std'].map(lambda x: f"{x:.3f}" if not pd.isnull(x) else x)


    df_results.to_csv('Data.csv', index=False)


if __name__ == "__main__":
    main()
