import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

def calculate_A12(X, Y):
    m, n = len(X), len(Y)
    r = np.argsort(np.concatenate((X, Y)))
    ranks = np.empty_like(r)
    ranks[r] = np.arange(len(r))
    R1 = np.sum(ranks[:m])
    A12 = (R1 - m * (m + 1) / 2) / (m * n)
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
        X = df1[df1['TARGET_CLASS'] == cls]['Total_Time'].values
        Y = df2[df2['TARGET_CLASS'] == cls]['Total_Time'].values

        # Convert Total_Time to seconds
        X = X / 1000
        Y = Y / 1000

        if len(X) > 0 and len(Y) > 0:
            A12 = calculate_A12(X, Y)
            p_value = calculate_p_value(X, Y)
            mean_X = np.mean(X)
            mean_Y = np.mean(Y)
            std_X = np.std(X)
            std_Y = np.std(Y)
            diff = mean_X - mean_Y
            std_X_formatted = round(std_X, 3)
            std_Y_formatted = round(std_Y, 3)
            results.append((cls, mean_X, std_X_formatted, mean_Y, std_Y_formatted, diff, round(A12, 3), format_p_value(p_value)))
        else:
            results.append((cls, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))

    return results

def format_p_value(p_value):
    if p_value < 0.001:
        return "{:.2e}".format(p_value)
    else:
        return round(p_value, 3)

def format_time(value):
    if pd.isnull(value):
        return value
    int_part = int(value)
    if int_part >= 1000:
        int_part = int_part % 1000
    return f"{int_part}.{str(value).split('.')[1][:2]}s"

def main():
    file1 = 'Default_Version.csv'
    file2 = 'RL_Version.csv'
    results = process_files(file1, file2)


    df_results = pd.DataFrame(results, columns=['Class', 'Dynamosa', 'Dynamosa Std', 'RL-Dynamosa', 'RL-Dynamosa Std', 'Difference', 'Â12', 'p-value'])


    df_results['Dynamosa'] = df_results['Dynamosa'].map(format_time)
    df_results['RL-Dynamosa'] = df_results['RL-Dynamosa'].map(format_time)
    df_results['Difference'] = df_results['Difference'].map(format_time)


    df_results.to_csv('Data_time.csv', index=False)

if __name__ == "__main__":
    main()
