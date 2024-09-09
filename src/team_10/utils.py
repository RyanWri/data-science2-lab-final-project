import numpy as np
import matplotlib.pyplot as plt


# A function that translates the contents of a column in a Dataframe according to a given dictionary
def translate_column(df, translations_dict, column_name):
    for i in range(len(df[column_name])):
        df.loc[i, column_name] = translations_dict[df[column_name][i]]


def plot_basic_histogram(val_to_plot, xlabel, ylabel, title):
    plt.figure(figsize=(15, 8))
    plt.bar(val_to_plot.index, val_to_plot.values, width=0.5, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(val_to_plot.index)
    plt.xticks(fontsize=11)
    plt.show()


def calculate_optimal_split(df, column_name, number_of_quartiles):
    quantiles = np.linspace(0, 100, number_of_quartiles + 1)
    quartiles = np.percentile(df[column_name], quantiles)

    bin_edges = [df[column_name].min()] + list(quartiles[1:-1]) + [
        df[column_name].max()]

    return bin_edges


def plot_multi_color_basic_histogram_for_optimal_split(df, column_name, bin_edges, colors, xlabel, ylabel, title):
    plt.figure(figsize=(8, 5))
    n, bins, patches = plt.hist(df[column_name], bins=bin_edges, edgecolor='black')

    for i, patch in enumerate(patches):
        patch.set_facecolor(colors[i % len(colors)])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=8)

    plt.show()


def print_optimal_groups(bin_edges, num_of_quartiles):
    for i in range(num_of_quartiles):
        print(f"Group {i + 1}: {int(bin_edges[i])} - {int(bin_edges[i + 1])}")
    print()
