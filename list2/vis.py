import pandas as pd
import matplotlib.pyplot as plt

def plot_histograms(csvfile, name):
    df = pd.read_csv(csvfile)
    # Histogram
    plt.figure(figsize=(8,5))
    plt.hist(df['solution_length'], bins=range(df['solution_length'].min(), df['solution_length'].max()+2), edgecolor='black')
    plt.title(f'Histogram długości rozwiązania: {name}')
    plt.xlabel('Liczba ruchów do rozwiązania')
    plt.ylabel('Liczba przypadków')
    plt.grid(axis='y')
    plt.show()

    # Histogram
    plt.figure(figsize=(8,5))
    plt.hist(df['visited_states'], bins=40, edgecolor='black')
    plt.title(f'Histogram liczby odwiedzonych stanów: {name}')
    plt.xlabel('Odwiedzone stany')
    plt.ylabel('Liczba przypadków')
    plt.grid(axis='y')
    plt.show()

def plot_scatter(csvfile, name):
    df = pd.read_csv(csvfile)
    plt.figure(figsize=(8,6))
    plt.scatter(df['solution_length'], df['visited_states'], alpha=0.6)
    plt.title(f'Relacja: liczba stanów vs długość rozwiązania ({name})')
    plt.xlabel('Liczba ruchów do rozwiązania')
    plt.ylabel('Liczba odwiedzonych stanów')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plot_histograms('8h1.csv', 'Misplaced Tiles 3x3')
    plot_scatter('8h1.csv', 'Misplaced Tiles 3x3')

    plot_histograms('8h2.csv', 'Manhattan 3x3')
    plot_scatter('8h2.csv', 'Manhattan 3x3')

    plot_histograms('8h3.csv', 'Linear Conflicts 3x3')
    plot_scatter('8h3.csv', 'Linear Conflicts 3x3')

    plot_histograms('15h3.csv', 'Linear Conflicts 4x4')
    plot_scatter('15h3.csv', 'Linear Conflicts 4x4')
