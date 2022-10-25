import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_data():
    data = pd.read_csv('data/exact.csv')
    R_love = data['R']
    J_love = data['J']

    time_step = np.linspace(0., 1., 1000)

    plt.plot(time_step, R_love, 'r.', markersize=2, label='Romeo')
    plt.plot(time_step, J_love, 'g.', markersize=2, label='Juliet')
    plt.title('Dynamic of love between Romeo and Juliet :)')
    plt.legend(loc="upper right")
    plt.xlabel('Time (s)')
    plt.ylabel('Love')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    plot_data()
