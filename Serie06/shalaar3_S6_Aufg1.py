import numpy as np
import matplotlib.pyplot as plt

x_data = np.arange(0, 101, 10)
y_data = np.array([999.9, 999.7, 998.2, 995.7, 992.2, 988.1, 983.2, 977.8, 971.8, 965.3, 958.4])

n = np.size(x_data)
m = 3


def f1(x, power):
    return pow(x, power)


def plotter(id, data, titles, xlabel, ylabel):
    plt.figure(id)
    marker = 'o'
    linestyle = ''

    for i in range(0, len(data)):
        plt.plot(data[i][0], data[i][1], marker=marker, linestyle=linestyle, label=titles[i])
        marker = ''
        linestyle = '-'

    plt.grid()
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


A = np.zeros((n, m))
for i in range(0, n):
    A[i, 0:m] = np.array([f1(x_data[i], 2), f1(x_data[i], 1), f1(x_data[i], 0), ])

A_T = np.transpose(A)
lambada = np.linalg.solve(A_T @ A, A_T @ y_data)

Q, R = np.linalg.qr(A)
Q_T = np.transpose(Q)
lambada_qr = np.linalg.solve(R, Q_T @ y_data)

x_plot = np.array(x_data, copy=True)
y_plot = np.polyval(lambada, x_plot)
y_plot_qr = np.polyval(lambada_qr, x_plot)

# plotter(1, [[x_data, x_plot, x_plot], [y_data, y_plot, y_plot_qr]], ['Daten', 'ohne', 'mit'], 'Temperatur', 'Dichte')
plotter(1, [[x_data, y_data], [x_plot, y_plot], [x_plot, y_plot_qr]], ['Daten', 'ohne', 'mit'], 'Temperatur', 'Dichte')

print(f"Konditionszahl von A_T * A (2. Norm): {np.linalg.cond(A_T @ A, 2)}")
print(f"Konditionszahl von R (2. Norm): {np.linalg.cond(R, 2)}")
print(f"Delta der Konditionszahlen: {np.abs(lambada - lambada_qr)}")

lambada_poly = np.polyfit(x_data, y_data, 2)
y_plot_poly = np.polyval(lambada_poly, x_plot)

plotter(1, [[x_data, y_data], [x_plot, y_plot], [x_plot, y_plot_qr]], ['Daten', 'ohne', 'mit'], 'Temperatur', 'Dichte')


exit()