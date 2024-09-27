import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

THRESHOLD = 20
pr_data = pd.read_csv('res/pr.csv')

pr_error_aqo = pr_data['cardinality_error_with_aqo'].tolist()
pr_exe_time = pr_data['executions_with_aqo'].tolist()

for i in range(len(pr_error_aqo)):
    pr_error_aqo[i] = pr_error_aqo[i][1:-1].split(',')
    for j in range(len(pr_error_aqo[i])):
        pr_error_aqo[i][j] = float(pr_error_aqo[i][j])

x = range(1, 21)
y = []

for i in range(20):
    y.append(max(0, sum(pr_error_aqo[i][j] for j in range(len(pr_error_aqo[i])) if len(pr_error_aqo[i]) >= 20) / 20))

for test in pr_error_aqo:
    if len(test) >= 20:
        plt.plot(x, y, label='pr_error_aqo', linestyle='-')

plt.ylabel('Average Error')
plt.xlabel('Iterations')
plt.title('Online IPR Model')

plt.xticks(ticks=x)

plt.savefig('output_plot_pr.png')
plt.close()
