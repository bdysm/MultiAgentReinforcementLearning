import pandas as pd
import matplotlib.pyplot as plt
import random


def random_color(ind):
    """
    black = '#000000' blue = '#0000FF' green = '#008000' yellow = '#FFFF00'
    red = '#FF0000' cyan = '#00FFFF' magenta = '#FF00FF' lime = '#00FF00'
    gray = '#808080' maroon = '#800000' purple = '#800080' dark_slate_gray = '#2F4F4F'
    """

    colors = ['#000000', '#00FFFF', '#008000', '#808080', '#FF0000', '#FF00FF',
              '#0000FF', '#00FF00', '#FFFF00', '#800000', '#800080', '#2F4F4F']
    if ind < len(colors):
        return colors[ind]
    else:
        return [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

def get_columns_string(number):
    random_colors = []
    columns_string = '['
    for num in range(number):
        random_colors.append(random_color(num))
        columns_string = columns_string + 'dfo.columns[' + str(num) + ']'
        if num < (number - 1):
            columns_string = columns_string + ','
    columns_string = columns_string + ']'
    return (columns_string, random_colors)

file = 'Case_002_analysis.xlsx'
data = pd.ExcelFile(file)
dfo = pd.DataFrame()
print(len(data.sheet_names))
for i in range(len(data.sheet_names)):
    df = data.parse(i)

    if i > 0 and i<(len(data.sheet_names)-1):
        print(i)
        column_label = 'Portfolio#' + str(i)
    elif i == 0:
        print(i)
        column_label = 'Markovitz switch'
    else:
        print(i)
        column_label = 'SSWN Switch'
    dfo[column_label] = df['profit']
dfo = dfo.dropna(axis=0, how='any')
print(dfo)
no_columns = len(dfo.columns)
ax1 = dfo[dfo.columns[0]].plot(kind='line', color='gray', linestyle=':')
ax2 = dfo[dfo.columns[1]].plot(kind='line', color='green', linestyle='--')
ax3 = dfo[dfo.columns[2]].plot(kind='line', color='red', linestyle='--')
ax4 = dfo[dfo.columns[3]].plot(kind='line', color='blue', linestyle='--')
ax5 = dfo[dfo.columns[4]].plot(kind='line', color='purple', linestyle='--')
# ax6 = dfo[dfo.columns[5]].plot(kind='line', color='#2F4F4F', linestyle='--')
ax7 = dfo[dfo.columns[5]].plot(kind='line', color='black', linestyle='-',
                               title='Profit comparison for number of strategies')
plt.grid()
plt.legend()
plt.xlabel('Time')
plt.ylabel('Strategy profit')
plt.show()