import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np


def random_color(ind):
    """
    black = '#000000' blue = '#0000FF' green = '#008000' yellow = '#FFFF00'
    red = '#FF0000' cyan = '#00FFFF' magenta = '#FF00FF' lime = '#00FF00'
    gray = '#808080' maroon = '#800000' purple = '#800080' dark_slate_gray = '#2F4F4F'
    """

    colors = ['#000000', '#00FFFF', '#008000', '#FFFF00', '#FF0000', '#FF00FF',
              '#0000FF', '#00FF00', '#808080', '#800000', '#800080', '#2F4F4F']
    if ind < len(colors):
        return colors[ind]
    else:
        return [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]


def get_columns_string(number):
    columns_string = '['
    for num in range(number-1):
        random_colors.append(random_color(num))
        columns_string = columns_string + 'df.columns[' + str(num) + ']'
        if num < (number - 2):
            columns_string = columns_string + ','
    columns_string = columns_string + ']'
    return columns_string


file = 'Case_002_analysis.xlsx'
data = pd.ExcelFile(file)
df = data.parse(5)

df = df.dropna(axis=0, how='any')
no_columns = len(df.columns)

list_of_columns = []
for i in range(no_columns):
    if df[df.columns[i]].sum() == 0:
        list_of_columns.append(i)
no_columns = no_columns - len(list_of_columns)
random_colors = []

df = df.apply(pd.to_numeric, errors='ignore')
df = df.drop(df.columns[list_of_columns], axis=1)
df['x_ticks'] = df.index.strftime('%Y.%m')

title = 'Weighs of portfolio assets participation in the SSRFMPC algorithm in time'
ax = df[eval(get_columns_string(no_columns))].plot(kind='bar', legend=True, stacked=True, xticks=df.index,
                                                   rot=90, color=random_colors, ylim=(0, 1), grid=False,
                                                   title=title, align='edge')
ax.set_xticklabels(df.x_ticks)
ax.set_yticks([0.05*x for x in range(20)], minor=True)
ax.grid('on', which='minor', axis='y', color='0.65', linestyle=':')
handles, labels = ax.get_legend_handles_labels()
plt.legend(reversed(handles), reversed(labels), loc='center left', bbox_to_anchor=(1.09, 0.5))
plt.xlabel('Time')
plt.ylabel('Portfolio asset participation')


dod = df
dod['profit'][1:] = df['profit'][0:-1]
dod['profit'][0] = 100
print(dod['profit'])
ax2 = ax.twinx()
asd = ax2.plot([(x-0.3) for x in range(len(df['profit']))], dod['profit'],
               linestyle='-', marker='o', color='k', markersize=10, markerfacecolor='lightgrey')
plt.ylim(60, 140)
plt.legend(loc='center left', bbox_to_anchor=(1.09, 0.8))

plt.subplots_adjust(bottom=0.15, right=0.8, top=0.93, left=0.05)
plt.ylabel('Portfolio actual profit')
plt.show()
