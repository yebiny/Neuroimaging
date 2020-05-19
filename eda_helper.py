import os


import random
import seaborn as sns
import cv2
# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL

import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib
#import plotly.graph_objs as go
from IPython.display import Image, display
import joypy
import warnings
warnings.filterwarnings("ignore")


BASE_PATH = './'


print('Reading data...')
loading_data = pd.read_csv(f'{BASE_PATH}/data/loading.csv')
train_data = pd.read_csv(f'{BASE_PATH}/data/train_scores.csv')

display(train_data.head())
print("Shape of train_data :", train_data.shape)

display(loading_data.head())
print("Shape of loading_data :", loading_data.shape)


# checking missing data
total = train_data.isnull().sum().sort_values(ascending = False)
percent = (train_data.isnull().sum()/train_data.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_train_data.head())


total = loading_data.isnull().sum().sort_values(ascending = False)
percent = (loading_data.isnull().sum()/loading_data.isnull().count()*100).sort_values(ascending = False)
missing_loading_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_loading_data.head())


def plot_bar(df, feature, title='', show_percent = False, size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.barplot(np.round(df[feature].value_counts().index).astype(int), df[feature].value_counts().values, alpha=0.8, palette='Set2')

    plt.title(title)
    if show_percent:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(100*height/total),
                    ha="center", rotation=45) 
    plt.xlabel(feature, fontsize=12, )
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xticks(rotation=90)
    plt.show()

#plot_bar(train_data, 'age', 'age count and %age plot', show_percent=True, size=4)

temp_data =  loading_data.drop(['Id'], axis=1)
print(temp_data)
plt.figure(figsize = (20, 20))
sns.heatmap(temp_data.corr(), annot = True, cmap="RdYlGn")
plt.yticks(rotation=0) 
#plt.show()


# Draw Plot
targets= loading_data.columns[1:]
plt.figure(figsize=(16,10), dpi= 80)
fig, axes = joypy.joyplot(loading_data, column=list(targets), ylim='own', figsize=(14,10))

# Decoration
plt.title('Distribution of features IC_01 to IC_29', fontsize=22)
#plt.show()
