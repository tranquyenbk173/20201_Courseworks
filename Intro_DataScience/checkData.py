import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import random

df = pd.read_csv("dfBDS.csv")
#Quận
print(df['district'].unique())
#Phường
print(df['district'].unique())
#diện tích
print(df['square'].unique())

print("DONE")

