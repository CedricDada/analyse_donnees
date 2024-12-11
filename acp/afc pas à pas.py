import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd

def recup_donnees_tableau_contingence(path):
    df=pd.read_excel(path)
    
    