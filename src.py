import pandas as pd

def get_data(file1 = "Dataset.xlsx", file2="CycleOfLife.xlsx"): 
    Dataset = pd.read_excel("Dataset.xlsx")
    CycleOfLife = pd.read_excel("CycleOfLife.xlsx")

    return Dataset, CycleOfLife 
