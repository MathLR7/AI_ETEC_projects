import pandas as pd

def get_data(file1 = "Dataset.xlsx", file2="CycleOfLife.xlsx"): 

    """ returns entire dataset raw data and cycle of life data """

    Dataset = pd.read_excel("Dataset.xlsx")
    CycleOfLife = pd.read_excel("CycleOfLife.xlsx")

    return Dataset, CycleOfLife 

def get_quantile(df: pd.DataFrame) -> pd.DataFrame: 

    """
    Gets Cycle of life data for each cell and label them as "good" or "bad"
    good = top 45% cycle of life
    bad = bottom 45% cycle of life
    middle 10% gets excluded
    """

    bottom_cutoff = df["Cycle of Life"].quantile(0.45)
    top_cutoff = df["Cycle of Life"].quantile(0.55)

    bottom_45 = df[df["Cycle of Life"] <= bottom_cutoff].copy()
    bottom_45["label"] = "bad"

    top_45 = df[df["Cycle of Life"] >= top_cutoff].copy()
    top_45["label"] = "good"

    labeled_df = pd.concat([top_45, bottom_45]).sort_values("Cycle of Life")
    return labeled_df