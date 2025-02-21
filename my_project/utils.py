import pandas as pd

def load_data(file_path):
    """
    Load Excel data ,the data with per MW demand, solar and wind generation profiles.
    """
    df = pd.read_excel(file_path)
    return df
