# Import csv 
import pandas as pd
import numpy as np

def load_data():
    file = 'kc_house_data.csv'

    df = pd.read_csv(file)

    x = df[['sqft_living', 'sqft_lot']].astype(float).to_numpy()
    y = pd.to_numeric(df['price'], errors='coerce').fillna(0).to_numpy()

    return x, y
