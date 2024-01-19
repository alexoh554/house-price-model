# Import csv 
import pandas as pd

def load_data():
    file = 'kc_house_data.csv'

    df = pd.read_csv(file)

    x = df.drop(columns=['price']).to_dict(orient='list')
    y = df['price'].to_numpy()

    return x, y
