import pandas as pd

df = pd.read_csv('data2.csv')
df.columns =[c.lower() for c in df.columns]

from sqlalchemy import create_engine

url = 'postgresql://ergrgcvn:y5DUZFSXpR5NvRyQbcPBpcxm6WqBkNL-@drona.db.elephantsql.com/ergrgcvn'
engine = create_engine(url)

df.to_sql('data', engine)