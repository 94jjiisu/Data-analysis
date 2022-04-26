import psycopg2
import pandas as pd
import sqlalchemy

def get_db():

    url = 'postgresql://ergrgcvn:y5DUZFSXpR5NvRyQbcPBpcxm6WqBkNL-@drona.db.elephantsql.com/ergrgcvn'
    engine = sqlalchemy.create_engine(url)

    connection = engine.connect()
    metadata = sqlalchemy.MetaData()

    df = pd.read_sql_table('data', engine)

    return df