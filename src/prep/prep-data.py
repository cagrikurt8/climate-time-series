# import libraries
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

def main(args):
    # read data
    df = get_data(args.input_data)

    feature_engineered_data = feature_engineer_data(df)

    cleaned_data = clean_data(feature_engineered_data)

    normalized_data = normalize_data(cleaned_data)

    output_df = normalized_data.to_csv((Path(args.output_data) / "climate-train-data.csv"))

# function that reads the data
def get_data(path):
    df = pd.read_csv(path, index_col='date')
    df.index = pd.to_datetime(df.index)
    

    # Count the rows and print the result
    row_count = (len(df))
    print('Preparing {} rows of data'.format(row_count))
    
    return df

# function that feature engineers the data
def feature_engineer_data(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.day_of_week
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    
    return df

# function that removes missing values
def clean_data(df):
    df = df.dropna()
    
    return df

# function that normalizes the data
def normalize_data(df):
    scaler = MinMaxScaler()
    num_cols = ["meantemp", "wind_speed", "meanpressure", "hour", "dayofweek", "quarter", "month", "year", "dayofyear"]
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_data", dest='input_data',
                        type=str)
    parser.add_argument("--output_data", dest='output_data',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")