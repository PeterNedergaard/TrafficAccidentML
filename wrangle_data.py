import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


def wrangle_data():
    acc_data = pd.read_csv('data/Accident_Information.csv', dtype={0: str}, encoding='latin1')
    veh_data = pd.read_csv('data/Vehicle_Information.csv', dtype={0: str}, encoding='latin1')

    merged_data = pd.merge(acc_data, veh_data, on='Accident_Index')

    data_cut = pd.DataFrame(merged_data[
        ['Accident_Index', 'Accident_Severity', 'Speed_limit', 'Junction_Detail', 'Junction_Control',
         'Weather_Conditions', 'Road_Surface_Conditions', 'Carriageway_Hazards', 'Vehicle_Manoeuvre']])

    data_cut.replace('Data missing or out of range', np.nan, inplace=True)
    data_cut_cleaned = data_cut.dropna(how='any')

    for column in data_cut_cleaned.columns:
        # Get unique values for the column
        unique_values = data_cut[column].unique()
        # Print the column name and its unique values
        print(f'Unique values in {column}:', unique_values)

    data_cut_cleaned.to_csv('data/cleaned_data.csv', index=False)
