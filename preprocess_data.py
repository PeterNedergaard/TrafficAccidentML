import pandas as pd

pd.set_option('display.max_columns', None)


def preprocess():
    data_df = pd.read_csv('data/cleaned_data.csv')

    data_df_encoded = pd.get_dummies(data_df, dtype=int, columns=['Junction_Detail',
                                                                  'Junction_Control', 'Weather_Conditions',
                                                                  'Road_Surface_Conditions', 'Carriageway_Hazards',
                                                                  'Vehicle_Manoeuvre'])

    data_df_encoded['column_normalized'] = -1 + (
            (data_df_encoded['Speed_limit'] - data_df_encoded['Speed_limit'].min()) * 2) / (
                                            data_df_encoded['Speed_limit'].max() - data_df_encoded['Speed_limit'].min())

    data_df_encoded.drop(['Speed_limit', 'Accident_Index'], axis=1, inplace=True)

    # save the encoded data
    data_df_encoded.to_csv('data/encoded_data.csv', index=False)
