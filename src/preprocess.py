from pandas.core.frame import DataFrame
from sklearn.preprocessing import MinMaxScaler

# Refactoring
def copyFeatureNamesFrom(df: DataFrame, label_name: str):
    return df.drop(columns=label_name).columns.copy()

# Scaling
def scaleTo(min, max, data):
    scaler = MinMaxScaler(feature_range=(min, max))
    return scaler.fit_transform(data)

def scaleColumnsFrom(base_df: DataFrame, df: DataFrame, label_column: str) -> DataFrame:
    scaled_df = df
    for column in df.columns:
        if column is not label_column:
            scaled_df[column] = scaleTo(base_df[column].min(),
            base_df[column].max(), 
            df[column].values.reshape(-1, 1))
    return scaled_df