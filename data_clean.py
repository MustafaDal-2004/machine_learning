#data cleaning code
import pandas as pd

def drop_mostly_empty_columns(data, threshold=0.5):
    missing_ratio = data.isnull().mean()
    to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    data = data.drop(columns=to_drop)
    return data

def get_user_selected_columns(data):
   
    for col in data.columns:
        print(f"- {col}")

    user_input = input("Enter comma-separated column names : ")

    selected_cols = [col.strip() for col in user_input.split(",") if col.strip() in data.columns]

    print(selected_cols)

    return selected_cols

def encode_categoricals(data):
    for col in data:
        if not pd.api.types.is_numeric_dtype(data[col]):
            print(f"ncoding categorical column: {col}")
            data[col] = data[col].astype('category').cat.codes
        else:
            print(f"Column {col} is numeric. No encoding needed.")
    return data

def clean_data(data):
    print(data[1:5])
    col_list = []
    print('remove id columns')
    col_list = get_user_selected_columns(data)

    data = drop_mostly_empty_columns(data)
    data = encode_categoricals(data)
    data = data.drop(columns = col_list)

    return data

def euclidean_distance(row1, row2):
    diffs = []
    for i in row1.index:
        if pd.notnull(row1[i]) and pd.notnull(row2[i]):
            diffs.append((row1[i] - row2[i]) ** 2)
    if diffs:
        return sum(diffs) ** 0.5
    else:
        return float('inf') 

def knn_impute(data, k=3):
    data = data.copy()
    
    for col in data.columns:
        if data[col].isnull().sum() == 0:
            continue 
        for idx in data[data[col].isnull()].index:
            target_row = data.loc[idx]
            candidates = data[data[col].notnull()].copy()
            candidates['distance'] = candidates.drop(columns=[col]).apply(
                lambda row: euclidean_distance(target_row.drop(labels=[col]), row), axis=1
            )
            nearest_k = candidates.nsmallest(k, 'distance')[col]
            imputed_value = nearest_k.mean()
            print(f" - Imputed index {idx} with value {imputed_value}")
            data.at[idx, col] = imputed_value

    return data


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

data = pd.read_csv('/home/mustafa/Music/titanic/train.csv')

data = clean_data(data)

data = knn_impute(data)

print(data)
