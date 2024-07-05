import pandas as pd
from sklearn import preprocessing, tree

def printTree(clf, cols):
    text_representation = tree.export_text(clf, feature_names = cols, class_names = ['No Heart Disease', 'Heart Disease'], show_weights = True)
    print(text_representation)

def labelEncoder(df, colsList):
    le = preprocessing.LabelEncoder()
    for col in colsList:
        df[col] = le.fit_transform(df[col])

    return df

def oneHotEncoder(df, colsList):
    #Creates OneHotEncoder from sci-kit learn library
    one_hot_encoder = preprocessing.OneHotEncoder(sparse_output=False)

    #Applies OneHotEncoder to variable colsForOneHot
    encoded_data = one_hot_encoder.fit_transform(df[colsList])

    #Creates a new dataframe from the encoded data
    encoded_df = pd.DataFrame(
        encoded_data, columns=one_hot_encoder.get_feature_names_out(colsList))

    #Creates a new dataframe that combines the original dataframe with the encoded dataframe
    new_df = pd.concat([df.drop(colsList, axis=1), encoded_df], axis=1)

    return new_df