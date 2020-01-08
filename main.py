# -*- coding: utf-8 -*-
#
# from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Activation

# import pyper
import pandas as pd
import numpy as np

import category_encoders as ce # data mining

"""
データ読み込み
"""
def read_data():
    print(f'Read data')
    train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')
    test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')
    train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
    specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')
    sample_submission_df = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
    return train_df, test_df, train_labels_df, specs_df, sample_submission_df

"""
データ整理
"""
def data_mining(df_session, list_cols):
    # Eoncodeしたい列をリストで指定。もちろん複数指定可能。
    # list_cols = ['device']

    # 序数をカテゴリに付与して変換
    ce_oe = ce.OrdinalEncoder(cols=list_cols, handle_unknown='impute')
    df_session_ce_ordinal = ce_oe.fit_transform(df_session)

    df_session_ce_ordinal.head()

def main():
    # data set
    train_df, test_df, train_labels_df, specs_df, sample_submission_df = read_data()

    # display for now data
    print(f"train_df shape            : {train_df.shape}")
    print(f"test_df shape             : {test_df.shape}")
    print(f"train_labels_df shape     : {train_labels_df.shape}")
    # print(f"specs_df shape            : {specs_df.shape}")             # rm
    # print(f"sample_submission_df shape: {sample_submission_df.shape}") # rm
    train_df.head()
    test_df.head()
    train_labels_df.head()
    # specs_df.head()
    # sample_submission_df.head()

    # data mining
    data_mining(train_df, ['event_id', 'installation_id', 'title', 'type', 'world'])


    model = Sequential()
    model.add(Dense(32, activation='rule', input_dim=100))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', matrics=['accuracy'])

if __name__ == "__main__":
    main()