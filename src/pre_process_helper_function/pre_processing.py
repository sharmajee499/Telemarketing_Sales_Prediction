# ******************************************************************************
# IMPORTING THE LIBRARIES
# ******************************************************************************
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# ******************************************************************************
# PRE-PROCESSING CLASS
# ******************************************************************************


class pre_processing:
    def __init__(
        self,
        path=None,
        standard=True,
    ):

        path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'bank-additional-full.csv'))
        """
        Please provide the path as string or it will use the default path
        """
        # It will read the data as soon as the class is constructed
        self._path = path
        self._df = pd.read_csv(self._path, sep=";", na_values=["unknown"])
        self._edu_map = {
            "illiterate": 1,
            "basic.4y": 2,
            "basic.6y": 3,
            "basic.9y": 4,
            "high.school": 5,
            "university.degree": 6,
            "professional.course": 7,
        }
        self._one_hot_var = [
            "job",
            "marital",
            "housing",
            "loan",
            "contact",
            "month",
            "day_of_week",
            "poutcome",
            "y",
        ]
        self._num_var = [
            "age",
            "campaign",
            "previous",
            "emp.var.rate",
            "cons.price.idx",
            "cons.conf.idx",
            "euribor3m",
            "nr.employed",
        ]
        self._standard = standard

    # function to get the default path
    def get_path(self):
        return self._path

    # Getting the info of the df
    def data_info(self):
        return self._df.info()

    # Get the descriptive stat of the data_frame
    def data_desc_stat(self):
        return self._df.describe(include="all")

    # Getting the Null values info
    def data_null_info(self):
        return self._df.isna().sum()

    # cleans the pre_processing data
    def clean_df(self):

        # Remove the variables**************************************************
        var_rem = ["pdays", "default", "duration"]
        df = self._df
        df.drop(columns=var_rem, axis=1, inplace=True)

        # Null Values Removal***************************************************
        df.interpolate(method="pad", limit=2, inplace=True)
        df.dropna(inplace=True)

        # Ordinal Encoder*******************************************************
        edu_encode = df["education"].map(self._edu_map)
        edu_encode_df = pd.DataFrame(edu_encode, columns=["education"])
        edu_encode_df.reset_index(inplace=True, drop=True)

        # One Hot Encoding******************************************************
        oh_encode_df = pd.get_dummies(df[self._one_hot_var], drop_first=False)
        oh_encode_df.drop(["y_no"], axis=1, inplace=True)

        # drop the 'y_no' variable as it is redundent to 'y_yes'
        # oh_encode_df.drop(columns=["y_no"], inplace=True)

        # reseting the index of the oh_encode_df
        oh_encode_df.reset_index(inplace=True, drop=True)

        # Standardization of Numeric Variable***********************************

        # Subseting the df with num var
        df_num = self._df[self._num_var]
        df_num.reset_index(inplace=True, drop=True)

        # combining 'edu_encode_df' with numeric var for normalization
        df_merged = pd.concat([df_num, edu_encode_df], axis=1)

        # output the non-standard data-frame if needed
        if self._standard == False:
            df_final = pd.concat([df_merged, oh_encode_df], axis=1)
            return df_final

        # extracting column name for later purpose
        col_name = df_merged.columns

        std_num = StandardScaler().fit_transform(df_merged)
        std_df = pd.DataFrame(std_num, columns=col_name)

        # Combining the standardize data with one-hot
        df_final = pd.concat([std_df, oh_encode_df], axis=1)

        return df_final

    def no_encoded_df(self):

        # Remove the variables**************************************************
        var_rem = ["pdays", "default", "duration"]
        df = self._df
        df.drop(columns=var_rem, axis=1, inplace=True)

        # Null Values Removal***************************************************
        df.interpolate(method="pad", limit=2, inplace=True)
        df.dropna(inplace=True)
        return df


# ******************************************************************************
# HELPER FUNCTION - Transform the Standardized data
# ******************************************************************************

# Tranforming the standardized data to the original format
def reverse_transform_original(var_name, value):
    """
    This is transform back the standardized values to the original form.
    raw_df_path: path to the raw file
    var_name: variable name
    value: value that needed to be transformed back to original

    Returns: double number/integer
    """
    raw_df_path = "C:/Users/Sande/OneDrive - MNSCU/Telemarketing_Success_Paper/Code+Analytics/Telemarketing_Success_Paper/data/raw/bank-additional-full.csv"
    raw_df = pd.read_csv(raw_df_path, sep=";")

    if var_name == "education":
        # Ordinal Encoder: 'education'
        edu_map = {
            "illiterate": 1,
            "basic.4y": 2,
            "basic.6y": 3,
            "basic.9y": 4,
            "high.school": 5,
            "university.degree": 6,
            "professional.course": 7,
        }

        edu_encode = raw_df["education"].map(edu_map)
        edu_encode_df = pd.DataFrame(edu_encode, columns=["education"])

        rev_transform = round(
            np.std(edu_encode_df["education"]) * value
            + np.mean(edu_encode_df["education"]),
            2,
        )
        return rev_transform

    rev_transform = round(
        np.std(raw_df[var_name]) * value + np.mean(raw_df[var_name]), 2
    )

    return rev_transform


# ******************************************************************************
# HELPER FUNCTION - LIME TRANSFORMATION -- Extra
# ******************************************************************************
"""At first the 'LIME' is implemented on the standardized data but to get the 
correct value that is transformed back to the original. Basically we will change
the original lime plot from API to our original data plot. We are kust changing
the plot index"""


def lime_transform(exp):
    """Input the lime explaination object for a specific instance
    Output: The new dataframe with index and its repesctive value to make new
    lime plot."""

    # Extract the exp object result as list
    exp_lst = [i[0] for i in exp.as_list()]

    # Split the combined string into list of strings
    exp_lst_split = [i.split() for i in exp_lst]

    # Convert the string that represent float to float type

    # Initialize the empty list
    exp_lst_float = []

    for items in exp_lst_split:
        # Another empty list to store calculation
        lst_sub = []
        for i in items:
            # if the sub-string can be converted into float then try else...
            try:
                float(i)
                lst_sub.append(float(i))
            except ValueError:
                lst_sub.append(i)

        exp_lst_float.append(lst_sub)

    # variables with the standardized data
    trsf_var = [
        "age",
        "campaign",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]

    trsf_lst = []

    # global variable for storing info in later loop
    var1 = ""

    # This loop only change the variables that are standardized keeping all things same
    for items in exp_lst_float:
        for i in items:
            if i in trsf_var:
                var1 = i
        if var1 in items:
            trsf_sub_lst = []
            for i in items:
                if type(i) == float:
                    trsf_sub_lst.append(reverse_transform_original(var1, i))
                else:
                    trsf_sub_lst.append(i)
            trsf_lst.append(trsf_sub_lst)
        else:
            trsf_lst.append(items)

    # Converting the list into string
    trsf_lst = [[" ".join([str(i) for i in trsf_lst[j]])]
                for j in range(len(trsf_lst))]

    # converting to numpy array
    trsf_lst = np.array(trsf_lst)

    # Flatten the array
    trsf_lst = trsf_lst.flatten()

    # Extract the 'value' component of the lime output
    val = np.array([i[1] for i in exp.as_list()])

    # Combining the transformed name and value to a data-frame
    trsf_df = pd.DataFrame(val, index=trsf_lst, columns=["prob"])

    return trsf_df
