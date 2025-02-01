import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 


# There are many operations we can do to preprocessing the data:
    # 1. Replace missing data 
    # 2. Encoding the plain-text to numbers (Yes -> 1, No -> 0)
    # 3. Feature Scaling: reduce the range of values between different features such as(age, salary)
    
# 1. Read data using pandas 

dataset = pd.read_csv("Data.csv")

# 2. Extract features and targets 

X = dataset.iloc[:, :-1].values # all coulmns except last one. 
y = dataset.iloc[:, -1].values # last column only

# get number unique values in each row
dataset.nunique()

# print features, and targets 
print(X)
print(y)

# we now notice missing data 'nan', and the target is a
# plain text not numbers. 

# import SimpleImputer to dealing with missing data. 
from sklearn.impute import SimpleImputer

# replace each nan value with the mean value of thier column.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3]) # analyzes the data you provide and calculates the messary statistics.
X[:, 1:3] = imputer.transform(X[:, 1:3])# method uses the statistics computed during .fit() to replace the missing values


# print features after handling missing data 
print(X)


# Now we have a problem: The country column is a string but we work with numbers only how can we convert them?

# using Encoding approaches like OnHotEncoder()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
 
# There are many types of Econders: LableEncoder(), OnHotEncoder()
"""
    # what the next line do?
    1. transformers: is a list of tuples that specifies which columns to transform and how to transform them
    2. each tuple in the list needs (name_of_transformer, ObjectOfTransformer(), column_to_transform)
    3. remainder: we will apply onthe remainder column operation
    4. types of operations in remainder:
        
        1. passthrough: Includes the untransformed columns as-is
        2. drop       : Drops the columns that are not explicitly transformed. 
        3. transformer: Applies a specified transformer to the remaining columns
"""
column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X = np.array(column_transformer.fit_transform(X))

## See the column after encoding 
print(X)


# Change yes and no to 0 and 1 

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(y)


# 3. Split dataset into train and test. 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state= 42)
print(x_train) # Sparse Matrix
print(y_train) # Vector 


# 4. Feature and target scaling 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit_transform: is used for x_train to compute scaling parameters (mean, std) from the training data.
# transform: is used for x_test to apply the same scaling based on those parameters, ensuring consistency.

x_train = scaler.fit_transform(x_train[:, 3:])
x_test = scaler.transform(x_test[:, 3:])





