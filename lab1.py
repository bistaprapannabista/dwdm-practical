import pandas as pd
import numpy as np
from IPython.display import display_html

def discretization(data):
    print("Discretizing 'Clump Thickness' attributes of the breast cancer dataset visualizing distribution of attribute value")
    print(data['Clump Thickness'].value_counts(sort=False))
    print("For the equal width method, we can apply the cut() function to discretize the attribute into 4 bins of similar interview widths.")
    print("The value_counts() function can be used to determine the number of instances in each bin.")
    bins = pd.cut(data['Clump Thickness'], 4)
    print(bins.value_counts(sort=False))
    print("For the equal frequency method, the qcut() function can be used to partition the values into 4 bins such that each bin has nearly the same number of instances.")
    bins = pd.qcut(data['Clump Thickness'], 4)
    print(bins.value_counts(sort=False))

def sampling(data):
    print("Displaying the first five records of the table without sampling.")
    display_html(data.head())
    print("A sample of size 3 is randomly selected (without replacement) from the original data.")
    sample = data.sample(n=3)
    display_html(sample)
    print("Randomly select 1% of the data (without replacement) and display the selected samples.")
    sample = data.sample(frac=0.01, random_state=1)
    display_html(sample)
    print("A sampling with replacement to create a sample whose size is equal to 1% of the entire data.")
    sample = data.sample(frac=0.01, replace=True, random_state=1)
    display_html(sample)

def remove_duplicate(data):
    dups = data.duplicated()
    print(f"Number of duplicated rows = {dups.sum()}")
    print(f"Number of rows before discarding duplicates: {data.shape[0]}")
    
    # Displaying rows with index 11 and 28
    print("Rows with index 11 and 28:")
    print(data.loc[[11, 28]])
    
    data2 = data.drop_duplicates()
    print(f"Number of rows after discarding duplicates = {data2.shape[0]}")

def outlier(data):
    data2 = data.drop(['Class'], axis=1)
    data2["Bare Nuclei"] = pd.to_numeric(data2["Bare Nuclei"])
    Z = (data2 - data2.mean()) / data2.std()
    Z_slice = Z[20:25]  # Store the slice of Z for display
    print("Slice of Z from row 20 to 25:")
    print(Z_slice)
    
    print(f"Number of rows before discarding outliers = {Z.shape[0]}")
    
    # Filtering rows where all values are within the range [-3, 3]
    Z2 = Z[((Z > -3).sum(axis=1) == 9) & ((Z <= 3).sum(axis=1) == 9)]
    print(f"Number of rows after discarding outliers = {Z2.shape[0]}")

def remove_missing(data):
    print(f"Number of rows in original data: {data.shape[0]}")
    data = data.dropna()
    print(f"Number of rows after discarding missing values: {data.shape[0]}")

def replace_missing_value_by_median(data):
    data_copy = data.copy()  # Create a copy of the DataFrame
    data2 = data_copy['Bare Nuclei']
    print("Before replacing missing values:")
    print(data2[20:25])
    median_value = data2.median()  # Calculate the median
    data2 = data2.fillna(median_value)  # Replace missing values with median
    print("After replacing missing values by median:")
    print(data2[20:25])

def noise_handle(data):
    data = data.drop(['Sample code'], axis=1)
    
    data = data.replace('?', np.NaN)
    
    print(f"Number of instances = {data.shape[0]}")
    print(f"Number of attributes = {data.shape[1]}")
    print("Number of missing values: ")
    for col in data.columns:
        print('\t%s: %d' % (col, data[col].isna().sum()))
    
    print("To further preprocess, select an option:\n"
          "0. Exit\n"
          "1. Replace missing value by median\n"
          "2. Remove missing value\n"
          "3. Handle outlier\n"
          "4. Remove duplicate\n"
          "5. Sampling\n"
          "6. Discretization:")
    
    option = int(input())
    while option != 0:
        if option == 1:
            replace_missing_value_by_median(data) 
        elif option == 2:
            remove_missing(data)
        elif option == 3:
            outlier(data)
        elif option == 4:
            remove_duplicate(data)
        elif option == 5:
            sampling(data)
        elif option == 6:
            discretization(data)
        else:
            print("Enter correct choice")
        
        print("Select your option again:")
        option = int(input())

def view(data):
    print("First five rows of the dataframe:")
    display(data.head())
    print(f"Number of instances = {data.shape[0]}")
    print(f"Number of attributes = {data.shape[1]}")

def main():
    data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data")
    print("Number of columns in the DataFrame:", len(data.columns))
    print("Column names in the DataFrame:", data.columns)
    # Check the number of columns and adjust the list of column names accordingly
    # For example:
    column_names = ['Column_' + str(i) for i in range(len(data.columns))]
    data.columns = column_names
    print("Do you want to view data?")
    response = input().lower()
    if response == 'yes':
        view(data)
        
    print("Do you want to remove noise and further preprocess data?")
    response = input().lower()
    if response == 'yes':
        noise_handle(data)


main()
