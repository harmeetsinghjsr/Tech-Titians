import pandas as pd

def read_csv_file(file_path):
    data = pd.read_csv(file_path)
    return data

file_path = r'C:\Users\hs978\Downloads\diabetes.csv'
data = read_csv_file(file_path)
print(data)
