# Label encoding
data_set.loc[data_set['v1'] == 'spam', 'v1'] = 0
data_set.loc[data_set['v1'] == 'ham', 'v1'] = 1
data_set['v1'] = data_set['v1'].astype(int)
