"""
After downloading & extracting datasets, 
we need to process it further before feeding it to the model
"""
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


#Loading Datasets
train_data_path =  "/content/train.csv"
test_data_path = "/content/test.csv"

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

#merging train and test datasets
if sorted(train_data.columns) == sorted(test_data.columns):
    dataframe = pd.concat([train_data,test_data], ignore_index=True)
    #by default the concat method is set to concat row wise by axis=0

#dropping unrelated and unnecessary columns that don't support label prediction
colums_to_drop = ['contact','loan','campaign','poutcome']
dataframe=dataframe.drop(colums_to_drop, axis=1)

##visualizing "unknown" values 
# unknown_counts = dataframe.apply(lambda x: x.eq("unknown").sum())
# unknown_counts

#filtering unknown values
condition = (dataframe['job'] != "unknown") 
filtered_dataframe = dataframe[condition]

#setting unknown values in education colum to secondary
filtered_dataframe.loc[filtered_dataframe['education'] == 'unknown', 'education'] = 'secondary'

#setting -1 values to 0 in pdays column
filtered_dataframe["pdays"] = filtered_dataframe["pdays"].replace(-1,0)
(filtered_dataframe["pdays"] == -1).any() #replaced -1 with 0

#if value less than the mean then to 0 else 1
filtered_dataframe.loc[filtered_dataframe["pdays"] < filtered_dataframe["pdays"].mean(), "pdays"] = 0
filtered_dataframe.loc[filtered_dataframe["pdays"] >= filtered_dataframe["pdays"].mean(), "pdays"] = 1

#normalizing age
filtered_dataframe = filtered_dataframe.copy()
filtered_dataframe['age'] = filtered_dataframe['age'] / 100


################### Encoding
numerical_columns = ["balance","day","previous"]
for column in numerical_columns:
  filtered_dataframe[column] = filtered_dataframe[column].apply(lambda x:x/float(filtered_dataframe[column].max()))

categorical_columns = ["job","marital","education","default","housing","month","y"]
for column in categorical_columns:
  mapping = {Type:label for label,Type in enumerate(filtered_dataframe[column].unique())}
  filtered_dataframe[column] = filtered_dataframe[column].map(mapping)

#################### Convert to PyTorch Tensors
feature_columns = ["age","job","marital","education","default","balance","housing",
                      "month","day","duration",
                      "pdays","previous"]
label_column = ["y"]
device = "cuda" if torch.cuda.is_available() else "cpu"

feature = filtered_dataframe[feature_columns].values
label = filtered_dataframe[label_column].values

torch_feature = torch.tensor(feature, dtype=torch.float32).to(device)
torch_label = torch.tensor(label, dtype=torch.float32).to(device)

#train_test_split

#combining feature and label dataset
dataset = TensorDataset(torch_feature,torch_label) #by default axis=1
train_size = int(0.8*len(dataset))
test_size = len(dataset)-train_size
train_set, test_set = random_split(dataset,[train_size,test_size])


