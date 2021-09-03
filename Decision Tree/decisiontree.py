import pandas as pd
import numpy as np
from pprint import pprint


all_data = pd.read_csv('diabetes.csv', names=['Pregnancies','Glucose‌','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome',])


def entropy(target_col):

    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def Information_Gain(data,split_attribute_name,target_name="Outcome"):

    total_entropy = entropy(data[target_name])

    vals,counts= np.unique(data[split_attribute_name],return_counts=True)

    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])

    Information_Gain = total_entropy - Weighted_Entropy
    # print("feature: " , split_attribute_name , "\n","Total H:" , total_entropy , "WeightedH :" , Weighted_Entropy , "InfomationGain" , Information_Gain)
    return Information_Gain

def build_tree(data,originaldata,features,target="Outcome",parent = None):


    if len(np.unique(data[target])) <= 1:
        return np.unique(data[target])[0]

    elif len(data)==0:
        return np.unique(originaldata[target])[np.argmax(np.unique(originaldata[target],return_counts=True)[1])]


    elif len(features) ==0:
        return parent


    else:
        parent = np.unique(data[target])[np.argmax(np.unique(data[target],return_counts=True)[1])]

        item_values = [Information_Gain(data,feature,target) for feature in features]
        next_features_index = np.argmax(item_values)
        next_features = features[next_features_index]

        tree = {next_features:{}}


        features = [i for i in features if i != next_features]

        for value in np.unique(data[next_features]):


            value = value

            sub_data = data.where(data[next_features] == value).dropna()

            subtree = build_tree(sub_data,all_data,features,target,parent)

            tree[next_features][value] = subtree


        return(tree)



def Estimate(query,tree,default = 1):

    for key in list(query.keys()):
        if key in list(tree.keys()):

            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]

            if isinstance(result,dict):
                return Estimate(query,result)
            else:
                return result

def data_division(all_data):
    training_data = all_data.iloc[1:20].reset_index(drop=True)
    testing_data = all_data.iloc[20:].reset_index(drop=True)
    return training_data,testing_data


def testing_tree(data,tree):

    queries = data.iloc[:,:-1].to_dict(orient = "records")

    Estimateed = pd.DataFrame(columns=["Estimated"])

    for i in range(len(data)):
        Estimateed.loc[i,"Estimated"] = Estimate(queries[i],tree,1.0)

    print('accurancy : ',(np.sum(Estimateed["Estimated"] == data["Outcome"])/len(data))*100,'%')



def discretization(all_data):

    for att in  all_data.columns[:-1]:
        m =max(np.unique(all_data[att]).astype(float))
        n =min(np.unique(all_data[att]).astype(float))
        d = m-n
        for x in all_data[att]:
            if float(x) < n:
               index = all_data[all_data[att]== x].index.values
               all_data.loc[index[0]].at[att] ='TooLow'
            elif float(x) >= m:
                index = all_data[all_data[att]== x].index.values
                all_data.loc[index[0]].at[att]   = 'TooHigh'
            elif  float(x) >= n and float(x)<n+d/3 :

               index = all_data[all_data[att]== x].index.values
               all_data.loc[index[0]].at[att]  = 'Low'

            elif float(x) >= n+d/3 and float(x) < n+ 2*d/3:
                index = all_data[all_data[att]== x].index.values
                print(type(all_data.loc[index[0]].at[att]))

                all_data.loc[index[0]].at[att]  = 'Meduim'

            elif float(x) >= n+2*d/3 and float(x)<m:
                index = all_data[all_data[att]== x].index.values
                all_data.loc[index[0]].at[att]   = 'Hight'

    return all_data

def discretization2(all_data):

    for att in  all_data.columns[:-1]:
        m =max(np.unique(all_data[att]).astype(float))
        n =min(np.unique(all_data[att]).astype(float))
        d = m-n
        for x in all_data[att]:
            if float(x) < n:
               index = all_data[all_data[att]== x].index.values
               all_data.loc[index[0]].at[att] ='Low'
            elif float(x) >= m:
                index = all_data[all_data[att]== x].index.values
                all_data.loc[index[0]].at[att]   = 'High'
            elif  float(x) >= n and float(x)<n+d/2 :

               index = all_data[all_data[att]== x].index.values
               all_data.loc[index[0]].at[att]  = 'suspect'

            elif float(x) >= n+d/2 and float(x) < m:
                index = all_data[all_data[att]== x].index.values
                all_data.loc[index[0]].at[att]  = 'Meduim'


    return all_data



training_data = discretization(data_division(all_data)[0])
testing_data = discretization(data_division(all_data)[1])
tree = build_tree(training_data,training_data,training_data.columns[:-1])
# pprint( tree)
testing_tree(testing_data,tree)


# print("////////////////////SECOND TEST////////////////////")
# #Other discretization
# training_data2 = discretization2(data_division(all_data)[0])
# testing_data2 = discretization2(data_division(all_data)[1])
# tree2 = build_tree(training_data2,training_data2,training_data2.columns[:-1])
# pprint( tree2)
# testing_tree(testing_data2,tree2)
