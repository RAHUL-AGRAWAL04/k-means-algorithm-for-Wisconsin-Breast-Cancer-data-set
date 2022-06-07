import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def avg(data):
    avg = data.mean(axis='index')
    avg = round(avg, 1)
    return avg


def med(data):
    median = data.median(axis='index')
    median = round(median,1)
    return median

def std(data):
    std = data.std(axis = 'index')
    std = round(std,1)
    return std

def var(data):
    var = data.var(axis='index')
    var = round(var,1)
    return var


col = ["Scn", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Class"]

df = pd.read_csv('breast-cancer-wisconsin.data', na_values = '?', names = col)

#Impute missing values in column A7

missing = ["A7"]

for i in missing:
    df.loc[df.loc[:,i].isnull(),i]=df.loc[:,i].mean()


#A2 
print("Attribute A2 ---------")
print("Mean: \t", avg(df["A2"]))
print("Median: \t", med(df["A2"]))
print("Variance: \t", var(df["A2"]))
print("Standard Deviation: \t", std(df["A2"]))

df["A2"].plot.hist(bins = 10, color = "blue", alpha =0.5)
plt.title('Histogram of Attribute A2')
plt.xlabel('Value of the attribute')
plt.ylabel('Number of data points')
plt.show()


#A3 
print("\nAttribute A3 ---------")
print("Mean: \t", avg(df["A3"]))
print("Median: \t", med(df["A3"]))
print("Variance: \t", var(df["A3"]))
print("Standard Deviation: \t", std(df["A3"]))

df["A3"].plot.hist(bins = 10, color = "blue", alpha =0.5)
plt.title('Histogram of Attribute A3')
plt.xlabel('Value of the attribute')
plt.ylabel('Number of data points')
plt.show()

#A4 
print("\nAttribute A4 ---------")
print("Mean: \t", avg(df["A4"]))
print("Median: \t", med(df["A4"]))
print("Variance: \t", var(df["A4"]))
print("Standard Deviation: \t", std(df["A4"]))

df["A4"].plot.hist(bins = 10, color = "blue", alpha =0.5)
plt.title('Histogram of Attribute A4')
plt.xlabel('Value of the attribute')
plt.ylabel('Number of data points')
plt.show()

#A5 
print("\nAttribute A5 ---------")
print("Mean: \t", avg(df["A5"]))
print("Median: \t", med(df["A5"]))
print("Variance: \t", var(df["A5"]))
print("Standard Deviation: \t", std(df["A5"]))

df["A5"].plot.hist(bins = 10, color = "blue", alpha =0.5)
plt.title('Histogram of Attribute A5')
plt.xlabel('Value of the attribute')
plt.ylabel('Number of data points')
plt.show()

#A6 
print("\nAttribute A6 ---------")
print("Mean: \t", avg(df["A6"]))
print("Median: \t", med(df["A6"]))
print("Variance: \t", var(df["A6"]))
print("Standard Deviation: \t", std(df["A6"]))

df["A6"].plot.hist(bins = 10, color = "blue", alpha =0.5)
plt.title('Histogram of Attribute A6')
plt.xlabel('Value of the attribute')
plt.ylabel('Number of data points')
plt.show()


#A7 
print("\nAttribute A7 ---------")
print("Mean: \t", avg(df["A7"]))
print("Median: \t", med(df["A7"]))
print("Variance: \t", var(df["A7"]))
print("Standard Deviation: \t", std(df["A7"]))

df["A7"].plot.hist(bins = 10, color = "blue", alpha =0.5)
plt.title('Histogram of Attribute A7')
plt.xlabel('Value of the attribute')
plt.ylabel('Number of data points')
plt.show()


#A8 
print("\nAttribute A8 ---------")
print("Mean: \t", avg(df["A8"]))
print("Median: \t", med(df["A8"]))
print("Variance: \t", var(df["A8"]))
print("Standard Deviation: \t", std(df["A8"]))

df["A8"].plot.hist(bins = 10, color = "blue", alpha =0.5)
plt.title('Histogram of Attribute A8')
plt.xlabel('Value of the attribute')
plt.ylabel('Number of data points')
plt.show()


#A9 
print("\nAttribute A9 ---------")
print("Mean: \t", avg(df["A9"]))
print("Median: \t", med(df["A9"]))
print("Variance: \t", var(df["A9"]))
print("Standard Deviation: \t", std(df["A9"]))

df["A9"].plot.hist(bins = 10, color = "blue", alpha =0.5)
plt.title('Histogram of Attribute A9')
plt.xlabel('Value of the attribute')
plt.ylabel('Number of data points')
plt.show()


#A10 
print("\nAttribute A10 ---------")
print("Mean: \t", avg(df["A10"]))
print("Median: \t", med(df["A10"]))
print("Variance: \t", var(df["A10"]))
print("Standard Deviation: \t", std(df["A10"]))

df["A10"].plot.hist(bins = 10, color = "blue", alpha =0.5)
plt.title('Histogram of Attribute A10')
plt.xlabel('Value of the attribute')
plt.ylabel('Number of data points')
plt.show()


def Init(d):
    df = d[d.columns.difference([0, 10])]
    x = random.randint(0,699)
    y = random.randint(0,699)
    m2 = df.iloc[(x - 1), :]
    m4 = df.iloc[(y - 1), :]
    return(m2, m4, df)


#   Assign 
def Assign(m2, m4, df):
    C1 = pd.DataFrame(columns=[1,2,3,4,5,6,7,8,9])
    C2 = pd.DataFrame(columns=[1,2,3,4,5,6,7,8,9])
    
    for i in range(0, 699):
        l1 = 0
        l2 = 0
        
        for j in range(1,9):
            l1 = l1 + (df.iloc[i,j] - m2.iloc[j-1])**2
            
            m1dist = np.sqrt(l1)
            
        for j in range(1,9):
            l2 = l2 + (df.iloc[i,j] - m4.iloc[j-1])**2
            
            m2dist = np.sqrt(l2)
        
        #assign to cluster 1
        if (m1dist <= m2dist):
            C1 = C1.append(df.iloc[i,:])
            
        #assign to cluster 2
        else:
            C2 = C2.append(df.iloc[i,:])
            
    return(C1, C2)

#   Recompute

def recompute(C1, C2):
    m2 = []
    m4 = []
    
    for j in range(0,9):
        mean1 = C1.iloc[:,j].mean()
        m2.append(round(mean1, 2))
    
    for j in range(0,9):
        mean1 = C2.iloc[:,j].mean()
        m4.append(round(mean1, 2))
    
    m2 = pd.Series(m2)
    m4 = pd.Series(m4)
    return(m2,m4)


def clus(C1, C2, CA):
    print("Final cluster assignment: ")
    CA.iloc[:,0]=df.iloc[:,0]          
    CA.iloc[:,1]=df.iloc[:,10]
        
    for i in range(0,699):
 
        if(i in C1.index):  
            CA.iloc[i , 2] = 2      # If index match for clsuter1, then class is "mu2"
        if(i in C2.index):  
            CA.iloc[i , 2] = 4      # If index match for clsuter2, then class is "mu4"    
           
    return(CA)

CA = pd.DataFrame(columns = ['Scn', 'Class', 'Predicted Class'])


