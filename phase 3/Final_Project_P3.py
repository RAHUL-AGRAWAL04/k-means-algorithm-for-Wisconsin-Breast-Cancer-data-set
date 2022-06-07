import numpy as np
import pandas as pd
import random


col = ["Scn", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Class"]

df = pd.read_csv('breast-cancer-wisconsin.data', na_values = '?', names = col)

#Impute missing values in column A7

missing = ["A7"]

for i in missing:
    df.loc[df.loc[:,i].isnull(),i]=df.loc[:,i].mean()


def Init(d):
    df = d[['A2','A3','A4','A5','A6','A7','A8','A9','A10']]
    x = random.randint(0,698)
    y = random.randint(0,698)
    m2 = df.iloc[x, :]
    m4 = df.iloc[y, :]
    return(x, y, m2, m4, df)


#   Assign 
def Assign(m2, m4, df):
    C1 = pd.DataFrame()
    C2 = pd.DataFrame()
    for i in range(0, 699):
        l1 = 0
        l2 = 0
        
        for j in range(0,9):
            l1 = l1 + (df.iloc[i,j] - m2.iloc[j])**2
            
        m1dist = np.sqrt(l1)
            
        for j in range(0,9):
            l2 = l2 + (df.iloc[i,j] - m4.iloc[j])**2
            
        m2dist = np.sqrt(l2)
        
        #assign to cluster 1
        if (m1dist < m2dist):
            C1 = C1.append(df.iloc[i])
            
        #assign to cluster 2
        else:
            C2 = C2.append(df.iloc[i])
            
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
    CA.iloc[:,0] = df.iloc[:,0]          
    CA.iloc[:,1] = df.iloc[:,10]
        
    for i in range(0,699):
 
        if(i in C1.index):  
            CA.iloc[i , 2] = 2      # If index match for clsuter1, then class is "mu2"
        if(i in C2.index):  
            CA.iloc[i , 2] = 4      # If index match for clsuter2, then class is "mu4"    
           
    return(CA)

CA = pd.DataFrame(columns = ['ID', 'Class', 'Predicted Class'])
x, y, m2, m4, df1 = Init(df)

print(f'Randomly selected row {x} for centriod m2.')
print(m2)
print(f'\nRandomly selected row {y} for centriod m4.')
print(m4)

latest_C1 = pd.DataFrame()
latest_C2 = pd.DataFrame()
for i in range(50):
    C1, C2 = Assign(m2, m4, df1)
    m2, m4 = recompute(C1, C2)
    if C1.equals(latest_C1) and C2.equals(latest_C2):
        break
    else:
        latest_C1 = C1
        latest_C2 = C2
print(f'\nProgram ended after {i+1} iteration(s).')
print(f'\nFinal centroid m2:')
print(m2)
print(f'\nFinal centroid m4:')
print(m4)
print()

CA = clus(C1, C2, CA)
print(CA.head(21))


def error_B(CA):
    error_42=0
    pcalss_2=0
    for i in range(699):
      if(CA.iloc[i,2]==4 and CA.iloc[i,1]==2):     #Predicted vs  Actual
        error_42 = error_42 + 1
      if(CA.iloc[i,2]==2):                         # If total predicated=2  
        pcalss_2 = pcalss_2 + 1
     
    error_B = (error_42/pcalss_2)*100
    return(error_B)

def error_M(CA):
    error_24=0
    pcalss_4=0
    for i in range(699):
      if(CA.iloc[i,2]==2 and CA.iloc[i,1]==4):     #Predicted vs  Actual
        error_24=error_24+1
      if(CA.iloc[i,2]==4):                         # If total predicated=2  
        pcalss_4=pcalss_4+1
    error_M = (error_24/pcalss_4)*100
    return(error_M)

def error_total():
    error_all = 0
    class_all = 699
    for i in range(699):
      if(CA.iloc[i,2]==2 and CA.iloc[i,1]==4):     
        error_all = error_all + 1
      if(CA.iloc[i,2]==4 and CA.iloc[i,1]==2):     
        error_all = error_all + 1
    
    error_T = (error_all/class_all)*100
    return error_T

def swap(CA):
    for i in range(699):
        if CA.iloc[i,2] == 2 : 
            CA.iloc[i,2] = 4
        elif CA.iloc[i,2] == 4:
            CA.iloc[i,2] = 2
    return CA
err_B = error_B(CA)
err_M = error_M(CA)
TotalError = error_total()
print('\n\n')
print("Error B:",err_B,'%')
print("Error M",err_M,'%')
print("Total Error:",TotalError,'%')


if TotalError > 50:
    print('\n\nClusters are swapped as total_error > 50 % \nRecalculating the errors...\n')
    CA = swap(CA)
    error_B = error_B(CA)
    error_M = error_M(CA)
    TotalError = error_total()
    print("Error B:",error_B,'%')
    print("Error M",error_M,'%')
    print("Total Error:",TotalError,'%')
