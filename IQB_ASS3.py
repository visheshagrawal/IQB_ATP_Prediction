#import tensorflow as tf
import pandas as pd
from sklearn.svm import SVC 
import numpy as np

def convertStringtoMatrix(string): #This function will take a 7 lettered string and return the vector representation of 7*21.
    dictionary={"A":0,"a":0,"C":1,"c":1,"D":2,"d":2,"E":3,"e":3,"F":4,"f":4,"G":5,"g":5,"H":6,"h":6,"I":7,"i":7,"K":8,"k":8,"L":9,"l":9,"M":10,"m":10,"N":11,"n":11,"P":12,"p":12,"Q":13,"q":13,"R":14,"r":14,"S":15,"s":15,"T":16,"t":16,"V":17,"v":17,"W":18,"w":18,"X":20,"x":20,"Y":19,"y":19}
    n=len(string)
    #matrix_to_return=[[0 for i in range(7)] for j in range(21)]
    matrix_to_return=[[0 for i in range(17)] for j in range(21)] 
    for i in range(n):
        # if ord(string[i])>64 and ord(string[i])<96:
        #     matrix_to_return[dictionary[string[i]]][i]=1
        # elif ord(string[i])>96:
        #     matrix_to_return[dictionary[string[i]]][i]=1
        matrix_to_return[dictionary[string[i]]][i]=1
    return matrix_to_return

# print(convertStringtoMatrix("VNikTNP"))
def maindo(string,x,y): #x and y are arrays that we append to 
    #string="XXX"+string+"XXX"
    string="XXXXXXXX"+string+"XXXXXXXX"
    for i in range(len(string)-16):
        x.append(convertStringtoMatrix(string[i:i+17]))
        if ord(string[i+8])>64 and ord(string[i+8])<96:
            y.append(-1)
        else:
            y.append(1)
    return x,y

df_train=pd.read_csv("C://Users//Vishesh//Desktop//train.data")
#print(len(df_train.index))
#print(df_train.iloc[i,1])

# print(maindo())

x = [ [ [0 for cc1 in range(17)] for cc2 in range(21)]]
y=[]    
len(df_train.index)
for i in range(len(df_train.index)):
    c=maindo(df_train.iloc[i,1],[],[]) # This gives us a 3d array of seqlength*7*21, we only need to add these.
    #x=x+c[0]
    x=x+c[0]
    y=y+c[1]
x.pop(0)
x=np.array(x)
x=np.reshape(x,(len(x), -1))
y=np.array(y)

# print(x)
# print(y)

clf = SVC(kernel='linear')   
# fitting x samples and y classes 
clf.fit(x,y)

print("Training complete")

# ######PREDICTION

# # # f = open("C://Users//Vishesh//Desktop//test1.txt",r)
# # # print(f.read())
mylist=()
string=""
with open('test.txt') as f:
    mylist = [tuple( i.split(',')) for i in f]
mylist.pop(0)
for i in range(len(mylist)):
    string=string+mylist[i][1][0]

w=[]
z=[]
w=maindo(string,w,z)[0]
##z=maindo(string,w,z)[1]

w=np.reshape(w,(len(w),-1))

z=clf.predict(w)
z=np.array(z)

# print(w)
# print(z)

f=open('sample.txt','w')
f.write("ID,Lable"+"\n")
i=0
#print(z)
#print(mylist)
for ele in mylist:
    if str(z[i])=="1":
        f.write(ele[0]+",+"+str(z[i])+'\n')
    else:
        f.write(ele[0]+","+str(z[i])+'\n')
    i+=1
f.close()
# # read_file = pd.read_csv ("C://Users//Vishesh//Desktop//Sample.txt")
# # read_file.to_csv ("C://Users//Vishesh//Desktop//Sample.csv", index=None)
# df = pd.read_csv("C://Users//Vishesh//Desktop//Sample.txt",delimiter=',')
# df.to_csv("Sample.csv")
i=0
dataa=[]
for i in range(len(mylist)):
    if str(z[i])=="1":
        dataa.append([mylist[i][0]]+["+"+str(z[i])])
    else:
        dataa.append([mylist[i][0]]+[str(z[i])])
df = pd.DataFrame(dataa,columns=["ID", "Lable"])
df.to_csv('Sample.csv', index=False)
print("Testing finished")




    




