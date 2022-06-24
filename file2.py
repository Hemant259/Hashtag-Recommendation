



# loading the shuffled dataset
import pandas as pd
df = pd.read_csv("datainsta_shuff.csv")

# loading the sentence embedding downloaded from previous file
training_set_embedding = pd.read_csv("embedding_train.csv")
test_set_embedding = pd.read_csv("embedding_test.csv")

#cleaning the embeddings
import numpy as np
test_set_embedding =test_set_embedding.to_numpy()
training_set_embedding =training_set_embedding.to_numpy()
training_set_embedding = np.delete(training_set_embedding, 0,1)
test_set_embedding = np.delete(test_set_embedding, 0,1)

##################################################################################################


# preprocessing the captions
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

def preprocess_text(sen):
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    sta = word_tokenize(sentence)
    st = re.compile(r'[-.,:;!?()|0-9]')
    tokens_without_sw = [word for word in sta if word not in stopwords.words('english')]
    post_punc = ""
    for words in tokens_without_sw:
      word=st.sub("", words)
      if len(word)>0:
        post_punc = post_punc + word + ' '
    return post_punc
import re
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


# cleaning the captions
import numpy as np
xx=list(df['TEXT'])
X=[]
for sen in xx:
    X.append(preprocess_text(sen))
tags=list(df['hashtags'].astype(str))
tag_freq={}
templ=[]
num1=np.array(df['number_of_tags'])
Y=[]
p=0


#making a dictionary for tag identification (tag_freq)
# storing the hashtags-id attached to a post in Y ()
for i in range(len(tags)):
  k=0
  temp=0
  for j in range(len(tags[i])):
    if tags[i][j]=='#':
      k=1
      temp=j+1
    elif tags[i][j]==',' or tags[i][j]==']':
      if temp<=j-1:
        if (tags[i][temp:j-1] in tag_freq):
          k=tag_freq.get(tags[i][temp:j-1])
          templ.append(k)
        else:
          tag_freq[tags[i][temp:j-1]]=p
          templ.append(p)
          p=p+1
        k=0
        
      else:
        num1[i]-=1
  
  
  Y.append(templ)
  templ=[]

#############################################################################################################


#splitting the data in sections
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0,shuffle= False)


from sentence_transformers import util

import tensorflow as tf
import tensorflow_text as text

# comparing the cosine similarity between sentence embeddings
A=util.pytorch_cos_sim(test_set_embedding, training_set_embedding)

#calculating bert tags
import numpy as np
bert_tag = []
for i in range(len(A)):
  w=max(A[i])
  x=np.where(A[i]==w)
  bert_tag.append(y_train[x[0][0]])

##################################################################################################################

## processing the data into usable forms for calculating AUFP
Users = df['USERNAME'].unique()
num_post_by_each_user = df['USERNAME'].value_counts()
cont_pop = np.array(df['social_popularity'])
T = len(tag_freq)
D = df.shape[0]
Mat=np.zeros((D,T))
for i in range(D):
  for j in Y[i]:
    Mat[i][j]=1

username = np.array(df['USERNAME'])
N=Users.size
User_pos=[]
for i in range(N):
  x = np.where(username==Users[i])
  User_pos.append(x[0])


User_pop = np.zeros(N)
for i in range(N):
  for j in range(len(User_pos[i])):
    User_pop[i]+=cont_pop[User_pos[i][j]]

User_tag=np.zeros((N, T))
for i in range(N):
  for j in range(len(User_pos[i])):
    for k in range(T):
      User_tag[i][k]+=Mat[User_pos[i][j]][k];
pp = np.copy(User_tag)
pp[pp!=0]=1
tot_user_pop = np.dot(pp.T, User_pop) 
num_tag_each_user=np.sum(User_tag,axis=1)


popsum_itag = np.dot(Mat.T, cont_pop)
num_tag_each_con = np.array(num1)

########################################################################################################

def get_key(val):
    for key, value in tag_freq.items():
         if val == value:
             return key
 
    return "key doesn't exist"

def AFP(cont_pop,popsum_itag,num_tag_each_con,Mat):
  n=np.multiply(Mat.T, cont_pop.T)
  Cp=np.divide(n.T,popsum_itag).T
  Ct=np.divide(Mat.T,num_tag_each_con)
  afp=np.dot(Cp,Ct.T)
  return afp

def AUP(User_pop,tot_user_pop,User_tag,Mat,num_tag_each_user,pp):
  n=np.multiply(pp.T, User_pop.T)
  Up=np.divide(n.T,tot_user_pop).T
  Ut=np.divide(User_tag.T,num_tag_each_user) 
  aup=np.dot(Up,Ut.T)              
  return aup

def RUFP(rufp,aufp,p,alpha):
  q=np.multiply(alpha,(np.dot(aufp,rufp)))
  rufp=np.add(q,np.multiply(1-alpha,p))
  return rufp

def set_preference_vector(p, bert_tag, D):
  cnt=0
  for i in range(len(bert_tag)):
    for j in range(len(bert_tag[i])):
      p[bert_tag[i][j]][i]=1
      cnt=cnt+1
  return p,cnt

def setval(p,aa,T):
  for i in range(len(bert_tag)):
    for j in range(T):
      p[j][i]= aa
  return p


## setting up prefrence vector
p1=np.zeros((T,len(bert_tag)))
p0=np.ones((T,len(bert_tag)))
p1,q=set_preference_vector(p1, bert_tag, D)
aa = q/(len(bert_tag)*T)
p0=np.multiply(p0,aa)

## initailizing rufp
rufp1=np.ones((T,len(bert_tag)))
rufp0=np.ones((T,len(bert_tag)))
wufp=np.zeros((T,len(bert_tag)))
alpha=0.85

## calculating aufp
afp=AFP(cont_pop,popsum_itag,num_tag_each_con,Mat)
aup=AUP(User_pop,tot_user_pop,User_tag,Mat,num_tag_each_user,pp)
aufp=np.multiply(aup,afp)


## calculating wufp matrix
for i in range(10):
  rufp0=RUFP(rufp0,aufp,p0,alpha)
  rufp1=RUFP(rufp1,aufp,p1,alpha)
  wufp=np.subtract(rufp0,rufp1)
wufp=wufp.T

# function to get hashtags name from id
def get_key(val):
    for key, value in tag_freq.items():
         if val == value:
             return key
 
    return "key doesn't exist"

### finally calculating AUFP tags
aufp_tag=[]
for i in range(len(bert_tag)):
  ind = np.argpartition(wufp[i], 8)[:8]
  aufp_tag.append(ind)
  ind=[]
aufp_tag

#######################################################################################################
### EVALUATION PART STARTS
followers = list(df['FOLLOWERS'])
following = list(df['FOLLOWING'])

print(D,T)

## calculating input for training the data
one_hot = np.zeros((D-2000, T+2))
for i in range(D-2000):
  for j in Y[i+2000]:
    one_hot[i][j]=1
  one_hot[i][T]=followers[i+2000]
  one_hot[i][T+1]=following[i+2000]


## calculating input for testing the "trained model"
train_test= np.zeros((2000, T+2))
for i in range(2000):
  for j in Y[i]:
    train_test[i][j]=1
  train_test[i][T]=followers[i]
  train_test[i][T+1]=following[i]


## calculating input for testing the "our methodology"
test_data=np.zeros((len(X_test),T+2))
for i in range(len(X_test)):
  for j in aufp_tag[i]:
    test_data[i][j]=1
  test_data[i][T]=followers[len(X_train)+i]
  test_data[i][T+1]=following[len(X_train)+i]


#########################################################################################################

##applying SVR model

file1 = open("res1.txt", "w")
from sklearn.svm import LinearSVR
eps = 5
svr = LinearSVR(epsilon=eps, C=0.01, fit_intercept=True)
svr.fit(one_hot, cont_pop[2000:D])

file1.write("SVR:  \n")

#  predicting pop for testing methodology
predicted_pop = svr.predict(test_data)

# predicting pop to test the trained model
tt = svr.predict(train_test) 

## print results of tested methodology
cnt=0
for i in  range(len(predicted_pop)):
  if(predicted_pop[i]>cont_pop[i+len(X_train)]):
    cnt+=1
file1.write(str(cnt/len(predicted_pop)))
file1.write("\nResults   :\n")

## print results of tested methodology
for i in range(10):
  file1.writelines([str(len(X_train)+i),  " \t"])
  file1.writelines([str(predicted_pop[i]), " \t"])
  file1.writelines([str(cont_pop[i+len(X_train)]),"\n"])
file1.write("\n\n\n\n")
file1.write("train accuracy\n")
for i in range(10):
  file1.writelines([str(tt[i]), " \t"])
  file1.writelines([str(cont_pop[i]),"\n"])

file1.write("\n\n\n\n")

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
tt= tt.tolist()
ww = cont_pop[0:2000].tolist()
tt.extend(ww)
tt=np.array(tt)
tt=tt.reshape(-1,1)
qq=scaler.fit_transform(tt)
from sklearn.metrics import mean_squared_error
rms = mean_squared_error(qq[2000:4000], qq[0:2000], squared=False)
file1.write("\n RMS : ")
file1.write(str(rms))

#########################################################################################################
#Applying XGBOOST model


file1.write("\n\n\nXGBOOST   :  \n")
import xgboost as xgb
model = xgb.XGBRegressor(n_estimators=1000, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=0.7,learning_rate = 0.01)
model.fit(one_hot,cont_pop[2000:D])
res = model.predict(test_data)

tt= model.predict(train_test)

for i in range(10):
  file1.writelines([str(len(X_train)+i),  " \t"])
  file1.writelines([str(res[i]), " \t"])
  file1.writelines([str(cont_pop[i+len(X_train)]),"\n"])

cnt=0
for i in range(len(res)):
  if(res[i]>cont_pop[i+len(X_train)]):
    cnt+=1
file1.writelines([str(cnt/len(res)),"\n"])

for i in range(10):
  file1.writelines([str(tt[i]), " \t"])
  file1.writelines([str(cont_pop[i]),"\n"])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
tt= tt.tolist()
ww = cont_pop[0:2000].tolist()
tt.extend(ww)
tt=np.array(tt)
tt=tt.reshape(-1,1)
qq=scaler.fit_transform(tt)
from sklearn.metrics import mean_squared_error
rms = mean_squared_error(qq[2000:4000], qq[0:2000], squared=False)
file1.write("\n RMS : ")
file1.write(str(rms))

#########################################################################################################
#Applying Linear regression model


file1.write("LinearRegression:  \n")
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(one_hot, cont_pop[2000:D])
predicted_pop = reg.predict(test_data)

train_test_predict = reg.predict(train_test)

cnt=0
for i in  range(len(predicted_pop)):
  if(predicted_pop[i]>cont_pop[i+len(X_train)]):
    cnt+=1
file1.write(str(cnt/len(predicted_pop)))
file1.write("\nResults   :\n")
for i in range(10):
  file1.writelines([str(len(X_train)+i),  " \t"])
  file1.writelines([str(predicted_pop[i]), " \t"])
  file1.writelines([str(cont_pop[i+len(X_train)]),"\n"])
file1.write("\n\n\n\n")
file1.write("train accuracy\n")
for i in range(10):
  file1.writelines([str(train_test_predict[i]), " \t"])
  file1.writelines([str(cont_pop[i]),"\n"])

file1.write("\n\n\n\n")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
tt= tt.tolist()
ww = cont_pop[0:2000].tolist()
tt.extend(ww)
tt=np.array(tt)
tt=tt.reshape(-1,1)
qq=scaler.fit_transform(tt)
from sklearn.metrics import mean_squared_error
rms = mean_squared_error(qq[2000:4000], qq[0:2000], squared=False)
file1.write("\n RMS : ")
file1.write(str(rms))


#########################################################################################################
#Applying gradient boosting model


file1.write("\n\n\Gradient Boosting   :  \n")
from sklearn.ensemble import GradientBoostingRegressor
sklearn_gbm = GradientBoostingRegressor(
    n_estimators=1000, 
    learning_rate=0.01, 
    max_depth=10
)
sklearn_gbm.fit(one_hot, cont_pop[2000:D])
res = sklearn_gbm.predict(test_data)

tt= model.predict(train_test)

for i in range(10):
  file1.writelines([str(len(X_train)+i),  " \t"])
  file1.writelines([str(res[i]), " \t"])
  file1.writelines([str(cont_pop[i+len(X_train)]),"\n"])

cnt=0
for i in range(len(res)):
  if(res[i]>cont_pop[i+len(X_train)]):
    cnt+=1
file1.writelines([str(cnt/len(res)),"\n"])

for i in range(10):
  file1.writelines([str(tt[i]), " \t"])
  file1.writelines([str(cont_pop[i]),"\n"])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
tt= tt.tolist()
ww = cont_pop[0:2000].tolist()
tt.extend(ww)
tt=np.array(tt)
tt=tt.reshape(-1,1)
qq=scaler.fit_transform(tt)
from sklearn.metrics import mean_squared_error
rms = mean_squared_error(qq[2000:4000], qq[0:2000], squared=False)
file1.write("\n RMS : ")
file1.write(str(rms))
