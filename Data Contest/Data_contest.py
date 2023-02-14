#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries

import numpy as np
import os
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import statistics 
from statistics import mode 
from sklearn.cluster import KMeans
import csv
import sklearn
import numbers
import math


# Supporting Functions

def time_conversion(t1, t2):
    try:
        t1 = datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
    except:
        return np.nan
    value = abs(int((t2 - t1).total_seconds()))
    return value



# In[2]:


# Reading the csv Files

with open('Processed_data/bookings.csv','r') as f:
    bookings = pd.read_csv(
        f, dtype={"booking_id_2": str, "customer_id": str, "booking_status": str, "booking_create_timestamp": str, "booking_approved_at": str, "booking_checkin_customer_date": str})
    
with open('Processed_data/bookings_data.csv','r') as f:
    bookings_data = pd.read_csv(
        f, dtype={"booking_id": str, "booking_sequence_id": int, "hotel_id": str, "seller_agent_id": str, "booking_expiry_date": str, "price": float, "agent_fees" : float})
    
with open('Processed_data/customer_data.csv','r') as f:
    customer_data = pd.read_csv(
        f, dtype={"customer_id": str, "customer_unique_id": str, "country": str})
    
# with open('Processed_data/hotels_data.csv','r') as f:
#     hotels_data = pd.read_csv(
#         f, dtype={"hotel_id": str, "hotel_category": int, "hotel_name_length": int, "hotel_description_length": int, "hotel_photos_qty": int})
    

hotels_data = pd.read_csv('Processed_data/hotels_data.csv')
payments_data = pd.read_csv('Processed_data/payments_data.csv')
train_data = pd.read_csv('Processed_data/train_data.csv')
sample_submission = pd.read_csv('Processed_data/sample_submission_5.csv')


# In[3]:


# Examining csv files data - Contains missing data

#print(bookings.info()) 
#print(bookings_data.info()) 
#print(customer_data.info()) 
#print(hotels_data.info()) 
#print(payments_data.info()) 
#print(train_data.info()) 


# In[4]:


# Merging the data

merged_df = bookings.merge(bookings_data, left_on='booking_id', right_on='booking_id')
#print(merged_df.info()) # Contains missing data


#Adding column

for i in range(merged_df.shape[0]):

    #converting timestamps to int
    t1 = merged_df.at[i, 'booking_create_timestamp']
    t2 = merged_df.at[i, 'booking_approved_at']
    if t1 == np.nan or t2 == np.nan:
        merged_df.at[i, 'time1'] = np.nan
    else:
        merged_df.at[i, 'time1'] = time_conversion(t1, t2)

    t3 = merged_df.at[i, 'booking_checkin_customer_date']
    t4 = merged_df.at[i, 'booking_create_timestamp']
    if t3 == np.nan or t4 == np.nan:
        merged_df.at[i, 'time2'] = np.nan
    else:
        merged_df.at[i, 'time2'] = time_conversion(t3, t4)

    t5 = merged_df.at[i, 'booking_create_timestamp']
    t6 = merged_df.at[i, 'booking_expiry_date']
    if t5 == np.nan or t6 == np.nan:
        merged_df.at[i, 'time3'] = np.nan
    else:
        merged_df.at[i, 'time3'] = time_conversion(t5, t6)

        
mean_val = merged_df['time1'].mean(axis=0)
merged_df['time1'].replace(np.nan, mean_val, inplace=True)

mean_val = merged_df['time2'].mean(axis=0)
merged_df['time2'].replace(np.nan, mean_val, inplace=True)

mean_val = merged_df['time3'].mean(axis=0)
merged_df['time3'].replace(np.nan, mean_val, inplace=True)


merged_df = merged_df.merge(customer_data, left_on='customer_id', right_on='customer_id')
#print(merged_df.info()) # Contains missing data

merged_df = merged_df.merge(hotels_data, left_on='hotel_id', right_on='hotel_id')
#print(merged_df.info()) # Contains missing data

merged_df = merged_df.merge(payments_data, left_on='booking_id', right_on='booking_id')
#print(merged_df.info()) # Contains missing data



# In[5]:


# Handling Missing Data


booking_status_labels, levels = pd.factorize(merged_df.booking_status)
merged_df['booking_status'] = booking_status_labels

country_labels, levels = pd.factorize(merged_df.country)
merged_df['country'] = country_labels

seller_id_labels, levels = pd.factorize(merged_df.seller_agent_id)
merged_df['seller_agent_id'] = seller_id_labels

hotel_id_labels, levels = pd.factorize(merged_df.hotel_id)
merged_df['hotel_id'] = hotel_id_labels

customer_unique_id_labels, levels = pd.factorize(merged_df.customer_unique_id)
merged_df['customer_unique_id'] = customer_unique_id_labels

payment_type_labels, levels = pd.factorize(merged_df.payment_type)
merged_df['payment_type'] = payment_type_labels


missing_columns = ['hotel_category', 'hotel_name_length', 'hotel_description_length', 'hotel_photos_qty']

for i in range(len(missing_columns)):
    mean_val = merged_df[missing_columns[i]].mean(axis = 0)
    merged_df[missing_columns[i]].replace(np.nan, mean_val, inplace=True)

    
#print(merged_df.info()) # Missing data handled except booking_approved_at and booking_checkin_customer_date


# In[6]:


# Preparing Train Data

#print(train_data.shape)

train_data_unique = train_data.drop_duplicates(subset = 'booking_id', inplace = False)

#print(train_data_unique.shape)


train_data_merged = train_data_unique.merge(merged_df, how = 'inner', left_on='booking_id', right_on='booking_id')
#print(train_data_merged.info()) 

train_data_merged.sort_values(['booking_id', 'payment_installments', 'booking_sequence_id'], ascending = [False, False, False],inplace = True)
train_data_merged.drop_duplicates(subset = 'booking_id', inplace = True)
#print(train_data_merged.info()) 


# In[7]:


# Loading X_train and y_train

X_train = np.vstack((train_data_merged.booking_status, train_data_merged.time1, train_data_merged.time2, train_data_merged.time3, train_data_merged.hotel_id, train_data_merged.seller_agent_id, train_data_merged.price, train_data_merged.agent_fees, train_data_merged.customer_unique_id, train_data_merged.country, train_data_merged.hotel_category, train_data_merged.hotel_name_length, train_data_merged.hotel_description_length, train_data_merged.hotel_photos_qty, train_data_merged.payment_sequential, train_data_merged.payment_type, train_data_merged.payment_installments ))  
#X_train = np.vstack((train_data_merged.agent_fees, train_data_merged.price,train_data_merged.time1, train_data_merged.time2))  



X_train = np.transpose(X_train)
y_train = train_data_merged.rating_score

#print(X_train.shape)
#print(y_train.shape)
#print(train_data_merged.iloc[0])


# In[8]:


# Training Classifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)

#print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

RFclassifier = RandomForestClassifier(n_estimators=100)
# AdaBoostClassifier()
RFclassifier.fit(X_train, y_train)



# In[9]:


# Testing on Validation Data

y_pred = RFclassifier.predict(X_valid)

# #print(classification_report(X_train, y_pred))
# #print(confusion_matrix(y_train, y_pred))


from sklearn.metrics import accuracy_score
RFAcc = accuracy_score(y_pred, y_valid)
#print('Random Forest accuracy is: {:.2f}%'.format(RFAcc*100))

mse = sklearn.metrics.mean_squared_error(y_pred, y_valid)
#print("MSE on validation data = ", mse)


# In[10]:


# Preparing test data

test_data = sample_submission.copy(deep = True)
#print(test_data.info())
##print(test_data['rating_score'].nunique())

#print(test_data.shape)

test_data_merged = test_data.merge(merged_df, left_on='booking_id', right_on='booking_id')
#print(test_data_merged.info()) 


test_data_merged.sort_values(['booking_id', 'payment_installments', 'booking_sequence_id'], ascending = [False, False, False],inplace = True)
test_data_merged.drop_duplicates(subset = 'booking_id', inplace = True)

#print(test_data_merged.shape)


# In[11]:


# Loading X_test 

X_test = np.vstack((test_data_merged.booking_status, test_data_merged.time1, test_data_merged.time2, test_data_merged.time3, test_data_merged.hotel_id, test_data_merged.seller_agent_id, test_data_merged.price, test_data_merged.agent_fees, test_data_merged.customer_unique_id, test_data_merged.country, test_data_merged.hotel_category, test_data_merged.hotel_name_length, test_data_merged.hotel_description_length, test_data_merged.hotel_photos_qty, test_data_merged.payment_sequential, test_data_merged.payment_type, test_data_merged.payment_installments ))  

# X_test = np.vstack((test_data_merged.agent_fees, test_data_merged.price, test_data_merged.hotel_category, test_data_merged.hotel_photos_qty, test_data_merged.payment_sequential, test_data_merged.payment_installments, test_data_merged.booking_status, test_data_merged.country, test_data_merged.seller_agent_id, test_data_merged.time1, test_data_merged.time2))  
X_test = np.transpose(X_test)


#print(X_test.shape)


# In[12]:


# Predicting y_test

y_pred_test = RFclassifier.predict(X_test)
#print(y_pred_test.shape)



new_df = pd.DataFrame()
new_df['booking_id'] = test_data_merged['booking_id']
new_df['y_pred_test'] = y_pred_test

#print(new_df.info())
##print(train_data.info())
##print(test_data.head(5))



ans = test_data.merge(new_df, left_on = 'booking_id', right_on = 'booking_id', how='left')
#print(ans.info())
##print(ans.head(5))

mean_val = ans['y_pred_test'].mean(axis = 0)
ans['y_pred_test'].replace(np.nan, round(mean_val), inplace=True)

##print(ans.info())
ans.drop(['rating_score'], axis=1, inplace=True)
ans.rename(columns = {'y_pred_test':'rating_score'}, inplace = True)
#print(ans.info())


# In[13]:


# SANITY CHECK
all(ans['booking_id'] == sample_submission['booking_id'])


# In[14]:


# Writing to csv file

ans.to_csv('prediction.csv', index = False)


# In[ ]:





# In[15]:


### ROUGH

a = list(test_data['booking_id'].unique())
b = list(bookings['booking_id'].unique())
c = list(bookings_data['booking_id'].unique())
d = list(payments_data['booking_id'].unique())
e = list(train_data['booking_id'].unique())


dif1 = np.setdiff1d(b, c)

#print(len(dif1))
dif1


# In[ ]:





# In[ ]:




