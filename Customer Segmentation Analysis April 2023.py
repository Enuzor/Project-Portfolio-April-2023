#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Step1: Import the relevant libraries and download the dataset


# In[27]:


import pandas as pd
import numpy as np

#viz Libraries
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import seaborn as sns

#warnings
import warnings
warnings.filterwarnings("ignore")

#datetime
import datetime as dt

#StandardSccaler
from sklearn.preprocessing import StandardScaler

#KMeans
from sklearn.cluster import KMeans
#file directoryy
import os


# In[28]:


df = pd.read_csv("sales_data_sample.csv", encoding="unicode_escape")
df.head()


# In[29]:


df.info() #tis gives a glimpse of the dataset


# In[30]:


df.count() #a summary of the data count


# In[ ]:


#downloaded data is cleaned and duplicate or irrelvant data is taken off


# In[31]:


to_drop = ['PHONE','ADDRESSLINE1','ADDRESSLINE2','STATE','POSTALCODE']
df = df.drop(to_drop, axis=1)


# In[32]:


df.isnull().sum()


# In[ ]:


#From the above we can see that the territory column has a lot of null values, thi can be ignored as it will not impact our analysis. 


# In[ ]:


#Step 2: Exploratory Data Analysis
Exploratory data analysis is done to get an understanding of the data and identify any patterns or trends.


# In[ ]:


#to check for inconsistent data types


# In[33]:


df.dtypes


# In[34]:


#the variable datatype has to be changed from object to datetime
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])


# In[35]:


#summary statistics of quantitative variables

quant_vars = ['QUANTITYORDERED','PRICEEACH','SALES','MSRP']
df[quant_vars].describe()


# In[ ]:


#Average Quantity ordered is 35 units
#Average price is $83
Average Sales is $3,553


# #Exploring Variables 

# In[36]:


plt.figure(figsize=(9,6))
sns.distplot(df['QUANTITYORDERED'])
plt.title('Order Quantity Distribution')
plt.xlabel('Quantity Ordered')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


#from the distribution of the graph above, we can see that the maority of the orders are in bulk within a range of 20 to 60 orders. 


# In[37]:


plt.figure(figsize=(9,6))
sns.distplot(df['PRICEEACH'])
plt.title('Price Distribution')
plt.xlabel('Price Ordered')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


#The distribution of Price is Left Skewed with max price of 100$. Interestingly, many of the orders recieved are of this price.


# In[38]:


plt.figure(figsize=(9,6))
sns.distplot(df['SALES'])
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


#As stated earlier, the bulk of the sales orders is between 2,000 - 6,000 units per order


# In[43]:


#Analyzing the shipping status
df['STATUS'].value_counts(normalize = True)


# In[44]:


# create a pandas series with the data
data = pd.Series([0.927028, 0.021254, 0.016649, 0.015586, 0.014524, 0.004959],
                 index=['Shipped', 'Cancelled', 'Resolved', 'On Hold', 'In Process', 'Disputed'])

# create a bar plot
data.plot(kind='bar')

# set the plot title and axis labels
plt.title('Order Status Distribution')
plt.xlabel('Order Status')
plt.ylabel('Percentage')

# show the plot
plt.show()


# In[ ]:


#we can see from the above graph that the company has a very high successfull shipping rate of 92% while the disputed orders are just 0.004%


# In[45]:


#checking the time range of the data set: 
df.groupby(['YEAR_ID'])['MONTH_ID'].nunique()


# In[ ]:


#we can see that the data for 2005 isn't complete, as such we would discard the first five months to enable a fair comparison.


# In[46]:


#Analyzing Deal Sizes
plt.figure(figsize=(9,6))
df['DEALSIZE'].value_counts(normalize = True).plot(kind = 'bar')
plt.title('DealSize distribution')
plt.xlabel('Deal Size')
plt.ylabel('% Proportion')
plt.show()


# In[ ]:


#About 50% of the deals made are of medium size. 


# In[ ]:





# # KPI ANALYSIS

# In[ ]:


# Annual Revenue


# In[47]:


plt.figure(figsize=(9,6))
df.groupby(['YEAR_ID'])['SALES'].sum().plot()
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.title('Annual Revenue')
plt.xticks(np.arange(2003,2006,1))
plt.show()


# In[ ]:


#There's a significant peak in revenue in 2004 and it dropped significantly in 2015, this is because we only have five month data in 2015. 


# In[48]:


#Monthly Revenue
plt.figure(figsize=(9,6))

monthly_revenue = df.groupby(['YEAR_ID','MONTH_ID'])['SALES'].sum().reset_index()
monthly_revenue
sns.lineplot(x="MONTH_ID", y="SALES",hue="YEAR_ID", data=monthly_revenue)
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Monthly Revenue')
plt.show()


# In[ ]:


# for the year 2013 and 2014 we can see that that monthly sales peaked in November, this is because the goods being sold are seasonal and bought by consumers in November and December for Thaksgiving and Christmas.
# We also observed that the sales for the first 5 months in 2015 is better than previous years. This needs to be studied further so we can have more insights and create better selling strategies. 


# In[ ]:


#Monthly Revenue Growth Rate:


# In[49]:


monthly_revenue['MONTHLY GROWTH'] = monthly_revenue['SALES'].pct_change()
monthly_revenue.head()


# In[50]:


#Monthly Sales Growth Rate
plt.figure(figsize=(9,6))
sns.lineplot(x="MONTH_ID", y="MONTHLY GROWTH",hue="YEAR_ID", data=monthly_revenue)
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Monthly Sales Growth Rate')
plt.show()


# In[ ]:


#We can also see that apart from the expected high/low growth rates during the seasonal months, there is high growth rate from Apr 2005 to May 2005. Further checks shows that this was due to the sales promo done during the Easter and Ramadan seasons in 2014 and 2015. 


# In[51]:


#Top 10 countries by sales

plt.figure(figsize=(9,6))
top_cities = df.groupby(['COUNTRY'])['SALES'].sum().sort_values(ascending=False)
top_cities.plot(kind = 'bar')
plt.title('Top 10 countries by Sales')
plt.xlabel('Country')
plt.ylabel('Total Sales')
plt.show()


# In[ ]:


#the USA had the largest amount of sales followed by Spain, while Ireland had the lowest sales. 


# In[52]:


#Monthly Active Customers

#plt.figure(figsize=(10,8))
df['YEAR_MONTH'] = df['YEAR_ID'].map(str)+df['MONTH_ID'].map(str).map(lambda x: x.rjust(2,'0'))
monthly_active = df.groupby(['YEAR_MONTH'])['CUSTOMERNAME'].nunique().reset_index()
monthly_active.plot(kind='bar',x='YEAR_MONTH',y='CUSTOMERNAME')
#plt.figure(figsize=(10,8))
plt.title('Monthly Active Customers')
plt.xlabel('Month/Year')
plt.ylabel('Number of Unique Customers')
plt.xticks(rotation=90)
#plt.figure(figsize=(10,8))
plt.show()


# In[ ]:


#As expected, there's a significant increase in the number of users in October and November, due to the seasonal nature of the products. 
#We also observed a general growth in customer retention between 2004 and 2005 which is a reflection of the success in the change in the retention strategies drawn up by the marketing team. 


# In[53]:


#Average Sales per Order
average_revenue = df.groupby(['YEAR_ID','MONTH_ID'])['SALES'].mean().reset_index()
plt.figure(figsize=(10,6))
sns.lineplot(x="MONTH_ID", y="SALES",hue="YEAR_ID", data=average_revenue)
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.title('Average Sales per Order')
plt.show()


# In[54]:


#New Customers Growth Rate
df_first_purchase = df.groupby('CUSTOMERNAME').YEAR_MONTH.min().reset_index()
df_first_purchase.columns = ['CUSTOMERNAME','FirstPurchaseDate']

plt.figure(figsize=(10,6))
df_first_purchase.groupby(['FirstPurchaseDate'])['CUSTOMERNAME'].nunique().pct_change().plot(kind='bar')
plt.title('New Customers Growth Rate')
plt.xlabel('YearMonth')
plt.ylabel('Percentage Growth Rate')
plt.show()


# In[ ]:


#From the analysis above, we can see that the highest growth rate was in February 2004. A further analysis into the sales for that month is needed so we can have a better understanding of the factors that contributed to the signifficant increase. 


# # Customer Segmentation 

# Segmentation with number of clusters chosen randomly
# 

# In[55]:


df['ORDERDATE'] = [d.date() for d in df['ORDERDATE']]
df.head()


# #Calculate Recency, Frequency and Monetary value for each customer
# 
# Assuming that we are analyzing the next day of latest order date in the data set. Creating a variable 'snapshot date**' which is the latest date in data set.
# 
# Recency : Recency is the number of days between the customer's latest order date and the snapshot date
# Frequency: Number of purchases made by the customer
# MonetaryValue: Revenue generated by the customer

# In[56]:


snapshot_date = df['ORDERDATE'].max() + dt.timedelta(days=1) #latest date in the data set
df_RFM = df.groupby(['CUSTOMERNAME']).agg({
    'ORDERDATE': lambda x: (snapshot_date - x.max()).days,
    'ORDERNUMBER': 'count',
    'SALES':'sum'})

#Renaming the columns
df_RFM.rename(columns={'ORDERDATE': 'Recency',
                   'ORDERNUMBER': 'Frequency',
                   'SALES': 'MonetaryValue'}, inplace=True)


# In[57]:


df_RFM.head()


# In[ ]:


#Customer base is randomly divided into 4 segments


# In[58]:


# Create a spend quartile with 4 groups - a range between 1 and 5
MonetaryValue_quartile = pd.qcut(df_RFM['MonetaryValue'], q=4, labels=range(1,5))
Recency_quartile = pd.qcut(df_RFM['Recency'], q=4, labels=list(range(4, 0, -1)))
Frequency_quartile = pd.qcut(df_RFM['Frequency'], q=4, labels=range(1,5))


# Assign the quartile values to the Spend_Quartile column in data
df_RFM['R'] = Recency_quartile
df_RFM['F'] = Frequency_quartile
df_RFM['M'] = MonetaryValue_quartile

#df_RFM[['MonetaryValue_Quartile','Recency_quartile','Frequency_quartile']] = [MonetaryValue_quartile,Recency_quartile,Frequency_quartile]

# Print data with sorted Spend values
#print(df_RFM.sort_values('MonetaryValue'))


# In[59]:


df_RFM.head()


# In[60]:


# Calculate RFM_Score
df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)
df_RFM.head()


# In[65]:


#Naming Levels
# Define rfm_level function
def rfm_level(df):
    if np.bool(df['RFM_Score'] >= 10):
        return 'High Value Customer'
    elif np.bool((df['RFM_Score'] < 10) & (df['RFM_Score'] >= 6)):
        return 'Mid Value Customer'
    else:
        return 'Low Value Customer'

# Create a new variable RFM_Level
df_RFM['RFM_Level'] = df_RFM.apply(rfm_level, axis=1)

# Print the header with top 5 rows to the console
df_RFM.head(20)


# In[62]:


plt.figure(figsize=(10,6))
df_RFM['RFM_Level'].value_counts(normalize = True).plot(kind='bar')
plt.title('RFM_level Distribution')
plt.xlabel('RFM_Level')
plt.ylabel('% Proportion')
plt.show()


# # Segmentation using KMeans Clustering

# Data Preprocessing for KMeans
# K Means Assumptions
# 
# All variables have symmetrical (Normal) Distribution
# All Variables have same average value(approx)
# All Variables have same variance(approx)

# In[64]:


#checking the distribution of the variables

data = df_RFM[['Recency','Frequency','MonetaryValue']]
data.head(20)


# In[66]:


plt.figure(figsize=(10,6))

plt.subplot(1,3,1)
data['Recency'].plot(kind='hist')
plt.title('Recency')

plt.subplot(1,3,2)
data['Frequency'].plot(kind='hist')
plt.title('Frequency')

plt.subplot(1,3,3)
data['MonetaryValue'].plot(kind='hist')
plt.xticks(rotation = 90)
plt.title('MonetaryValue')

plt.tight_layout()
plt.show()


# In[67]:


#Removing the skewness by performing log transformation on the variables

data_log = np.log(data)
data_log.head()


# In[68]:


#Distribution of Recency, Frequency and MonetaryValue after removing the skewness
plt.figure(figsize=(10,6))

#plt.subplot(1,3,1)
sns.distplot(data_log['Recency'],label='Recency')

#plt.subplot(1,3,1)
sns.distplot(data_log['Frequency'],label='Frequency')

#plt.subplot(1,3,1)
sns.distplot(data_log['MonetaryValue'],label='MonetaryValue')

plt.title('Distribution of Recency, Frequency and MonetaryValue after Log Transformation')
plt.legend()
plt.show()


# In[69]:


#Standardizing the variables using StandardScaler() for equal variance and mean

# Initialize a scaler
scaler = StandardScaler()

# Fit the scaler
scaler.fit(data_log)

# Scale and center the data
data_normalized = scaler.transform(data_log)

# Create a pandas DataFrame
data_normalized = pd.DataFrame(data_normalized, index=data_log.index, columns=data_log.columns)

# Print summary statistics
data_normalized.describe().round(2)


# In[71]:


#Choosing number of Clusters using Elbow Method
# Fit KMeans and calculate SSE for each k
sse={}
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(data_normalized)
    sse[k] = kmeans.inertia_ 

    
plt.figure(figsize=(10,6))
# Add the plot title "The Elbow Method"
plt.title('The Elbow Method')

# Add X-axis label "k"
plt.xlabel('k')

# Add Y-axis label "SSE"
plt.ylabel('SSE')
# Plot SSE values for each key in the dictionary
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.text(4.5,60,"Largest Angle",bbox=dict(facecolor='lightgreen', alpha=0.5))
plt.show()


# In[72]:


#Running KMeans with 5 clusters
# Initialize KMeans
kmeans = KMeans(n_clusters=5, random_state=1) 

# Fit k-means clustering on the normalized data set
kmeans.fit(data_normalized)

# Extract cluster labels
cluster_labels = kmeans.labels_

# Assigning Cluster Labels to Raw Data
# Create a DataFrame by adding a new cluster label column
data_rfm = data.assign(Cluster=cluster_labels)
data_rfm.head()


# In[73]:


# Group the data by cluster
grouped = data_rfm.groupby(['Cluster'])

# Calculate average RFM values and segment sizes per cluster value
grouped.agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
  }).round(1)


# # Building Customer Personas
# Customer Pesonas can build by determining the summary stats of RFM values or Snake Plot. Snake Plots is a Market Research technique used to compare segments. Visual representation of each segment's attributes helps us to determine the relative Importance of segment attributes
# 
# #Snake Plot

# In[75]:


data_rfm_melt = pd.melt(data_rfm.reset_index(), id_vars=['CUSTOMERNAME', 'Cluster'],
                        value_vars=['Recency', 'Frequency', 'MonetaryValue'], 
                        var_name='Metric', value_name='Value')

plt.figure(figsize=(10,6))
# Add the plot title
plt.title('Snake plot of normalized variables')

# Add the x axis label
plt.xlabel('Metric')

# Add the y axis label
plt.ylabel('Value')

# Plot a line for each value of the cluster variable
sns.lineplot(data=data_rfm_melt, x='Metric', y='Value', hue='Cluster')
plt.show()


# In[77]:


# Calculate average RFM values for each cluster
cluster_avg = data_rfm.groupby(['Cluster']).mean() 
print(cluster_avg)
            


# In[78]:


# Calculate average RFM values for the total customer population
population_avg = data.mean()
print(population_avg)


# In[79]:


# Calculate relative importance of cluster's attribute value compared to population
relative_imp = cluster_avg / population_avg - 1

# Print relative importance score rounded to 2 decimals
print(relative_imp.round(2))


# In[80]:


# Initialize a plot with a figure size of 8 by 2 inches 
plt.figure(figsize=(8, 2))

# Add the plot title
plt.title('Relative importance of attributes')

# Plot the heatmap
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
plt.show()


# In[ ]:




