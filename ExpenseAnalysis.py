# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, cross_decomposition
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#from tabulate import tabulate
import datetime
import time

style.use('ggplot')

#importing consumer expenditure data

dataframe1 = pd.read_csv('cx30y.csv')

'''
First compare demographic categories age,income,race, and their correlation with spending on 
different consumer items over the years and how the changes reflect spending power
'''

#comparing meat,poultry,fish and eggs spending
#replacing series_id's we need for this analysis
replaced1=dataframe1.replace(['CXUANIMALLB0102M ','CXUANIMALLB0106M ','CXUANIMALLB0402M ','CXUANIMALLB0403M ','CXUANIMALLB0404M ','CXUANIMALLB0405M ','CXUANIMALLB0406M ','CXUANIMALLB0407M ','CXUANIMALLB0806M ','CXUANIMALLB0809M ','CXUANIMALLB0902M ','CXUANIMALLB0905M ','CXUANIMALLB01002M '],['Lowincome','Highincome','age<25','age25-35','age35-45','age45-55','age55-65','age>65','urban','rural','white/asian','black','hispanic'])

Lowincomempfe=replaced1['seriesid']=="Lowincome"

Lowincomempfe2=replaced1['year']<2016

Lowincomempfe3=replaced1[Lowincomempfe & Lowincomempfe2]

Highincomempfe=replaced1['seriesid']=="Highincome"

Highincomempfe2=replaced1['year']<2016

Highincomempfe3=replaced1[Highincomempfe & Highincomempfe2]

#Highincomempfe3.corrwith(Lowincomempfe3,axis=0,drop=False)
#Highincomempfe3.merge(Lowincomempfe3, left_on='seriesid', right_on='year', how='outer')

mpfeincomeframe=[Lowincomempfe3,Highincomempfe3]

resultincomempfe=pd.concat(mpfeincomeframe)
#plt.plot(resultincomempfe['year'],resultincomempfe['value'])

#plt.bar(resultincomempfe['year'],resultincomempfe['value'])


#g=sns.FacetGrid(resultincomempfe,col='seriesid')
#plt.title('Comparison of meat,poultry,fish,eggs spending by income groups')
#g.map(sns.regplot,"year","value")
#plt.show()

agempfe=replaced1['seriesid']=="age<25"

agempfe2=replaced1['year']<2016

agempfe3=replaced1[agempfe & agempfe2]

age2mpfe=replaced1['seriesid']=="age25-35"

age2mpfe2=replaced1['year']<2016

age2mpfe3=replaced1[age2mpfe & age2mpfe2]

age3mpfe=replaced1['seriesid']=="age35-45"

age3mpfe2=replaced1['year']<2016

age3mpfe3=replaced1[age3mpfe & age3mpfe2]

age4mpfe=replaced1['seriesid']=="age35-45"

age4mpfe2=replaced1['year']<2016

age4mpfe3=replaced1[age4mpfe & age4mpfe2]

age5mpfe=replaced1['seriesid']=="age45-55"

age5mpfe2=replaced1['year']<2016

age5mpfe3=replaced1[age5mpfe & age5mpfe2]

age6mpfe=replaced1['seriesid']=="age55-65"

age6mpfe2=replaced1['year']<2016

age6mpfe3=replaced1[age6mpfe & age6mpfe2]

age7mpfe=replaced1['seriesid']=="age>65"

age7mpfe2=replaced1['year']<2016

age7mpfe3=replaced1[age7mpfe & age7mpfe2]

#age2mpfe3.corrwith(agempfe3,axis=0,drop=False)
#Highincomempfe3.merge(Lowincomempfe3, left_on='seriesid', right_on='year', how='outer')

mpfeageframe=[agempfe3,age2mpfe3,age3mpfe3,age4mpfe3,age5mpfe3,age6mpfe3,age7mpfe3]

resultagempfe=pd.concat(mpfeageframe)
#plt.plot(resultincomempfe['year'],resultincomempfe['value'])

'''g=sns.FacetGrid(resultagempfe,col='seriesid')
plt.title('Comparison of meat,poultry,fish,eggs spending by age groups')
g.map(sns.regplot,"year","value")
plt.show()
'''

areampfe=replaced1['seriesid']=="urban"
area2mpfe2=replaced1['year']<2016
area1mpfe=replaced1[areampfe & area2mpfe2]

areampfe2=replaced1['seriesid']=="rural"
area3mpfe2=replaced1['year']<2016
area1mpfe2=replaced1[areampfe2 & area3mpfe2]
mpfeareaframe=[area1mpfe2,area1mpfe]
resultareampfe=pd.concat(mpfeareaframe)

'''g=sns.FacetGrid(resultareampfe,col='seriesid')
plt.title('Comparison of meat,poultry,fish,eggs spending by area groups')
g.map(sns.regplot,"year","value")
plt.show()'''

c=replaced1['year']<2016
whitempfe=replaced1['seriesid']=="white/asian"
white2mpfe2=replaced1['year']<2016
white1mpfe=replaced1[whitempfe & white2mpfe2]

blackmpfe2=replaced1['seriesid']=="black"
black1mpfe2=replaced1[blackmpfe2 & c]

hispanicmpfe2=replaced1['seriesid']=="hispanic"
hispanic2mpfe2=replaced1['year']<2016
hispanic1mpfe2=replaced1[hispanicmpfe2 & hispanic2mpfe2]

mpferaceframe=[white1mpfe,black1mpfe2,hispanic1mpfe2]
resultracempfe=pd.concat(mpferaceframe)

#g=sns.FacetGrid(resultracempfe,col='seriesid')
#plt.title('Comparison of meat,poultry,fish,eggs spending by racial groups')
#g.map(sns.regplot,"year","value")
#plt.show()

mainframempfecom1=[Highincomempfe3,Lowincomempfe3,white1mpfe,black1mpfe2,area1mpfe2,area1mpfe]
finalmpfeframe1=pd.concat(mainframempfecom1)

#print(finalfood)
#finalmpfeframe.describe()

newmpfeframe1=finalmpfeframe1.pivot(index='year', columns='seriesid', values='value')

newmpfeframe1['totalpie']=newmpfeframe1.sum(axis=1)

#sframe=newmpfeframe1.stack()

# plot chart
#sframe.plot(kind='pie' , autopct='%1.1f%%', subplots=True)
#plt.title('Pie chart of MPFE for 2015')
#plt.xlabel('2015')

#plt.ylabel('spending')

#fig = plt.figure(figsize=(6,6), dpi=100)

#ax = plt.subplot(111)

#sframe.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=270, fontsize=17, subplots=True)
#Taking a small subset of all food items and replacing their series_id with titles
#All consumer units value are taken here
#data2 has foods data

data2=dataframe1.replace(['CXUANIMALLB0101M ','CXUCERBAKRYLB0101M ','CXUDAIRYLB0101M ','CXUFOODAWAYLB0101M ','CXUFRUITVEGLB0101M ','CXUOTHRFOODLB0101M '],['MPFE','CBP','DP','FDAWAY','FV','OFAH'])
'''MPFE =meat,poultry,fish and eggs
CBP=cereals and bakery products
DP=Dairy Products
FDAWAY=Food consumed on outside trips
FV=Fruits and Vegetables
OFAH=Other Foods at home
'''
#datahousing has Shelter,Utilities,Household Operations,Supplies and Furnishings instead of all the categories in Housing as these were the most important
#for all consumer units

datahousing=dataframe1.replace(['CXUSHELTERLB0101M ','CXUUTILSLB0101M ','CXUHHOPERLB0101M ','CXUHKPGSUPPLB0101M ','CXUHHFURNSHLB0101M '],['Shelter','Utilities','HouseholdOper','Supplies','Furnishings'])

datahealthcare=dataframe1.replace(['CXUHEALTHLB0101M '],['Healthcare'])

#g=sns.FacetGrid(datahealthcare,col='seriesid')
#g.map(sns.regplot,"year","value")

dataapparel=dataframe1.replace(['CXUAPPARELLB0101M '],['Apparel'])

dataentertainment=dataframe1.replace(['CXUENTEROTHLB0101M '],['Entertainment'])

dataeducation=dataframe1.replace(['CXUEDUCATNLB0101M '],['Education'])

datatransport=dataframe1.replace(['CXUNEWCARSLB0101M ','CXUGASOILLB0101M ','CXUOTHVEHCLLB0101M ','CXUPUBTRANSLB0101M '],['VehiclePur','Gas','Other','Public/Other'])

#dataframe dataMPFE1 has the subcategories for consumer spending in meat,poultry,fish and eggs

dataMPFE1=dataframe1.replace(['CXUANIMALLB0201M ','CXUANIMALLB0401M ','CXUANIMALLB0601M ','CXUANIMALLB0801M ','CXUANIMALLB0901M ','CXUANIMALLB1101M ','CXUANIMALLB1301M '],['Income','Age','MaritalStatus','Housing','Race','Location','Education'])

print(datahousing)

#a holds year values<2016 for which the dataframe data2 has values for

a=data2['year']<2020

#Storing in new dataframes values for all the subcategories and merging with year<2016
Shelter1=datahousing['seriesid']=='Shelter'
Shelter2=datahousing[Shelter1 & a]

Utilities1=datahousing['seriesid']=='Utilities'
Utilities2=datahousing[Utilities1 & a]

Houseoperations1=datahousing['seriesid']=='HouseholdOper'
Houseoperations2=datahousing[Houseoperations1 & a]

Supplies1=datahousing['seriesid']=='Supplies'
Supplies2=datahousing[Supplies1 & a]

Furnishings1=datahousing['seriesid']=='Furnishings'
Furnishings2=datahousing[Furnishings1 & a]

#For health,education,entertainment,apparel

health1=datahealthcare['seriesid']=='Healthcare'
health2=datahealthcare[health1 & a]

education1=dataeducation['seriesid']=='Education'
education2=dataeducation[education1 & a]

entertainment1=dataentertainment['seriesid']=='Entertainment'
entertainment2=dataentertainment[entertainment1 & a]

apparel1=dataapparel['seriesid']=='Apparel'
apparel2=dataapparel[apparel1 & a]

mainframeheae=[health2,entertainment2,education2,apparel2]
finalheae=pd.concat(mainframeheae)

newheae=finalheae.pivot(index='year', columns='seriesid', values='value')
newheae['totalheae']=newheae.sum(axis=1)
newheae['totalchangeheae']=newheae['totalheae'].pct_change()
#For transport
Vehpur1=datatransport['seriesid']=='VehiclePur'
Vehpur2=datatransport[Vehpur1 & a]

Gas1=datatransport['seriesid']=='Gas'
Gas2=datatransport[Gas1 & a]

Other1=datatransport['seriesid']=='Other'
Other2=datatransport[Other1 & a]

Public1=datatransport['seriesid']=='Public/Other'
Public2=datatransport[Public1 & a]

#mainframehouse has all the dataframes for house category and finalhouse constitues all dataframes concatenated
mainframehouse=[Shelter2,Utilities2,Houseoperations2,Supplies2,Furnishings2]
finalhouse=pd.concat(mainframehouse)

#similarly mainframetrans and finaltrans for transport
mainframetrans=[Vehpur2,Gas2,Other2,Public2]
finaltrans=pd.concat(mainframetrans)

#newhouse dataframe has index replaced with year and columns represented as series_id's through pivot function

newhouse=finalhouse.pivot(index='year', columns='seriesid', values='value')
#newhouse[['Shelter','Utilities','HouseholdOper','Supplies','Furnishings']].plot(kind='line', use_index=True)

newhouse['totalhouse']=newhouse.sum(axis=1)
newhouse['totalchangehouse']=newhouse['totalhouse'].pct_change()
print(newhouse)

newtrans=finaltrans.pivot(index='year', columns='seriesid', values='value')
#newtrans[['VehiclePur','Gas','Other','Public/Other']].plot(kind='line', use_index=True)
newtrans['totaltrans']=newtrans.sum(axis=1)
newtrans['totalchangetrans']=newtrans['totaltrans'].pct_change()
#Similarly creating dataframes for each subcategory for MPFE and merging with 'a' to create new dataframes
incomempfe=dataMPFE1['seriesid']=='Income'
incomempfe2=dataMPFE1[incomempfe & a]

agempfe=dataMPFE1['seriesid']=='Age'
agempfe2=dataMPFE1[agempfe & a]

maritalmpfe=dataMPFE1['seriesid']=='MaritalStatus'
maritalmpfe2=dataMPFE1[maritalmpfe & a]

housingmpfe=dataMPFE1['seriesid']=='Housing'
housingmpfe2=dataMPFE1[housingmpfe & a]

racempfe=dataMPFE1['seriesid']=='Race'
racempfe2=dataMPFE1[racempfe & a]

locationmpfe=dataMPFE1['seriesid']=='Location'
locationmpfe2=dataMPFE1[locationmpfe & a]

educationmpfe=dataMPFE1['seriesid']=='Education'
educationmpfe2=dataMPFE1[educationmpfe & a]

mainframempfe=[incomempfe2,agempfe2,maritalmpfe2,housingmpfe2,racempfe2,locationmpfe2,educationmpfe2]
finalmpfe=pd.concat(mainframempfe)

print(finalmpfe.head(10))

#using seaborn library plotting mpfe subcategories spending over the 30years(all consumer units)
#to verify data is correct
#g=sns.FacetGrid(finalmpfe,col='seriesid')
#g.map(sns.regplot,"year","value")

#food data
mpfe1=data2['seriesid']=='MPFE'
mpfe2=data2[mpfe1 & a]

cbp=data2['seriesid']=='CBP'
cbp2=data2[cbp&a]

dp=data2['seriesid']=='DP'
dp2=data2[dp&a]

fdaway=data2['seriesid']=='FDAWAY'
fdaway2=data2[fdaway&a]

fv=data2['seriesid']=='FV'
fv2=data2[fv&a]

ofah=data2['seriesid']=='OFAH'
ofah2=data2[ofah&a]
mainframefood=[mpfe2,cbp2,dp2,fdaway2,fv2,ofah2]
finalfood=pd.concat(mainframefood)

#print(finalfood)
finalfood.describe()

newfoods=finalfood.pivot(index='year', columns='seriesid', values='value')
#print(newfoods)

#create a new column in newfoods dataframe called total and save the sum of all column values in that. This column represents total average consumer annual expenditure
#on food and what we will be using to predict future expenditure
newfoods['total']=newfoods.sum(axis=1)
newfoods['totalchangefoods']=newfoods['total'].pct_change()
#print(newfoods)

#Bigframe containing all the data and total of consumer expenditure for different types of food, housing, healthcare, transport, entertainment, apparel,education
Bigframe1=[Shelter2,Utilities2,Houseoperations2,Supplies2,Furnishings2,health2,entertainment2,education2,apparel2,Vehpur2,Gas2,Other2,Public2,mpfe2,cbp2,dp2,fdaway2,fv2,ofah2]
Bigframe2=pd.concat(Bigframe1)

newbig=Bigframe2.pivot(index='year', columns='seriesid', values='value')
newbig['averageconsumerexpense']=newbig.sum(axis=1)
newbig['aveconsumerexpchange']=newbig['averageconsumerexpense'].pct_change()
#plot averageconsumerexpense
newbig[['averageconsumerexpense']].plot(kind='line', use_index=True)

#correlation matrix heatmap for newfoods
'''corr = newfoods.corr()
corr = (corr)
sns.heatmap(corr, 
xticklabels=corr.columns.values,
yticklabels=corr.columns.values)
sns.plt.title('Heatmap of Correlation Matrix')
corr
plt.show()'''

#Applying Linear regression to predict future consumption
#select column to predict future values for

Predictionbase = 'averageconsumerexpense'

#replace any na values if present with a significantly low value that does not effect our final results

newbig.fillna(value=-99999, inplace=True)

#Only predicting 0.033% of future values
Output = int(math.ceil(0.033 * len(newbig)))

#shifting whole dataset up 1 and saving in new column label
newbig['label'] = newbig[Predictionbase].shift(-Output)

#converting dataframe to numpy array for training and testing we need data in array form
X = np.array(newbig.drop(['label'], 1))

#preprocessing step

X = preprocessing.scale(X)

X_lately = X[-Output:]

X = X[:-Output]

newbig.dropna(inplace=True)

y = np.array(newbig['label'])

#splitting into training and testing set whereby 20% of complete data is test set
trainingX, testingX, trainingY, testingY = cross_decomposition.train_test_split(X, y, test_size=0.2)

#Building a classifier
clf = LinearRegression(n_jobs=-1)
clf.fit(trainingX, trainingY)
accuracy = clf.score(testingX, testingY)
print(accuracy)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, Output)

newbig['Forecast'] = np.nan
'''newbig.index = pd.to_datetime(newbig.index, format='%Y')

last_date = newbig.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_year = 31536000
next_unix = last_unix + one_year

for i in forecast_set:
next_date = datetime.datetime.fromtimestamp(next_unix)
next_unix += 31536000
newbig.loc[next_date] = [np.nan for _ in range(len(newbig.columns)-1)]+[i]
'''
print(newbig.tail(10))

newbig['averageconsumerexpense'].plot()

newbig['Forecast'].plot()

plt.legend(loc=22)

plt.xlabel('year')

plt.ylabel('spending')

plt.show()
