from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt
cardio = pd.read_csv("cardio.csv")
print(cardio.head())
print(cardio.describe(include="all")) 
print(cardio.info())
cardio.hist(figsize = (20,30))
plt.show()
sns.boxplot(x="Gender", y="Age", data=cardio)
plt.show()
new_tab = pd.crosstab(cardio["Product"], cardio["Gender"])
print(new_tab)
new_tab2 = pd.crosstab(cardio["Product"], cardio["MaritalStatus"])
print(new_tab2)
sns.countplot(x="Product",hue="Gender",data=cardio)
new_tab3 = pd.pivot_table(cardio, index=["Product", "Gender"], columns=["MaritalStatus"] , aggfunc= len)
print(new_tab3)
new_tab4 = pd.pivot_table(cardio, "Income",  index=["Product", "Gender"], columns=["MaritalStatus"])
print(new_tab4) 
new_tab5 = pd.pivot_table(cardio, "Miles",  index=["Product", "Gender"], columns=["MaritalStatus"])
print(new_tab5)
sns.pairplot(cardio)
#plt.show()
print(cardio["Age"].std())
print(cardio["Age"].mean())
sns.displot(cardio["Age"])
#plt.show()
cardio.hist(by="Gender", column = "Age")
#plt.show()
cardio.hist(by="Gender", column = "Income")
cardio.hist(by="Gender", column = "Miles")
#plt.show()
cardio.hist(by="Product", column = "Miles", figsize=(20,30))
#plt.show()
corr = cardio.corr()
print(corr)
sns.heatmap(corr, annot=True)
plt.show()
regr = linear_model.LinearRegression()
y = cardio["Miles"]
x = cardio[["Usage" , "Fitness"]]
print(regr.fit(x,y))
print(regr.coef_)
print(regr.intercept_)
# MilesPredicted = -56.74 + 20.21*Usage + 27.20*Fitness
