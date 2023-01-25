################# Before Application #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# After Application #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 500)
pd.set_option("display.width", 1000)
df = pd.read_csv("RBC/persona.csv")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


df["SOURCE"].value_counts()


df["PRICE"].nunique()


df["PRICE"].value_counts()


df["COUNTRY"].value_counts()


df.groupby("COUNTRY").agg({"PRICE": "sum"})


df.groupby("SOURCE").agg({"PRICE": "count"})


df.groupby("COUNTRY").agg({"PRICE": "mean"})


df.groupby("SOURCE").agg({"PRICE": "mean"})


df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": "mean"})


df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})


df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)


agg_df = (df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False))
agg_df

#converting index to variable
agg_df.reset_index(inplace=True)
agg_df

df["AGE"].nunique()

# categorization of age
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0, 18, 25, 35, 45, 67], labels=['0_18', '19_25', '26_35', '36-45', '46_67'])

agg_df

#We define new level based customers and add them as variables to the dataset.

agg_df["customers_level_based"] = [col[0].upper() + "_" + col[1].upper() + "_" + col[2].upper() + "_" + col[5].upper() for col in agg_df.values]

agg_df.head()

agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})

agg_df.head()

#We segment new customers.(USA_ANDROID_MALE_0_18)

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df

agg_df.groupby("SEGMENT").agg({"PRICE": ["mean", "max", "sum"]})

#Let's classify the new customers and estimate how much income they can bring.

agg_df = agg_df.reset_index()
new_user = "TUR_ANDROID_FEMALE_26_35"

agg_df[agg_df["customers_level_based"] == new_user]

new_user2 = "FRA_ANDROID_FEMALE_26_35"

agg_df[agg_df["customers_level_based"] == new_user2]
