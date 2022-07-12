import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


def fix_gender(x):
    if "Man" in x:
        return "Man"
    if "Woman" in x:
        return "Woman"
    if "Woman;Man" in x:
        return "trans"
    if "Man;Woman" in x:
        return "trans"


def clean_data():
    df = pd.read_csv('survey_results_public.csv', usecols=[
                     "Age", "Age1stCode", "ConvertedComp", "Country", "EdLevel", "Gender", "YearsCode", "YearsCodePro", "LanguageWorkedWith"])
    df.dropna(how="any", subset=["Age", "YearsCode"], inplace=True)
    df.replace({"Age1stCode": "[A-Za-z]"}, " ", regex=True, inplace=True)
    df.replace({"Age1stCode": "nan"}, np.nan, inplace=True)
    df["Age1stCode"] = df.Age1stCode.astype(float)
    df.dropna(how="any", subset=["Age1stCode"], inplace=True)
    df.replace({"YearsCodePro": "[A-Za-z]"}, " ", regex=True, inplace=True)
    df["YearsCodePro"] = df.YearsCodePro.astype(float)
    df.dropna(how="any", subset=["YearsCodePro"], inplace=True)
    df.rename(columns={"ConvertedComp": "Salary"}, inplace=True)
    df.dropna(how="any", subset=["Salary"], inplace=True)
    df.dropna(how="any", subset=["Country", "EdLevel",
              "Gender", "YearsCode"], inplace=True)
    df["Gender"] = df["Gender"].apply(fix_gender)
    return df


def plots():
    df = clean_data()
    st.write("##### Note: Below all visualization are based on data after preprocessing #####")
    plt.style.use("ggplot")
    st.write("     ")
    st.write("## Percentage of Man and Women in survey ##")
    fig2 = plt.figure(figsize=(10, 5))
    df["Gender"].value_counts().plot(kind="pie", autopct="%.2f")
    st.pyplot(fig2)
    fig1 = plt.figure(figsize=(8, 4))
    sns.barplot(x=df["Gender"], y=df["Salary"])
    st.write("## Salary Visual of Man's & Women ##")
    fig1 = plt.figure(figsize=(8, 4))
    sns.barplot(x=df["Gender"], y=df["Salary"])
    st.pyplot(fig1)
    st.write("## Visual between Age and Professional Code with Gender  ##")
    fig3 = plt.figure(figsize=(10, 8))
    a = st.number_input(
        "Enter or Increase Age till you want to visaulize the distributions", max_value=50, min_value=5, step=4)
    df1 = df[df["Age"] <= a]
    sns.scatterplot(x=df1["Age"], y=df["Salary"], hue=df["Gender"])
    st.pyplot(fig3)
