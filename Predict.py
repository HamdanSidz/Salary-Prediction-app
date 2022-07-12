from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from explore import plots
from sklearn.metrics import mean_squared_error


le_edu = LabelEncoder()
le_country = LabelEncoder()

ran_for_reg = RandomForestRegressor(random_state=0)


countries = ['United Kingdom', 'Spain', 'Netherlands', 'United States',
             'Canada', 'Brazil', 'France', 'Poland', 'Germany', 'Ukraine',
             'India', 'Mexico', 'Sweden', 'Turkey', 'Australia', 'Italy',
             'Norway', 'Pakistan', 'Israel', 'Russian Federation',
             'South Africa', 'Ireland', 'Switzerland']

edu = ["Bachelor's degree", "Master's degree",
       "Post Graduate degree", "Professional degree", "Associate degree"]

countries_val = []
edu1_val = []

dic_countries = {}
dic_edu1 = {}


def shortcountry(lis, cut):
    dic = {}
    for i in range(len(lis)):
        if lis[i] >= cut:
            dic[lis.index[i]] = lis.index[i]
        else:
            dic[lis.index[i]] = "other"
    return dic


def clear_edu(x):
    if "Bachelor? degree" in x:
        return "Bachelor's degree"
    if "Master? degree" in x:
        return "Master's degree"
    if "Professional degree" in x:
        return "Professional degree"
    if "Post Graduate" in x:
        return "Post Graduate degree"
    if "Associate degree" in x:
        return "Associate degree"
    else:
        return "other(Primary/Secondary school)"


def clean_train_data():

    df = pd.read_csv("Book1.csv")
    df.dropna(how="any", subset=["ConvertedComp",
              "YearsCodePro"], inplace=True)
    df.rename(columns={"ConvertedComp": "Salary"}, inplace=True)
    df = df[df.Employment == "Employed full-time"]
    dic = shortcountry(df["Country"].value_counts(), 250)
    df["Country"] = df["Country"].map(dic)
    df = df[df["Salary"] >= 1000]
    df = df[df["Salary"] <= 80000]
    df = df[df["Country"] != "other"]
    df.dropna(how="any", subset=["YearsCodePro"], inplace=True)
    df.replace({"YearsCodePro": "[A-Za-z]"}, " ", regex=True, inplace=True)
    df.dropna(how="any", subset=["EdLevel"], inplace=True)
    df["EdLevel"] = df["EdLevel"].apply(clear_edu)

    le_edu = LabelEncoder()
    df["EdLevel"] = le_edu.fit_transform(df["EdLevel"])

    le_country = LabelEncoder()
    df["Country"] = le_country.fit_transform(df["Country"])

    df.drop(["Employment"], axis=1, inplace=True)
    x = df.drop("Salary", axis=1)
    y = df["Salary"]
    ran_for_reg.fit(x, y)

    dic_countries = dict(zip(countries, df["Country"].unique()))
    dic_edu1 = dict(zip(edu, df["EdLevel"].unique()))
    return dic_countries, dic_edu1, y, x


def predict_page():

    st.title("💰 App For Salary Prediction ")
    st.write("### Before proceeds we need some inputs to predict salary ###")
    country = st.selectbox("Select country", countries)
    education = st.selectbox("Select education", edu)
    exp = st.slider("Year(s) of Experience", 0, 50, 1)
    # funct call train_train_data()
    dic_countries, dic_edu1, y, dup_x = clean_train_data()
    on = st.button("Predict Salary")

    def fix_countries(a):
        for keys in dic_countries:
            if keys == a:
                return dic_countries[keys]

    def fix_edu1(b):
        for keys in dic_edu1:
            if keys == b:
                return dic_edu1[keys]

    if on == True:
        x1 = []
        x = [country, education, exp]

        a = x[0]
        a = fix_countries(a)
        x[0] = a
    #    st.subheader(x[0:1])

        b = x[1]
        b = fix_edu1(b)
        x[1] = b

    #    st.subheader(x[1:2])
        x1.append(x)
    #   print(x)

        y_pre = ran_for_reg.predict(x1)
    #   error=np.sqrt(mean_squared_error(y,y_pre))
    #   print(error)
    #   print(ran_for_reg.score(dup_x,y))
        st.subheader(f"Your estimated annual salary is: ${y_pre.round(3)}")
        st.write("       ")
        st.write("       ")
        st.write("       ")
        st.write("       ")
        st.write("       ")
        st.write("##### Note: This prediction is based on Stackoverflow Developers Survey 2021 in which full time employed developers are considered. #####")


def app():
    a = st.sidebar.selectbox("Selects From Following Pages", [
                             "Prediction app", "Visualizations on data"])
    if a == "Visualizations on data":
        plots()
    else:
        predict_page()


app()
