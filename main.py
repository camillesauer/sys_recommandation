import sys
sys.path.insert(0, "C:\\Users\\Apprenant\\PycharmProjects\\sys_recommandation")

import streamlit as st
import pandas as pd

from functions import get_recommandation, get_idea

# Importing the dataset
df = pd.read_csv("C:\\Users\\Apprenant\\PycharmProjects\\sys_recommandation\\data\\recommandation.csv")
df = df.rename(columns={df.columns[0]: "index"})
df['content'] = df[['product_name', 'brands', 'generic_name', 'categories']].astype(str).apply(lambda x: ' // '.join(x),
                                                                                               axis=1)
df['content'].fillna('Null', inplace=True)

# ##########################################################################################"
#header
st.header("Démonstration NLP:")
st.image('epingle.png', width=60)
# ###################################################################################
# sidebar
tool = st.radio('Choose Tool:',["TfidfVectorizer","CountVectorizer", "BERT"])
form = st.form(key='my_form')
new_input = form.text_input(label='Enter some text')
submit_button = form.form_submit_button(label='Chercher')
# ################################################################################
# display
title = ""
if new_input :
    if tool == "TfidfVectorizer":
        results = get_recommandation(new_input, df)
        title = "Vous aimez le " + new_input + "?"
    elif tool == "CountVectorizer":
        results = get_idea(new_input, df)
        title = "Mais vous avez envie de changement?"
    else:
        title = ("et Bert?")


    st.title(title)
    for i in range(4):
        st.write(df['product_name'].iloc[results[i][1]])
        st.write(df['brands'].iloc[results[i][1]])
        st.write(df['generic_name'].iloc[results[i][1]])
        st.write(df['categories'].iloc[results[i][1]])
        st.balloons()

