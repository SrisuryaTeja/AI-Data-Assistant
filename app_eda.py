import os
import pandas as pd
import streamlit as st
from apikey import apikey
from dotenv import load_dotenv,find_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_experimental.agents import create_pandas_dataframe_agent


dotenv_path=find_dotenv()
load_dotenv(dotenv_path)


st.title('AI Assistant For Data Science')

st.write('Hello, I am your AI assistant and I am here to help you with your data science projects')

with st.sidebar:
    st.write("*Your Data Science Adventure Begins with a CSV Upload*")
    st.caption('''You may already know that every exciting data science journey 
               starts with a dataset. That's why I'd love for you to upload a CSV file. Once we have
                your data in hand, we'll dive into understanding it and have some fun exploring it.
                Then, we'll work together to shape your business challenge into a data science 
               framework. I'll introduce you to the coolest machine learning models, and we'll 
               use them to tackle your problem. Sounds fun right?''')

    st.divider()
    st.caption("<p style='text-align:center'>made by Sri</p>",unsafe_allow_html=True)

if 'clicked' not in st.session_state:
    st.session_state.clicked={1:False}

def clicked(button):
    st.session_state.clicked[button]=True

st.button("Let's get Started",on_click=clicked,args=[1])
if st.session_state.clicked[1]:
    user_csv=st.file_uploader("Upload your files",type='csv')

    if user_csv is not None :
        user_csv.seek(0)
        df=pd.read_csv(user_csv,low_memory=False)
        llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-v0.1",
                     model_kwargs={'temperature': 0.1},
                     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        
        @st.cache_data
        def steps_eda():
            steps_eda=llm('What are the steps of EDA')
            return steps_eda
        
        pandas_agent=create_pandas_dataframe_agent(llm,df,verbose=True,allow_dangerous_code=True)

        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your data looks like this")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df=pandas_agent.run("What are the meaning of the columns")
            st.write(columns_df)
            missing_values=pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates=pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation")
            st.write(df.describe())
            correlation_analysis=pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers=pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features=pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return 
        
        @st.cache_data
        def function_question_variable():
            st.line_chart(df,y=[user_question_variable])
            summary_statistics=pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
            st.write(summary_statistics)
            normality=pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            outliers=pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
            st.write(outliers)
            trends=pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            missing_values=pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
            st.write(missing_values)
            return
        
        @st.cache_data
        def function_question_dataframe():
            dataframe_info=pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return
        

        st.header("Exploratory data analysis")
        st.subheader('General information about the dataset')

        with st.sidebar:
            with st.expander("What are the steps of EDA"):
                st.write(steps_eda())

        function_agent()

        st.subheader('Variable of study')

        user_question_variable=st.text_input('What variable are you interested in')

        if user_question_variable is not None and user_question_variable!="":
            function_question_variable()

            st.subheader('Further Study')
     
        if user_question_variable:
            user_question_dataframe=st.text_input("Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe is not None and user_question_dataframe not in ("","no","NO"):
                function_question_dataframe()
            if user_question_dataframe in ("no","NO"):
                st.write("")








    