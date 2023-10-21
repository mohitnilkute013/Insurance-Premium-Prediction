import streamlit as st
import streamlit.components.v1 as components

import os
import sys
from logger import logging
from exception import CustomException
import pandas as pd

from pipelines.prediction_pipeline import CustomData, PredictPipeline
from pipelines import training_pipeline


def predict_query(form_query):
    data = CustomData(
        age = form_query['age'],
        sex = form_query['sex'],
        bmi = form_query['bmi'],
        children = form_query['children'],
        smoker = form_query['smoker'],
        region = form_query['region']
    )

    final_new_data=data.get_data_as_dataframe()
    predict_pipeline=PredictPipeline()
    pred=predict_pipeline.predict(final_new_data)

    result = round(pred[0], 2)

    return result


def ChangeWidgetFontSize(wgt_txt, wch_font_size = '12px'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        { elements[i].style.fontSize='""" + wch_font_size + """';} } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)
    

def home_page(parameters):

    st.title("Insurance Premium Prediction App")

    # Using object notation
    # add_selectbox = st.sidebar.selectbox(
    #     "How would you like to be contacted?",
    #     ("Email", "Home phone", "Mobile phone")
    # )

    # Using "with" notation
    with st.sidebar:
        mode_radio = st.radio(
            "Mode",
            ("Train", "Predict")
        )

        # print(add_radio)

        ChangeWidgetFontSize('Mode', '32px')

    if mode_radio == 'Predict':

        st.write(""" ### Please fill the below information required for prediction! """)

        form_dict = {}
        for param, param_value in parameters.items():
            if param_value == 'text':
                form_dict[param] = st.text_input(param)
            elif hasattr(param_value, '__iter__'):
                form_dict[param] = st.selectbox(param, param_value)


        submit = st.button("Calculate Expense")

        if submit:
            print("Let's Predict.")

            results = predict_query(form_query = form_dict)

            st.subheader(f"The Predicted Expense is: {results}.")
    
    else:
        
        st.write(""" Let's train the Model ! AGAIN ?""")

        submit = st.button("Train our model")

        if submit:
            print("Let's train the model ! ")

            with st.spinner("Training on current stored data ..."):
                completed = training_pipeline.Train()
                if completed:
                    st.success(":D Training Completed!")
                    st.balloons()
                    # st.snow()
                else:
                    st.error(':| Something went wrong while training!')

        


if __name__ == '__main__':
    
    parameters = {
            'age': 'text',
            'sex': ['male', 'female'],
            'bmi': 'text',
            'children': 'text',
            'smoker': ['yes', 'no'],
            'region': ['southeast', 'southwest', 'northwest', 'northeast']
            # Add more parameters as needed
        }
    
    home_page(parameters=parameters)