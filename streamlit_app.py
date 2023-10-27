﻿# -*- coding: utf-8 -*-

from ibm_watson_machine_learning.utils import load_model_from_directory
import streamlit as st
import requests
import json
import io
import os
# from dotenv import load_dotenv

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# load_dotenv()

# api_endpoint = st.secrets['API_URL']
# api_key = st.secrets['API_KEY']
# project_id = st.secrets['PROJECT_ID']

api_endpoint = "https://us-south.ml.cloud.ibm.com"
api_key = "1C6jb8U7WQTER5ua5wFeX7HcFaC9OVRo37krWDBifr-n"
project_id = "717b33f7-c31a-4e0d-9d7e-906a32ad111d"

wxai_credentials = {
    "url": api_endpoint,
    "apikey": api_key
}

st.set_page_config(
    layout="wide",  # Set the layout to wide
    initial_sidebar_state="auto"  # Set the initial state of the sidebar
)

# Define the CSS style
css = """
{visibility: hidden;}
footer {visibility: hidden;}
body {overflow: hidden;}
data-testid="ScrollToBottomContainer"] {overflow: hidden;}
section[data-testid="stSidebar"] {
    width: 600px !important; # Set the width to your desired value
}
"""

# Display the dataframe with the custom CSS style
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# @st.cache_data(ttl=900)
def load_data(json_file_url):
    return requests.get(json_file_url)

loading_text = st.text("Loading data...")
# json_file_url = (
#     "https://raw.githubusercontent.com/martinrysanek/solaris/main/input.json"
# )

with open('input.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# response = load_data(json_file_url)
# if response.status_code == 200:
#     data = json.load(io.BytesIO(response.content))
# else:
#     st.error(f"Failed to download file. Status code: {response.status_code}")
#     exit(1)

loading_text.empty()

area_selected = st.sidebar.selectbox(
    "Select a topic",
    list(data.keys()),
    help="Select a topic",
)
button_name = data[area_selected]['buttonName']
area_language = data[area_selected]['language']
topics_list = data[area_selected]['topics']

topic_names=[]
topic_keys=[]
for topic in topics_list:
    topic_names.append(data[area_selected]['topics'][topic]['topicName'])
    topic_keys.append(topic)

selected_topic_idx = st.sidebar.radio('Select a task', range(len(topic_names)), format_func=lambda i: topic_names[i], help='Select a task')
     
model_instruction = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["instruction"]["detailed"]
model_instruction_summary = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["instruction"]["summarized"]
model_id = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["modelId"]
model_parameters = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["parameters"]
model_input = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["inputPrefix"] 
model_output = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["outputPrefix"] 
model_examples = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["examples"]

st.sidebar.write("Model Description:\n" + "\n    Task      : " + model_instruction_summary + "\n    Model name: " + model_id + "\n    Language  : " + area_language + "\n    Min tokens: " + str(model_parameters["min_new_tokens"]) + " \n    Max tokens: " + str(model_parameters["max_new_tokens"]))

MAX_LINE_LEN = 80
MAX_ROWS = 1

options=[]
options_long=[]
for row in data[area_selected]['inputs']:
    first_two_lines = row.split('\n')[:MAX_ROWS]
    first_two_lines = '\n\r'.join(first_two_lines)
    first_two_lines = first_two_lines[:MAX_LINE_LEN]
    options.append(first_two_lines)
    options_long.append(row)
    
# options_all_lines = '\n'.join(options_long)

selected_index = st.sidebar.radio("Select a case", range(len(options)), format_func=lambda i: options[i], help="Select a case")

st.subheader('Text to be evaluated', divider='rainbow')
with st.expander("Input text", expanded=True):
    st.warning(data[area_selected]['inputs'][selected_index])
  
prompt_line = "\n"
prompt_prefix = model_instruction + prompt_line 
prompt = "\n"
for example in model_examples:
    # prompt += model_input + prompt_line + example["input"] + prompt_line + prompt_line + model_output + prompt_line + example["output"] + prompt_line + prompt_line 
    # prompt += model_input + prompt_line + example["input"] + prompt_line + model_output + prompt_line + example["output"] + prompt_line 
    prompt += model_input + " " + prompt_line + example["input"] + prompt_line + prompt_line + model_output + example["output"] + " " + prompt_line + prompt_line 
prompt_suffix = model_input  + " " + data[area_selected]['inputs'][selected_index] + prompt_line + model_output + " " + prompt_line

prompt = prompt_prefix + prompt + prompt_suffix

st.subheader('Constructed prompt', divider='rainbow')
with st.expander("Prompt"):
    prompt_display = prompt.replace('\n','\n\r')
    # prompt_display = prompt
    # st.info(prompt_display)
    # st.text_area(prompt_display)
    print (prompt_display)
    st.write(prompt_display)
    
st.subheader('Result of the model', divider='rainbow')
if st.sidebar.button(button_name):
    loading_text = st.text("Loading and querying the model ...")
    parameters = model_parameters
    wxai_model = Model(
        model_id=model_id, 
        params=parameters, 
        credentials=wxai_credentials,
        project_id=project_id)

    llm_result = wxai_model.generate_text(prompt)
    # llm_result = "TEST RESULT - FRONT END DEV"
    loading_text.empty()
    with st.expander("Model result", expanded=True):
        llm_result_display = llm_result.replace('\n','\n\r')
        st.success(llm_result_display)    
    

