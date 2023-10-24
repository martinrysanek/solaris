# -*- coding: utf-8 -*-

from ibm_watson_machine_learning.utils import load_model_from_directory
import streamlit as st
import requests
import json
import io
import os
from dotenv import load_dotenv

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

load_dotenv()

api_endpoint = os.getenv('API_URL')
api_key = os.getenv('API_KEY')
project_id = os.getenv('PROJECT_ID')

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
json_file_url = (
    "https://raw.githubusercontent.com/martinrysanek/solaris/main/input.json"
)

# with open('input.json', 'r') as file:
#     data = json.load(file)

response = load_data(json_file_url)
if response.status_code == 200:
    data = json.load(io.BytesIO(response.content))
else:
    st.error(f"Failed to download file. Status code: {response.status_code}")
    exit(1)

loading_text.empty()

area_selected = st.sidebar.selectbox(
    "Select a topic",
    list(data.keys()),
    help="Select a topic",
)
button_name = data[area_selected]['buttonName']
area_language = data[area_selected]['language']
topics_list = data[area_selected]['topics']

st.sidebar.markdown('**Language**: &nbsp;&nbsp;'+ area_language)

topic_names=[]
topic_keys=[]
for topic in topics_list:
    topic_names.append(data[area_selected]['topics'][topic]['topicName'])
    topic_keys.append(topic)

selected_topic_idx = st.sidebar.radio('Select a task', range(len(topic_names)), format_func=lambda i: topic_names[i])
     
# st.write(f"The index of the selected topic is: {selected_topic_idx}")

model_instruction = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["instruction"]["detailed"]
st.sidebar.write("Task: " + model_instruction)
model_id = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["modelId"]
model_parameters = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["parameters"]
model_input = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["inputPrefix"] 
model_output = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["outputPrefix"] 
model_examples = data[area_selected]['topics'][topic_keys[selected_topic_idx]]["examples"]

st.sidebar.markdown("**Model name**: &nbsp;" + model_id + " &nbsp;&nbsp;&nbsp; **Min tokens**: &nbsp;" + str(model_parameters["min_new_tokens"]) + " &nbsp;&nbsp;&nbsp; **Max tokens**: &nbsp;" + str(model_parameters["max_new_tokens"]))

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

selected_index = st.sidebar.radio("Select a case", range(len(options)), format_func=lambda i: options[i])

st.subheader('Text to be evaluated', divider='rainbow')
with st.expander("Input text", expanded=True):
    st.warning(data[area_selected]['inputs'][selected_index])
  
prompt_line = "\n"
prompt = model_instruction + prompt_line + prompt_line
for example in model_examples:
    prompt += model_input + prompt_line + example["input"] + prompt_line + prompt_line + model_output + prompt_line + example["output"] + prompt_line + prompt_line 
prompt += model_input + prompt_line 
prompt += data[area_selected]['inputs'][selected_index] + prompt_line + model_output + prompt_line

st.subheader('Constructed prompt', divider='rainbow')
with st.expander("Prompt"):
    prompt_display = prompt.replace('\n','\n\r')
    st.info(prompt_display)

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
        st.success(llm_result)    
    

