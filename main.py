import torch
import torch.nn as nn
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig

from CustomModel import CustomModel
from utils import prepare_input

import streamlit as st
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# ======== Config and Paths ==========
class config:
    DEBUG = False
    GRADIENT_CHECKPOINTING = True
    MAX_LEN = 1024
    MODEL = "microsoft/deberta-v3-base"


class paths:
    MODEL_PATH = "DebertaV3Base"
    BEST_MODEL_PATH = "DebertaV3Base\microsoft_deberta-v3-base_fold_0_best.pth"

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(paths.MODEL_PATH + "/tokenizer")
# === Add special tokens ===
vocabulary = tokenizer.get_vocab()

# ======== MODEL ==========
model = CustomModel(config, config_path=paths.MODEL_PATH + "/config.pth", pretrained=False)
state = torch.load(paths.BEST_MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state)
model.to(device)
model.eval() # set model in evaluation mode

st.set_page_config(page_title='Detect AI Generated Text ', page_icon='images\SF_logo_icon.png', layout='wide', initial_sidebar_state='expanded')
st.sidebar.image('images\sideimage.png')

col1, col2 = st.columns([5, 1])
col1.title('Detect AI Generated Text')
col2.image('images\question.gif')
MAX_KEY_PHRASES = 800
with st.form("my_form"):
    text = st.text_area(
            label=" ",
            # The height
            height=200,
            # The tooltip displayed when the user hovers over the text area.
            help="Input text must less than or equal to "
            + str(MAX_KEY_PHRASES)
            + " keyphrases max in 'unlocked mode'. You can tweak 'MAX_KEY_PHRASES' in the code to change this",
            key="1",
            placeholder="Enter text here!",
        )
    
    submit_button = st.form_submit_button(label="Submit")

if not submit_button:
    st.stop()

elif submit_button and not text:
    st.warning("❄️ There is no text to classify")
    st.session_state.valid_inputs_received = False
    st.stop()

else:
    test_input = prepare_input(config, text, tokenizer)
    for k, v in test_input.items():
        test_input[k] = v.unsqueeze(0)

    with torch.no_grad():
        y_preds = torch.sigmoid(model(test_input))
        st.slider(f'0-Student 1-LLM',0.0, 1.0, y_preds.item(), disabled=True)



