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
class config1:
    DEBUG = False
    GRADIENT_CHECKPOINTING = True
    MAX_LEN = 512
    MODEL = "microsoft/codebert-base-mlm"

class config2:
    DEBUG = False
    GRADIENT_CHECKPOINTING = True
    MAX_LEN = 1024
    MODEL = "microsoft/deberta-v3-base"


class paths1:
    MODEL_PATH = "BertBase"
    BEST_MODEL_PATH = "BertBase\microsoft_codebert-base-mlm_fold_0_best.pth"

class paths2:
    MODEL_PATH = "DebertaV3Base"
    BEST_MODEL_PATH = "DebertaV3Base\microsoft_deberta-v3-base_fold_0_best.pth"


# === Load tokenizer ===
tokenizer1 = AutoTokenizer.from_pretrained(paths1.MODEL_PATH + "/tokenizer")
tokenizer2 = AutoTokenizer.from_pretrained(paths2.MODEL_PATH + "/tokenizer")
# === Add special tokens ===
vocabulary1 = tokenizer1.get_vocab()
vocabulary2 = tokenizer2.get_vocab()

# ======== MODEL ==========
model1 = CustomModel(config1, config_path=paths1.MODEL_PATH + "/config.pth", pretrained=False)
state1 = torch.load(paths1.BEST_MODEL_PATH, map_location=torch.device('cpu'))
model1.load_state_dict(state1)
model1.to(device)
model1.eval() # set model in evaluation mode

model2 = CustomModel(config2, config_path=paths2.MODEL_PATH + "/config.pth", pretrained=False)
state2 = torch.load(paths2.BEST_MODEL_PATH, map_location=torch.device('cpu'))
model2.load_state_dict(state2)
model2.to(device)
model2.eval() # set model in evaluation mode

st.set_page_config(page_title='Detect AI Generated Text ', page_icon='images\SF_logo_icon.png', layout='wide', initial_sidebar_state='expanded')
st.sidebar.image('images\sideimage.png')

col1, col2 = st.columns([5, 1])
col1.title('Detect AI Generated Text')
col2.image('images\question.gif')
MAX_KEY_PHRASES = 1000
with st.form("my_form"):
    text = st.text_area(
            label=" ",
            # The height
            height=100,
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
    test_input1 = prepare_input(config1, text, tokenizer1)
    test_input2 = prepare_input(config2, text, tokenizer2)

    for k, v in test_input1.items():
        test_input1[k] = v.unsqueeze(0)
    
    for k, v in test_input2.items():
        test_input2[k] = v.unsqueeze(0)

    with torch.no_grad():
        y_preds1 = torch.sigmoid(model1(test_input1))
        st.write("BertBase")
        st.slider(f'0-Student 1-LLM',0.0, 1.0, y_preds1.item(), disabled=True)

        y_preds2 = torch.sigmoid(model2(test_input2))
        st.write("DeBertaV3Base")
        st.slider(f'0-Student 1-LLM',0.0, 1.0, y_preds2.item(), disabled=True)



