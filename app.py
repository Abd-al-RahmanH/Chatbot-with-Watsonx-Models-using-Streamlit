import os
from dotenv import load_dotenv
import streamlit as st

# from genai.credentials import Credentials
# from genai.schemas import GenerateParams
# from genai.model import Model

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM


load_dotenv()
# api_key = os.getenv("GENAI_KEY", None)
# api_endpoint = os.getenv("GENAI_API", None)

# creds = Credentials(api_key,api_endpoint)

# params = GenerateParams(
#     decoding_method="sample",
#     max_new_tokens=200,
#     min_new_tokens=1,
#     stream=False,
#     temperature=0.7,
#     top_k=50,
#     top_p=1,
#     stop_sequences= ["Human:","AI:"],
# )

api_key =   os.getenv("API_KEY")
project_id = os.getenv("PROJECT_ID")

creds = {
    "url"    : "https://us-south.ml.cloud.ibm.com",
    "apikey" : api_key
}

params = {
    GenParams.DECODING_METHOD:"sample",
    GenParams.MAX_NEW_TOKENS:2000,
    GenParams.MIN_NEW_TOKENS:1,
    GenParams.TEMPERATURE:0.7,
    GenParams.TOP_K:50,
    GenParams.TOP_P:1,
    GenParams.STOP_SEQUENCES: ["Human:","AI:"]
}

with st.sidebar:
    st.title("WATSONX CHAT")
    st.write("WATSONX.AI")
    st.write("RAHMAN")

st.title("CHAT WITH WATSONX")

with st.chat_message("system"):
    st.write("Hello 👋, lets chat with watsonx")

if "messages" not in st.session_state:
    st.session_state.messages = []

llm = Model(ModelTypes.CODELLAMA_34B_INSTRUCT_HF,creds,params,project_id)

#llm = Model(model="meta-llama/llama-3-405b-instruct",credentials=creds, params=params)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Say something"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    prompttemplate = f"""
    [INST]<<SYS>>Respond in English<<SYS>>
    {prompt}
    [/INST]
    """
    response_text = llm.generate_text(prompttemplate)
    answer = response_text
    # for response in response_text[0].generated_text
    #     answer += response[0].generated_text

    st.session_state.messages.append({"role": "agent", "content": answer}) 

    with st.chat_message("agent"):
        st.markdown(answer)
