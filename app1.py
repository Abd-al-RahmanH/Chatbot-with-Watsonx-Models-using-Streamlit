import streamlit as st
import requests

def get_access_token(api_key):
    """Fetch a new access token from IBM Cloud using the API key."""
    token_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key,
    }

    response = requests.post(token_url, headers=headers, data=data)
    if response.status_code != 200:
        raise Exception(f"Failed to get access token: {response.text}")

    return response.json()["access_token"]

def call_ibm_api(access_token, project_id, model_id, user_input):
    """Make an API call to the IBM Cloud ML service."""
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

    body = {
        "input": user_input,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 900,
            "min_new_tokens": 0,
            "repetition_penalty": 1
        },
        "model_id": model_id,
        "project_id": project_id
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.post(url, headers=headers, json=body)

    if response.status_code != 200:
        raise Exception(f"Non-200 response from IBM API: {response.text}")

    return response.json()

def main():
    """Main function to run the Streamlit app."""
    st.title("Chat with Advanced Models")

    # Define your API key and project ID here
    api_key = "r6zSAPJm7t8GbkqJENPzmXPpOKokltDGcMREKRr5fWdh"  # Replace with your actual API key
    project_id = "833c9053-ef07-455e-819f-6557dea2f8bc"  # Replace with your actual project ID

    # Get access token securely
    access_token = get_access_token(api_key)

    # Model selection sidebar
    models = {
        'CODELLAMA_34B_INSTRUCT_HF': 'codellama/codellama-34b-instruct-hf',
        'FLAN_T5_XL': 'google/flan-t5-xl',
        'FLAN_T5_XXL': 'google/flan-t5-xxl',
        'FLAN_UL2': 'google/flan-ul2',
        'GRANITE_13B_CHAT_V2': 'ibm/granite-13b-chat-v2',
        'GRANITE_20B_CODE_INSTRUCT': 'ibm/granite-20b-code-instruct',
        'LLAMA_3_405B_INSTRUCT': 'meta-llama/llama-3-405b-instruct',
        'MISTRAL_LARGE': 'mistralai/mistral-large',
        'OPT_175B': 'facebook/opt-175b',
        'GPT_NEOX_20B': 'EleutherAI/gpt-neox-20b'
    }

    selected_model = st.sidebar.selectbox("Select a Model", list(models.keys()))

    # Input box for user query
    user_input = st.text_area("Your Query:", "")

    if st.button("Submit"):
        if user_input:
            try:
                # Call the selected model's API
                model_id = models[selected_model]
                response_data = call_ibm_api(access_token, project_id, model_id, user_input)
                
                # Display the response
                st.write("**Response:**")
                st.write(response_data)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
