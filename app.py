import streamlit as st
import os
from dotenv import load_dotenv
from llm_providers import create_llm_provider, PREDICTION_RANGES

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="Health LLM Assistant",
    page_icon="🏥",
    layout="wide",
)

# Available prediction types and their models
PREDICTION_TYPES = {
    'fatigue': 'output/fatigue_model/final_model',
    'stress': 'output/stress_model/final_model',
    'readiness': 'output/readiness_model/final_model',
    'sleep_quality': 'output/sleep_quality_model/final_model'
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! How can I assist you with health predictions today?"}
    ]
if "current_model" not in st.session_state:
    st.session_state["current_model"] = "Fine-tuned Model"
if "current_mode" not in st.session_state:
    st.session_state["current_mode"] = "zero-shot"
if "current_prediction" not in st.session_state:
    st.session_state["current_prediction"] = "sleep_quality"
if "assistant" not in st.session_state:
    st.session_state["assistant"] = None

# Sidebar for model configuration
with st.sidebar:
    st.title("Model Configuration")
    
    # Model selection
    selected_model = st.selectbox(
        "Model", 
        ["Fine-tuned Model", "OpenAI", "Google Gemini"],
        index=["Fine-tuned Model", "OpenAI", "Google Gemini"].index(st.session_state["current_model"]),
        key="model_selector",
        help="Choose the model to use for predictions"
    )
    
    # Mode selection
    selected_mode = st.selectbox(
        "Mode",
        ["zero-shot", "few-shot", "few-shot_cot", "few-shot_cot-sc"],
        index=["zero-shot", "few-shot", "few-shot_cot", "few-shot_cot-sc"].index(st.session_state["current_mode"]),
        key="mode_selector",
        help="zero-shot: Direct prediction\nfew-shot: Uses examples\nfew-shot_cot: Chain of thought reasoning\nfew-shot_cot-sc: Chain of thought with self-checking"
    )
    
    # Prediction type selection
    selected_prediction = st.selectbox(
        "Prediction Type",
        list(PREDICTION_TYPES.keys()),
        index=list(PREDICTION_TYPES.keys()).index(st.session_state["current_prediction"]),
        key="prediction_selector",
        help="Select what to predict from the sensor data"
    )
    
    # Show prediction range
    min_val, max_val, target_name = PREDICTION_RANGES[selected_prediction]
    st.info(f"Predicting {target_name} (range: {min_val}-{max_val})")

# Load configuration from .env
API_KEYS = {
    "OpenAI": os.getenv("OPENAI_API_KEY"),
    "Google Gemini": os.getenv("GOOGLE_API_KEY"),
}
BASE_MODEL = os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Function to initialize the assistant
def initialize_assistant(model_type):
    try:
        kwargs = {}

        if model_type == "Fine-tuned Model":
            provider_type = "fine-tuned"
            kwargs = {
                "model_path": PREDICTION_TYPES[st.session_state["current_prediction"]],
                "base_model_name": BASE_MODEL,
                "device": os.getenv("DEVICE", None)  # Get from .env or let provider auto-detect
            }
        
        elif model_type in API_KEYS:
            if not API_KEYS[model_type] or API_KEYS[model_type] == "your-api-key":
                raise ValueError(f"⚠️ Please set your {model_type} API key in the .env file")
            
            if model_type == "OpenAI":
                provider_type = "openai"
                kwargs = {"api_key": API_KEYS[model_type], "model": "gpt-3.5-turbo"}
            else : #Google Gemini 
                provider_type = "gemini"
                kwargs = {"api_key": API_KEYS[model_type]}
        
        else:
            raise ValueError("Invalid model type selected.")
        
        return create_llm_provider(provider_type, **kwargs)

    except Exception as e:
        st.error(f"Error initializing the model: {str(e)}")
        return None

# Main chat interface
st.title("🏥 Health Prediction Assistant")

# Display current configuration
st.markdown(f"""
**Current Configuration:**
- Model: {selected_model}
- Mode: {selected_mode}
- Predicting: {target_name} (range: {min_val}-{max_val})
""")

# Chat container for message history
chat_container = st.container()

# Display chat history
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Handle model, mode, or prediction type switching
if (selected_model != st.session_state["current_model"] or 
    selected_mode != st.session_state["current_mode"] or 
    selected_prediction != st.session_state["current_prediction"]):
    
    st.session_state["current_model"] = selected_model
    st.session_state["current_mode"] = selected_mode
    st.session_state["current_prediction"] = selected_prediction
    st.session_state["assistant"] = initialize_assistant(selected_model)
    
    if st.session_state["assistant"]:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Switched to {selected_model} with {selected_mode} mode for {target_name} prediction. How can I assist you?"
        })
    st.rerun()

# Initialize assistant if not already done
if st.session_state["assistant"] is None:
    st.session_state["assistant"] = initialize_assistant(st.session_state["current_model"])

# Chat input
prompt = st.chat_input(f"Enter sensor data to predict {target_name} (range: {min_val}-{max_val})...")

# Handle chat input
if prompt:
    if not st.session_state["assistant"]:
        st.error("⚠️ No model is currently initialized. Please check model configuration.")
        st.stop()

    # Add user message to chat
    with chat_container:
        st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response
    try:
        response = st.session_state["assistant"].generate_response(
            prompt, 
            mode=st.session_state["current_mode"],
            prediction_type=st.session_state["current_prediction"]
        )
        with chat_container:
            with st.chat_message("assistant"):
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"⚠️ Error generating response: {str(e)}")