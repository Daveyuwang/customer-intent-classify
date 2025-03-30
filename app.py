"""
Gradio web interface for customer intent recognition.
"""

import os
import torch
import gradio as gr
from transformers import AutoTokenizer
from openai import OpenAI

# Import configuration
from config import MAX_LEN, N_CLASS, DEVICE, MODEL_NAME, INTENT_CLASSES

# Import models
from models.bert_model import BertModel
from models.cnn_model import TextCNN
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.combined_model import BERT_CNN_LSTM

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def model_prediction(text, model_type):
    """
    Handle local model prediction for the given text.
    
    Args:
        text (str): The input text to classify
        model_type (str): The model type to use for prediction
        
    Returns:
        tuple: Probabilities, predicted class, and confidence score
    """
    if not text or text.strip() == "":
        return ({"No input provided": 1.0}, "Please enter a query", "N/A")
    
    model_map = {
        "LSTM+CNN+BERT": "combined",
        "CNN": "cnn",
        "LSTM": "lstm",
        "Transformer": "transformer",
        "BERT": "bert"
    }

    model_name = model_map.get(model_type, "combined")
    return predict_intent(text, model_name)

def llm_prediction(text, use_llm_flag):
    """
    Handle LLM prediction for the given text.
    
    Args:
        text (str): The input text to classify
        use_llm_flag (bool): Whether to use LLM for prediction
        
    Returns:
        tuple: Predicted class and confidence score
    """
    if not use_llm_flag or not text or text.strip() == "":
        return "", ""
    
    try:
        # Get API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "API key not configured", "N/A"
            
        llm_label, llm_conf_val = llm_predict(text, api_key)
        return llm_label, f"{llm_conf_val:.4f}"
    except Exception as e:
        print(f"Error in LLM prediction: {e}")
        return "Error in LLM prediction", "N/A"

def predict_intent(text, model_type="bert"):
    """
    Predict intent for the given text using the specified model.
    
    Args:
        text (str): The input text to classify
        model_type (str): The type of model to use
        
    Returns:
        tuple: Probabilities, predicted class, and confidence score
    """
    # Tokenize the input text
    inputs = tokenizer(
        text, 
        padding='max_length', 
        max_length=MAX_LEN, 
        truncation=True, 
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(DEVICE)

    # Choose model
    if model_type == "bert":
        model = bert_model
    elif model_type == "cnn":
        model = cnn_model
    elif model_type == "lstm":
        model = lstm_model
    elif model_type == "transformer":
        model = transformer_model
    elif model_type == "combined":
        model = combined_model
    else:
        model = bert_model  # default

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)  # (batch_size, n_class=11)

    # Softmax probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    pred_index = torch.argmax(probabilities, dim=1).item()
    pred_probability = probabilities[0][pred_index].item()
    pred_class = INTENT_CLASSES[pred_index]

    # Create dict of probabilities per class
    all_probs = probabilities[0].cpu().numpy()
    results = {
        INTENT_CLASSES[i]: float(all_probs[i])
        for i in range(len(INTENT_CLASSES))
    }

    return results, pred_class, f"{pred_probability:.4f}"

def llm_predict(text, api_key):
    """
    Predict intent using DeepSeek LLM API.
    
    Args:
        text (str): The text to classify
        api_key (str): The API key for DeepSeek
        
    Returns:
        tuple: Predicted class name and confidence score
    """
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    system_prompt = (
        "You are a helpful assistant that classifies user queries into one of "
        f"the following {len(INTENT_CLASSES)} categories:\n"
        + ", ".join(INTENT_CLASSES)
        + "\n\n"
        "Your task is to output ONLY the predicted class in the following strict format:\n"
        "'CLASS_NAME' with confidence=0.XXXX\n\n"
        "No additional text, no explanation."
    )
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )
    
    llm_output = response.choices[0].message.content.strip()
    class_name, confidence = llm_output.split(" with confidence=")
    confidence = float(confidence)
    
    return class_name, confidence

def load_models():
    """
    Load all trained models.
    
    Returns:
        dict: Dictionary of trained models
    """
    # Instantiate each model
    bert = BertModel()
    cnn = TextCNN()
    lstm = LSTMModel()
    transformer = TransformerModel()
    combined = BERT_CNN_LSTM()

    models_dict = {}

    # Load BERT weights
    try:
        bert.load_state_dict(torch.load('bert_intent_model.pth', map_location=DEVICE))
        models_dict["bert"] = bert
        print("BERT model loaded successfully!")
    except Exception as e:
        print(f"Could not load BERT model weights: {e}")
        models_dict["bert"] = bert

    # Load CNN weights
    try:
        cnn.load_state_dict(torch.load('cnn_intent_model.pth', map_location=DEVICE))
        models_dict["cnn"] = cnn
        print("CNN model loaded successfully!")
    except Exception as e:
        print(f"Could not load CNN model weights: {e}")
        models_dict["cnn"] = cnn

    # Load LSTM weights
    try:
        lstm.load_state_dict(torch.load('lstm_intent_model.pth', map_location=DEVICE))
        models_dict["lstm"] = lstm
        print("LSTM model loaded successfully!")
    except Exception as e:
        print(f"Could not load LSTM model weights: {e}")
        models_dict["lstm"] = lstm

    # Load Transformer weights
    try:
        transformer.load_state_dict(torch.load('transformer_intent_model.pth', map_location=DEVICE))
        models_dict["transformer"] = transformer
        print("Transformer model loaded successfully!")
    except Exception as e:
        print(f"Could not load Transformer model weights: {e}")
        models_dict["transformer"] = transformer

    # Load Combined model weights
    try:
        combined.load_state_dict(torch.load('combined_intent_model.pth', map_location=DEVICE))
        models_dict["combined"] = combined
        print("Combined model loaded successfully!")
    except Exception as e:
        print(f"Could not load combined model: {e}")
        models_dict["combined"] = combined

    return models_dict

# Initialize models
models = load_models()
bert_model = models.get("bert")
cnn_model = models.get("cnn")
lstm_model = models.get("lstm")
transformer_model = models.get("transformer")
combined_model = models.get("combined")

# Build the Gradio interface
with gr.Blocks(title="Customer Intent Recognition") as demo:
    gr.Markdown("## Customer Intent Recognition")
    gr.Markdown(
        "Enter a customer query or request, and the system will automatically "
        "identify the intent from the categories below."
    )

    with gr.Row():
        # Left column - Input and model selection
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Customer Query", 
                placeholder="Enter your query here...", 
                lines=3
            )
            
            # Model selection area
            model_choice = gr.Radio(
                choices=["LSTM+CNN+BERT", "CNN", "LSTM", "Transformer", "BERT"],
                value="LSTM+CNN+BERT",
                label="Model Selection"
            )
            
            # LLM prediction option
            use_llm = gr.Checkbox(
                label="Use LLM for prediction", 
                value=True,
                info="Powered by DeepSeek Chat"
            )
            
            # Submit button - positioned above LLM prediction area
            submit_button = gr.Button(
                "Predict Intent", 
                size="lg",
                elem_id="orange-button"
            )
            
            # LLM prediction results area
            gr.Markdown("### LLM Prediction")
            llm_label = gr.Textbox(label="LLM Predicted Intent")
            llm_conf = gr.Textbox(label="LLM Confidence Score")

        # Right column - Model prediction results
        with gr.Column(scale=1):
            gr.Markdown("## Model Prediction")
            prediction_label = gr.Textbox(label="Predicted Intent")
            confidence_score = gr.Textbox(label="Confidence Score")
            all_probabilities = gr.Label(label="All Intent Probabilities")
    
    # Add CSS style for the orange button
    gr.HTML("""
    <style>
    #orange-button {
        background-color: #FF7F50 !important;
        color: white !important;
    }
    </style>
    """)
    
    # Recognizable intent categories (collapsed by default)
    with gr.Accordion("Recognizable Intent Categories", open=False):
        gr.Markdown("\n".join([f"- {cls}" for cls in INTENT_CLASSES]))
    
    # Bind event handlers
    submit_button.click(
        fn=model_prediction,
        inputs=[text_input, model_choice],
        outputs=[all_probabilities, prediction_label, confidence_score]
    )
    submit_button.click(
        fn=llm_prediction,
        inputs=[text_input, use_llm],
        outputs=[llm_label, llm_conf]
    )

    # Example inputs
    gr.Markdown("## Example Inputs")
    examples = [
        ["Why am I being charged a cancellation fee?", "LSTM+CNN+BERT"],
        ["Please unsubscribe me from your newsletter", "LSTM+CNN+BERT"],
        ["I need to update my account information", "CNN"],
        ["Where is my order? I ordered last week", "CNN"],
        ["I want to request a refund for my purchase", "LSTM"],
        ["I need to change my shipping address", "LSTM"],
        ["When will my package be delivered?", "Transformer"],
        ["I haven't received my invoice yet", "Transformer"],
        ["I want to provide feedback about your service", "BERT"],
        ["I'm having trouble with my payment method", "BERT"]
    ]
    gr.Examples(examples, inputs=[text_input, model_choice])

if __name__ == "__main__":
    demo.launch()