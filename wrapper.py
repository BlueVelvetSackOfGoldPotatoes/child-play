# google cloud imports    
# import vertexai
# from vertexai.generative_models import GenerativeModel
# vertexai.init(project="PROJECT_ID", location="us-central1")

# Mistral imports
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage
# from mistral_api import key_mistral

# client = MistralClient(api_key=key_mistral)

# OpenAi imports
# from utils.gpt_api import key_openai
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

client = None

def ask(api_messages, temperature, model):
    global client
    origin, specific_model = model.split(':', 1)

    if origin == "oa":
        if client is None:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        params = {
            "model": specific_model,
            "messages": api_messages,
        }

        if specific_model.startswith(("o3", "o4")):
            if temperature <= 0.5:
                params["reasoning_effort"] = "low"
            elif temperature == 1:
                params["reasoning_effort"] = "medium"
            else:
                params["reasoning_effort"] = "high"
        else:
            params["temperature"] = temperature

        completion = client.chat.completions.create(**params)
        response = completion.choices[0].message.content

    # Hugging Face
    elif origin == "hf":
        from transformers import pipeline
        text_prompt = "\n\n".join([message["role"] + ":\n" + message["content"] for message in api_messages])
        pipe = pipeline("text-generation", model=specific_model)
        pipe_output = pipe(text_prompt)
        if pipe_output is None:
            raise RuntimeError("Hugging Face pipeline returned None")
        result = list(pipe_output)
        full_text = result[0]['generated_text']
        
        response = full_text[len(text_prompt):]
        
    elif origin == "ans":
        text_prompt = "\n\n".join([message["role"] + ":\n" + message["content"] for message in api_messages])
        response = specific_model
    
    # Mistral
    # elif origin == "mi":
    #     model = specific_model
    #     chat_response = client.chat(
    #         model=model,
    #         messages=[ChatMessage(role="user", content=api_messages)]
    #     )

    #     response = chat_response.choices[0].message.content

    # Google Cloud
    # elif origin == "gc":
    #     model = GenerativeModel(model_name="gemini-1.0-pro-002")

    #     model_response = model.generate_content(text_prompt)

    #     response = model_response.text

    else:
        raise ValueError(f"Unknown origin: {origin}")

    if response is not None:
        response = response.strip()

    return response

