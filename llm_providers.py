from abc import ABC, abstractmethod
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import openai
from google.generativeai import GenerativeModel
import google.generativeai as genai
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Initialize Jinja2 environment
jinja_env = Environment(
    loader=FileSystemLoader("prompts"),
    autoescape=select_autoescape()
)

PREDICTION_RANGES = {
    'fatigue': (1, 5, "fatigue level"),
    'stress': (1, 5, "stress level"),
    'readiness': (1, 5, "readiness level"),
    'sleep_quality': (1, 5, "sleep quality level")
}

class LLMProvider(ABC):
    @abstractmethod
    def generate_response(self, instruction: str, input_text: str = "", mode: str = "zero-shot", prediction_type: str = "sleep_quality") -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def _get_mode_prompt(self, mode: str, instruction: str, prediction_type: str) -> str:
        """Get the appropriate prompt template based on the mode and prediction type."""
        min_val, max_val, target_name = PREDICTION_RANGES[prediction_type]
        template_vars = {
            "instruction": instruction,
            "min_val": min_val,
            "max_val": max_val,
            "target_name": target_name
        }
        
        if mode == "zero-shot":
            template = jinja_env.get_template("base.jinja")
        elif mode == "few-shot":
            template = jinja_env.get_template("few_shot.jinja")
        elif mode == "few-shot_cot":
            template = jinja_env.get_template("few_shot_cot.jinja")
        elif mode == "few-shot_cot-sc":
            template = jinja_env.get_template("few_shot_cot_sc.jinja")
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return template.render(**template_vars)

class FineTunedProvider(LLMProvider):
    def __init__(self, model_path: str, base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        print(f"Using device: {device}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=device
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()

    def generate_response(self, instruction: str, input_text: str = "", mode: str = "zero-shot", prediction_type: str = "sleep_quality") -> str:
        prompt = f"### Instruction:\n{self._get_mode_prompt(mode, instruction, prediction_type)}\n\n"
        if input_text:
            prompt += f"### Input:\n{input_text}\n\n"
        prompt += "### Response:\n"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        return response

    @property
    def name(self) -> str:
        return "Fine-tuned Model"

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-o3-mini-high"):
        openai.api_key = api_key
        self.model = model

    def generate_response(self, instruction: str, input_text: str = "", mode: str = "zero-shot", prediction_type: str = "sleep_quality") -> str:
        messages = [{"role": "system", "content": "You are a helpful medical assistant that provides accurate and relevant information."}]
        
        prompt = self._get_mode_prompt(mode, instruction, prediction_type)
        if input_text:
            prompt += f"\nContext: {input_text}"
        
        messages.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        return response.choices[0].message.content

    @property
    def name(self) -> str:
        return "OpenAI (GPT-o3-mini-high)"

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = GenerativeModel("gemini-2.0-flash")

    def generate_response(self, instruction: str, input_text: str = "", mode: str = "zero-shot", prediction_type: str = "sleep_quality") -> str:
        prompt = self._get_mode_prompt(mode, instruction, prediction_type)
        if input_text:
            prompt += f"\nContext: {input_text}"
        
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def name(self) -> str:
        return "Google Gemini"

def create_llm_provider(provider_type: str, **kwargs) -> LLMProvider:
    """Factory function to create LLM providers"""
    if provider_type == "fine-tuned":
        return FineTunedProvider(**kwargs)
    elif provider_type == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_type == "gemini":
        return GeminiProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}") 