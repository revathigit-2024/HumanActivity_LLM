import os
import json
import torch
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)

def load_dataset(train_data):
    """
    Load and prepare the dataset for training.
    
    Args:
        train_data: Path to the training data file
    """
    if not os.path.exists(train_data):
        raise FileNotFoundError(f"Training file not found: {train_data}")
    
    # Load the dataset
    with open(train_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Format the data for training
    formatted_data = []
    for item in data:
        text = f"### Instruction:\n{item['instruction']}\n\n"
        if item['input']:
            text += f"### Input:\n{item['input']}\n\n"
        text += f"### Response:\n{item['output']}"
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def prepare_model_and_tokenizer(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cuda"):
    """Prepare the model and tokenizer for training."""
    print(f"Loading model {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the input texts."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

def train(args):
    """Main training function."""
    print(f"Starting training for {args.target}...")
    print(f"Training data: {args.train_data}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    full_dataset = load_dataset(args.train_data)
    
    # Split into train and validation
    train_test_split = full_dataset.train_test_split(
        test_size=args.validation_split, 
        shuffle=True, 
        seed=42
    )
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args.model_name, args.device)
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=False,
        load_best_model_at_end=True,
        optim="adamw_torch"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    print("Training completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a language model for health predictions")
    
    # Required arguments
    parser.add_argument("--target", type=str, required=True,
                      choices=['fatigue', 'stress', 'readiness', 'sleep_quality'],
                      help="Target prediction type")
    parser.add_argument("--train_data", type=str, required=True,
                      help="Path to training data JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save the model")
    
    # Optional arguments
    parser.add_argument("--model_name", type=str,
                      default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      help="Base model to fine-tune")
    parser.add_argument("--device", type=str, default="cuda",
                      choices=['cpu', 'cuda', 'mps'],
                      help="Device to use for training")
    parser.add_argument("--num_epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                      help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                      help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                      help="Learning rate")
    parser.add_argument("--validation_split", type=float, default=0.1,
                      help="Validation split ratio")
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main() 