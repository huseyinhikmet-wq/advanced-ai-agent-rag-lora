import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)

def train_lora_model(model_id="Qwen/Qwen2.5-1.5B-Instruct", output_dir="./lora_model"):
    print(">> LoRA Eğitimi Başlatılıyor...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    raw_data = {
        "text": [
            "Soru: AI Ajanı nedir?\nCevap: AI ajanı, çevresini algılayan, kararlar alan ve hedeflerine ulaşmak için araçlar kullanan otonom bir yapay zeka sistemidir.",
            "Soru: RAG mimarisi ne işe yarar?\nCevap: RAG, harici bilgi kaynaklarından veri çekerek modelin güncel ve doğru yanıtlar üretmesini sağlar."
        ]
    }
    dataset = Dataset.from_dict(raw_data)
    
    def tokenize_function(examples):
        tokens = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        num_train_epochs=3,
        fp16=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(">> LoRA Eğitimi Tamamlandı ve Model Kaydedildi.")
    return model, tokenizer
