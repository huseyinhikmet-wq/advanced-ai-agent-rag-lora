# Advanced LLM + RAG + LoRA AI Agent

This project implements a modular, lightweight, and efficient AI agent architecture combining **LoRA Fine-Tuning**, **RAG (Retrieval-Augmented Generation)**, and custom agent logic optimized for Google Colab T4 GPUs.

## Tech Stack
- **Python & PyTorch**
- **Hugging Face Transformers & PEFT (LoRA)**
- **BitsAndBytes (4-bit Quantization)**
- **LangChain & ChromaDB**

## Project Structure
```text
advanced-ai-agent-rag-lora/
│
├── data/                  # Knowledge base documents
├── src/
│   ├── __init__.py
│   ├── trainer.py         # LoRA Fine-tuning module
│   ├── rag_engine.py      # ChromaDB retriever module
│   └── agent.py           # AI Agent executor logic
│
├── main.py                # Main pipeline script
└── requirements.txt       # Dependencies
```
