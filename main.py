import sys
import os
sys.path.insert(0, '/content')

from src.trainer import train_lora_model
from src.rag_engine import setup_rag_system
from src.agent import AdvancedAIAgent

def main():
    knowledge_base = """
    Proje Bilgileri: Bu sistem LLM, LoRA fine-tuning, RAG ve otonom AI ajan bileşenlerini içeren hibrit bir yapay zeka mimarisidir.
    Teknoloji Stack: Python, PyTorch, Hugging Face Transformers, PEFT, LangChain, ChromaDB.
    Çalışma Ortamı: Google Colab T4 GPU üzerinde optimize edilmiştir.
    """
    
    model, tokenizer = train_lora_model()
    retriever = setup_rag_system(knowledge_base)
    agent = AdvancedAIAgent(model, tokenizer, retriever)
    
    query = "Bu projenin arkasındaki teknoloji stack nedir?"
    result = agent.run(query)
    
    print("\n" + "="*40)
    print("SONUÇ:")
    print(result)
    print("="*40)

if __name__ == "__main__":
    main()
