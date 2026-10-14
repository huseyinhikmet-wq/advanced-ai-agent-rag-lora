class AdvancedAIAgent:
    def __init__(self, model, tokenizer, retriever):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        
    def run(self, query):
        print(f">> Ajan Girdiyi İşliyor: '{query}'")
        relevant_docs = self.retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""Bağlam:
{context}

Soru: {query}
Yanıt:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.7, 
            top_p=0.9,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Yanıt:")[-1].strip()
        return answer
