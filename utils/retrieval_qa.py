import os
from groq import Groq
from .chunk_embeddings import HuggingFaceEmbeddings

MODEL_NAME = "llama-3.3-70b-versatile"

class RetrievalQA:
    def __init__(self, vectordb):
        """
        Initialize the Retrieval QA system
        
        Args:
            vectordb: QdrantDB instance for vector searches
        """
        self.vdb = vectordb
        self.embeddings = HuggingFaceEmbeddings()
        
        # Initialize Groq client
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        self.client = Groq(api_key=groq_api_key)

    def ask(self, query: str, top_k: int = 6, max_tokens: int = 1200, temperature: float = 0.2):
        """
        Ask a question and get an answer based on the indexed documents
        
        Args:
            query: The question to ask
            top_k: Number of relevant documents to retrieve
            max_tokens: Maximum tokens for the response
            temperature: Temperature for response generation
            
        Returns:
            Tuple of (answer, relevant_documents)
        """
        try:
            # 1) Encode the query
            print(f"Processing query: {query[:100]}...")
            q_emb = self.embeddings.encode([query], normalize=True, show_progress=False)
            
            # 2) Retrieve relevant documents
            print("Searching for relevant documents...")
            hits = self.vdb.search(q_emb, top_k=top_k)
            
            # 3) Build context from hits
            if not hits:
                context = "No relevant context found in the indexed Indian legal documents."
                print("No relevant documents found")
            else:
                context_parts = []
                for h in hits:
                    source = h['payload'].get('source', 'Unknown')
                    page = h['payload'].get('page', 'Unknown')
                    text = h['text'][:1000]  # Limit text length
                    context_parts.append(f"[Source: {source}, Page: {page}]\n{text}")
                
                context = "\n\n---\n\n".join(context_parts)
                print(f"Found {len(hits)} documents (scores: {[format(h.get('score', 0), '.1f') for h in hits[:3]]})")


            # 4) Create enhanced prompts
            system_prompt = """You are an expert Indian legal assistant specializing in Indian law with comprehensive knowledge of various Indian legal acts and regulations. Your role is to provide accurate, well-structured, and professional legal information based strictly on the provided context from official Indian legal documents.

**CORE RESPONSIBILITIES:**
- Provide accurate legal information based on Indian law
- Structure responses professionally with proper formatting
- Cite specific legal provisions, sections, and acts
- Maintain legal precision while ensuring accessibility

**RESPONSE STRUCTURE REQUIREMENTS:**
1. **Clear Introduction**: Brief overview of the legal topic
2. **Main Legal Analysis**: Detailed explanation with proper citations
3. **Relevant Provisions**: Specific sections and acts mentioned
4. **Practical Implications**: Real-world application if applicable
5. **Conclusion**: Summary of key legal points

**FORMATTING GUIDELINES:**
- Use ## for main sections
- Use ### for subsections  
- Use **bold** for important legal terms and provisions
- Use numbered lists for sequential procedures
- Use bullet points for related concepts
- Always cite: **Section X of Act Name, Year**
- Separate concepts with proper line spacing

**CITATION REQUIREMENTS:**
- Always reference specific sections: "Section 123 of the Indian Penal Code, 1860"
- Mention relevant acts: "As per the Consumer Protection Act, 2019"
- Include case references if mentioned in context
- Quote exact legal language when appropriate

**IMPORTANT CONSTRAINTS:**
- Base answers STRICTLY on provided context
- If information is incomplete, clearly state limitations
- Never provide general legal advice without contextual basis
- Always mention if specific details are not available in the source material"""

            user_prompt = f"""**Legal Query:** {query}

Please provide a comprehensive, professionally structured answer based on the following authenticated Indian legal document excerpts:

**CONTEXT FROM OFFICIAL LEGAL DOCUMENTS:**
{context}

**INSTRUCTIONS:**
1. Analyze the query in light of the provided legal context
2. Structure your response with clear headings and proper legal citations
3. Include all relevant legal provisions and sections mentioned
4. Ensure professional formatting with proper line breaks and emphasis
5. If the context doesn't fully answer the query, clearly indicate what information is missing

Provide a detailed, well-formatted legal analysis that would be suitable for legal professionals."""

            # 5) Generate response using Groq
            print("Generating response with Groq LLM...")
            completion = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = completion.choices[0].message.content if completion.choices else "Sorry, I couldn't generate a response."
            
            print("Response generated successfully")
            return answer, hits

        except Exception as e:
            print(f"Error in ask method: {e}")
            error_answer = f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again or rephrase your question."
            return error_answer, []