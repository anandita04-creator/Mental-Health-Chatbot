import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr
from langchain_groq import ChatGroq

def initialize_llm():
    llm = ChatGroq(
        temperature=0.3,
        max_tokens=150,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )
    return llm

def create_vector_db():
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory {data_dir}. Please place your PDF files here.")
        
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        print("No PDF files found in ./data to create ChromaDB.")
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    print("ChromaDB created successfully!")
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    template = """You are a compassionate mental health chatbot.
Answer the user's question briefly (2-3 sentences), clearly, and in a friendly conversational tone.
Always try to end with a short follow-up question to keep the conversation going.

Context:
{context}

User: {question}
Chatbot:
"""
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain

print("Initializing Chatbot...")
llm = initialize_llm()
db_path = './chroma_db'

if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

if vector_db:
    qa_chain = setup_qa_chain(vector_db, llm)
else:
    qa_chain = None

def chatbot_response(user_input, history=[]):
    if not user_input.strip():
        return "Please provide a valid input"
    if not qa_chain:
        return "Vector database is not initialized. Please add some PDFs to ./data and restart."
    text_lower = user_input.strip().lower()
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if any(g in text_lower for g in greetings):
        return "Hey there 😊 How's your day going so far?"
    try:
        response = qa_chain.invoke(user_input)
        if not response.strip():
            response = "I'm here to help, but could you please ask a more specific question?"
    except Exception as e:
        response = f"Sorry, I couldn't process your request. ({str(e)})"
    return response

css_styling = """
body { background-color: #f0f7f4; } 
#character-img { background-color: #2f4f4f; max-width: 300px; margin: auto; padding: 10px; border-radius: 15px; height: 100%; object-fit: cover; box-shadow: 0 4px 12px rgba(0,0,0,0.1); } 
.chatbot-panel { background-color: #2f4f4f; border-radius: 15px; padding: 15px; height: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.1); display: flex; flex-direction: column; } 
.chatbot-panel > div { flex-grow: 1; }
"""

with gr.Blocks() as app:
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            if os.path.exists("./1.png"):
                gr.Image(value="./1.png", type="filepath", elem_id="character-img", show_label=False)
            else:
                gr.Image(elem_id="character-img", show_label=False)
        with gr.Column(scale=2, elem_classes=["chatbot-panel"]):
            gr.Markdown("## 🧠 Mental Health Chatbot 🤖")
            gr.Markdown("Your friendly assistant for mental well-being 💙")
            gr.ChatInterface(fn=chatbot_response, title="Mental Health Chatbot")
            gr.Markdown("⚠️ *This Chatbot provides general support. For urgent issues, seek help.*")

if __name__ == "__main__":
    app.launch(theme="gstaff/xkcd", css=css_styling)
