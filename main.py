from fastapi import FastAPI
from pydantic import BaseModel
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import requests
import os

# Charger les variables d'environnement
load_dotenv()
# Récupérer la clé API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return " ".join([p.text for p in soup.find_all("p")])

def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return " ".join([page.page_content for page in pages])

def create_vector_db(text):
    # Découper le texte en morceaux
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.create_documents([text])

    # Générer les embeddings
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(documents, embeddings)

    return vector_db

def chatbot_query(query):
    # Charger la base vectorielle
    vector_db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # Configurer le modèle LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # Créer un moteur de recherche intelligent
    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    # Obtenir la réponse
    return qa_chain.invoke(query)


# Exemple d'utilisation
#url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
#url = "https://www.britannica.com/technology/artificial-intelligence/Reasoning"
url = "https://www.mhway.fr/"
site_text = scrape_website(url)
#print(site_text[:500])  # Afficher un extrait

pdf_text = extract_text_from_pdf("datas/ai.pdf")
#print(pdf_text[:500])  # Afficher un extrait

vector_db = create_vector_db(site_text + pdf_text)
vector_db.save_local("faiss_index")

#question = "What is Artificial intelligence ?"
#response = chatbot_query(question)
#print(response)

app = FastAPI()

# Allow CORS (Modify origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific domains for security
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    response = chatbot_query(query.question)
    return {"response": response}
