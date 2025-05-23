import os
import numpy as np
import requests
import pickle
from bs4 import BeautifulSoup
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import faiss

# ==== CACHÉ DE MODELOS ====
@st.cache_resource
def load_llm():
    return Ollama(model="mistral")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = load_llm()
embeddings = load_embeddings()

# ==== FAISS INDEX Y VECTOR_STORE ====
INDEX_FILE = "faiss_index.index"
STORE_FILE = "vector_store.pkl"
DIMENSION = 384  # Embedding dimension

# Cargar índice FAISS
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(DIMENSION)

# Cargar vector_store
if os.path.exists(STORE_FILE):
    with open(STORE_FILE, "rb") as f:
        vector_store = pickle.load(f)
else:
    vector_store = {}

# ==== SCRAPER ====
def scrape_website(url):
    st.write(f"🌎 Scrapeando sitio: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return f"⚠️ Error al obtener la URL: {url}"
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        return text[:3000]  # Limita el tamaño para mayor rendimiento
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==== ALMACENAMIENTO ====
def store_in_faiss(text, url):
    global index, vector_store

    if url in [info[0] for info in vector_store.values()]:
        return "⚠️ Esta URL ya fue almacenada previamente."

    st.write("💾 Almacenando datos en FAISS...")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)

    vectors = np.array(embeddings.embed_documents(texts=chunks), dtype=np.float32)
    index.add(vectors)

    vector_store[len(vector_store)] = (url, chunks)

    # Guardar persistencia
    faiss.write_index(index, INDEX_FILE)
    with open(STORE_FILE, "wb") as f:
        pickle.dump(vector_store, f)

    return "✅ Datos almacenados correctamente."

# ==== PREGUNTAS ====
def retrieve_and_answer(query):
    query_vector = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
    D, I = index.search(query_vector, k=3)

    context = ""
    for idx in I[0]:
        if idx in vector_store:
            context += " ".join(vector_store[idx][1]) + "\n\n"

    if not context:
        return "⚠️ No se encontraron datos relevantes."
    
    return llm.invoke(f"Basado en el siguiente contexto, responde a la pregunta:\n\n{context}\n\nPregunta: {query}\nRespuesta:")

# ==== INTERFAZ STREAMLIT ====
st.title("🤖 Web Scraper impulsado por IA")
st.write("🌐 Ingresa una URL y realiza preguntas sobre su contenido.")

# Entrada de URL
url = st.text_input("🔗 Ingresa la URL del sitio web:")
if url:
    content = scrape_website(url)
    if "⚠️" in content or "❌" in content:
        st.error(content)
    else:
        st.success("✅ Contenido extraído correctamente.")
        result = store_in_faiss(content, url)
        st.info(result)

# Entrada de preguntas
query = st.text_input("❓ Haz una pregunta sobre el contenido:")
if query:
    with st.spinner("Buscando respuesta..."):
        response = retrieve_and_answer(query)
        st.subheader("🤖 Respuesta de IA:")
        st.write(response)
