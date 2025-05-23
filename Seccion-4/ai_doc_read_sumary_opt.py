import streamlit as st
import faiss
import numpy as np
import PyPDF2
from langchain_ollama import OllamaLLM
# deprecate from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document





# Cargar el modelo
llm = OllamaLLM(model="mistral")

# Cargar embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name = "paraphrase-MiniLM-L3-v2")

# Inicializar índice y vector_store (vacío al inicio)
index = faiss.IndexFlatL2(384)
vector_store = {}

# Extraer texto desde un archivo PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Almacenar texto en FAISS
def store_in_faiss(text, filename):
    global index, vector_store

    # Reiniciar FAISS y vector_store cada vez que se carga un nuevo documento
    index = faiss.IndexFlatL2(384)
    vector_store = {}

    st.write(f"📦 Almacenando el documento '{filename}' en FAISS")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_text(text)
    vectors = embeddings.embed_documents(texts)
    vectors = np.array(vectors, dtype=np.float32)

    index.add(vectors)
    vector_store = {i: (filename, [texts[i]]) for i in range(len(texts))}

    return "✅ Documento guardado exitosamente"

#Funcion resumenes de documento generados con IA
def generate_sumary(text):
    global summary_text
    st.write("📝 Generado resumen...")
    summary_text = llm.invoke(f"Resume el siguiente documento, en español:\n\n{text[:3000]}")
    return summary_text

# Recuperar respuesta desde FAISS y LLM
def retrieve_and_answer(query):
    global index, vector_store

    query_vector = np.array(embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
    D, I = index.search(query_vector, k=2)

    context = ""
    for idx in I[0]:
        if idx in vector_store:
            context += " ".join(vector_store[idx][1]) + "\n\n"

    if not context:
        return "📦 No hay datos relevantes en los documentos almacenados"

    return llm.invoke(f"Basado en el siguiente contexto del documento, responde la pregunta en español:\n\n{context}\n\nPregunta:{query}")

# Funcion de descarga de archivo
def download_summary():
    if summary_text:
        st.download_button(
            label="⬇️ Descarga resumen",
            data=summary_text,
            file_name="Ai_Resumen.txt",
            mime= "text/plain"
        )

# Interfaz Streamlit
st.title("📄 Lector de documento con IA")
st.write("Subí un PDF, recibe un resumen y realizá preguntas sobre su contenido")

uploaded_file = st.file_uploader("🗃️ Subir documento PDF", type=["pdf"])
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    store_message = store_in_faiss(text, uploaded_file.name)
    st.write(store_message)

    # Genera el resumen con ia
    summary = generate_sumary(text)
    st.subheader("Resumen generado con IA")
    st.write(store_message)
    
    # Habilitar la descarga del resumen
    download_summary()

query = st.text_input("❓ Preguntá algo sobre el documento:")
if query:
    answer = retrieve_and_answer(query)
    st.subheader("🤖 Respuesta de la IA")
    st.write(answer)
