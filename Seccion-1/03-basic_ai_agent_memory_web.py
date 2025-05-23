# Agente básico con interfaz wez
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from streamlit import session_state

#Cargar modelo de IA
llm = OllamaLLM(model="mistral") #Se puede cambiar de modelo

# Inicializar memoria
if "chat_history" not in st.session_state:
	st.session_state.chat_history = ChatMessageHistory()  #Permite que recuede conversaciones

# Crea un prompt template para formatear la entrada
promt = PromptTemplate(
	input_variables=["chat_history", "question"],
	template= "Conversación previa: {chat_history}\nUsuario: {question}\nAI:",
)
#Funcion para correr el modelo de IA con memoria
def run_chain(question):
    # recupera el historial de mensajes
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])
    #Corre la respuesta del modelo de IA
    response = llm.invoke(promt.format(chat_history=chat_history_text, question=question))
    # Guarda la pregunta y respuesta en el historial de mensajes
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)

    return response

#Streamlit
st.title("\n🤖 Bienvenido a tu chatbot de AIAgents")
st.write("Preguntame lo que necesitas")

user_input = st.text_input("👤 Tú : ")
if user_input:
    response = run_chain(user_input)
    st.write(f"**Tú** {user_input}")
    st.write(f"**AI:** {response}")

#Muestra historial del chat
st.subheader("📝 Historial del chat")
for msg in st.session_state.chat_history.messages:
    st.write(f"** {msg.type.capitalize()}**: {msg.content}")