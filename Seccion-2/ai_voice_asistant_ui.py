import streamlit as st
import speech_recognition as sr #Importa el modulo de reconociento de voz
import pyttsx3 #Convierte el texto a voz
from langchain_community.chat_message_histories import ChatMessageHistory #Mantiene el historial del ChatMessageHistory
from langchain_core.prompts import PromptTemplate #Define el formato de los prompts de ia
from langchain_ollama import OllamaLLM #Clase para usar un modelo LLM para respuestas de IsADirectoryError

#Cargar el modelo de ia
llm = OllamaLLM(model="mistral") 

#Inicializa la memoria(Langchain v1.0+)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory() #Almacena la conversacio usuario-IA en el historial

#Inicializar el motor de texto a voz
engine = pyttsx3.init()
engine.setProperty("rate", 160) #Ajusta la velocidad de habla 

#Inicializar el reconocimiento de voz
recognizer = sr.Recognizer() #Instancia para reconocer el habla en texto

#Funcion para hablar
def speak(text):
    engine.say(text)
    engine.runAndWait() #Procesa y pronuncia el texto

#Funcion para escuchar
def listen():
    with sr.Microphone() as source: # Utiliza el microfono como fuente de audio
        st.write("\n🎙️ Escuchando...")
        recognizer.adjust_for_ambient_noise(source, duration= 1) #Disminuye ruidos ambientales
        audio = recognizer.listen(source) #graba el discurso del usuario
    try:
        query = recognizer.recognize_google(audio, language="es-AR") #Utiliza la api de google para pasar el habla a texto
        st.write(f"Tu dices: {query}")
        return query.lower()
    except sr.UnknownValueError:
        st.write("🤖 Lo siento, no pude entender, intenta nuevamente! ")
        return ""
    except sr.RequestError:
        st.write("⚠️ Servicio de reconocimiento de voz fuera de servicio")
        return ""
    

#Formato de chat de IA
prompt = PromptTemplate (
        input_variables = ["chat_history", "question"],
        template = "Previous conversation: {chat_history}\nUser: {question}\nAI:"
    )

#Funcion para procesar las respuestas de la ia 
def run_chain(question):
    #Recupera el historial de chat pasado manualmente
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])

    #Generacionde respuestas de la ia
    response = llm.invoke(prompt.format(chat_history= chat_history_text, question= question))

    #Almaceno la respuesta de la ia y usuario en memoria
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    return response

#Stream Web UI
st.title("🤖 Asistente de voz web")
st.write("🎙️ Click en el boton para hablar con el asistente!")

#Grabando la entrada de audio
if st.button("🎙️Comenzar"):
    user_query = listen()
    if user_query:
        ai_response = run_chain(user_query)
        st.write(f"**Tú: {user_query}")
        st.write(f"**AI: {ai_response}")

#Muestra el historial de chat
st.subheader("Historial de chat")
for msg in st.session_state.chat_history.messages:
    st.write(f"** {msg.type.capitalize()} **: {msg.content}")