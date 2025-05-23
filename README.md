# Introducción a chatbots inteligentes en entornos locales

Este proyecto está protegido por copyright © 2025 Emilio Hernán Mayer - ehmsoft.  
Se autoriza su uso únicamente con fines educativos.  
Ver [LICENSE](LICENSE) para más detalles.


## Objetivos:
1. Entender los casos de uso de los agentes de IA
2. Aprender por qué Ollama es mejor para correr IA localmente
3. Instalar y configurar Ollama
4. Construir un agente básico usando Langchain y Ollama

## ¿Qué son los agentes de IA?
Son sistemas inteligentes que pueden percibir, razonar y actuar para lograr un objetivo de manera autónoma.

## ¿En qué se diferencian los agentes de IA de la automatización tradicional?

| **Características**           | **Automatización tradicional** | **Agentes de IA**                    |
|-------------------------------|--------------------------------|--------------------------------------|
| **Basados en reglas**          | Sí (si - entonces - sino)      | No (usa IA/ML)                       |
| **Manejo de tareas complejas** | No                             | Sí                                   |
| **Aprender con el tiempo**     | No                             | Sí (vía memoria y ciclos de feedback)|
| **Respuestas dinámicas**       | No (predefinidas)              | Sí (Generadas por IA)               |

## Ejemplos de Agentes de IA:
- **Chatbots**: Atención al cliente, asistentes personales
- **Web Scraper**: Investigación automatizada y extracción de datos de sitios web
- **Automatización de tareas con bots**: Respuestas automáticas a correos electrónicos
- **Sistemas multiagentes**: Agentes de IA que trabajan juntos

## ¿Por qué usar Ollama?
- **Corre localmente**: No hay costos de API y es completamente privado.
- **Admite modelos abiertos**: Llama 3, Mistral, Gemma, etc.
- **Optimizado para inferencia de LLM**: Tiempos de respuesta más rápidos que las APIs basadas en la nube.
- **Funciona sin Internet**: No requiere conexión a Internet.

## Instalación de dependencias

### Sección 1 (Agentes básicos y agentes con memoria)
- **langchain**: Proporciona componentes para combinar funcionalidades LLM.
- **langchain-community**: Adiciones de la comunidad.
- **langchain-ollama**: Ejecutar LLM localmente.

### Sección 2 (Asistente de voz)
- **speechrecognition**: Convierte palabras habladas en texto.
- **pyttsx3**: Convierte las respuestas de la IA en habla.

### Sección 3 (Agente de Web Scraping)
- **beautifulsoup4**: Extrae y limpia texto desde HTML.

### Sección 4 (Lector de documentos PDF y agente de preguntas y respuestas con IA)
- **pypdf**: Extrae el texto de archivos PDF.
- **faiss-cpu**: Almacena el contenido de documentos para una recuperación rápida.
- **langchain_huggingface**: Utiliza embedding para la búsqueda de texto.

## Instalar las dependencias desde `requirements.txt`

```bash
pip install -r requirements.txt
