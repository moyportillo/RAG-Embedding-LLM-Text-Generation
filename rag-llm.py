import requests
import json
import time
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import re
try:
    response = requests.get('http://localhost:11434')
    if response.status_code == 200 and 'Ollama is running' in response.text:
        print("Ollama está corriendo y accesible")
    else:
        print(f"Ollama respondió con estado {response.status_code}.")
except requests.exceptions.ConnectionError:
    print("Error de Conexión")

# Definimos los valores para el modo RAG (Retrieval-Augmented Generation)

# Archivos de entrada/salida
ARCHIVO_JSONL = "train.jsonl"             # Archivo de preguntas de PRUEBA
ARCHIVO_CORPUS = "corpus.json"        # Archivo del corpus de CONTEXTO RAG (JSON completo)
OLLAMA_URL = "http://localhost:11434/api/generate"

# Configuración de la variante
MODEL = "llama3.1:8b"
VARIANTE = "rag"                         
ARCHIVO_CSV_SALIDA = VARIANTE + ".csv"

# Prompt base sin contexto ni Few-Shot (se usa si no hay datos de contexto)
PROMPT_BASE = """You are an AI model highly specialized in factual information retrieval. You are operating within the temporal context of 2018. All answers provided must strictly reflect the knowledge, events, and state of affairs valid up to the end of that year.

Your sole mission is to address the user's question with the absolute minimum number of words possible, delivering only the essential and requested information. If ambiguity arises, assume the user's intended question and prioritize the most probable correct answer without acknowledging the error.

The strict instructions that you may follow are the next ones:

Responses must be based exclusively on information available before or during 2018.

When an error in the question is identified (e.g., misspelling, wrong movie number) but points to a primary, well-known entity, provide the correct, assumed information directly.

Your response must consist solely of the requested facts. Prohibit all greetings, introductions, explanations, notes, or any extraneous text.

No use of abbreviations, acronyms, or initialisms. Provide full names and complete terms.

No use of terms of puntuation like . , or ;

If the answer requires multiple components (e.g., names, locations, dates), you must provide all essential components in their complete form.

Context: Assume a general geographic or cultural context for interpretation."""

def build_prompt_rag(passages, question):
    """
    Construye un prompt de RAG, inyectando pasajes de contexto antes de la pregunta.
    """
    # Usamos el PROMPT_BASE como plantilla para las instrucciones
    prompt = PROMPT_BASE.replace(". Question: {question}:", "") # Quitamos la parte de la pregunta final
    
    prompt += "\n\nYou have to take the information of the answer from the following passages:\n\n"

    # Inyectar los pasajes de contexto
    for i, p in enumerate(passages):
        # Usamos .get("text", ...) para obtener el texto del pasaje
        text_content = p.get("text", p.get("texto", "Pasaje sin contenido")) # Si usa 'text' o 'texto'
        
        # Formatear el pasaje
        prompt += f"Passage {i + 1}:\n{text_content}\n---\n"

    prompt += (
                "\nNow answer the next question following the same style and using ONLY the provided information:\n"
                f"Question: {question}"
            )
    return prompt


# Mantengo la función few-shot original solo por si la necesitas más tarde
def build_prompt_fewshot(examples, question):
    # Esto es idéntico a tu función original, solo le cambié el nombre
    prompt = PROMPT_BASE.replace(". Question: {question}:", "")
    prompt += "    Here are examples of questions and their correct answers:\n\n"
    for ex in examples:
        q = ex["question"]
        a = ", ".join(ex["answer"]) if isinstance(ex["answer"], list) else ex["answer"]
        prompt += f"Q: {q}\nA: {a}\n\n"
    prompt += (
            "\nNow answer the next question following the same style:\n"
            f". Question: {question}: "
        )
    return prompt

def ask_ollama(question, context_data = None):
    """
    Consulta a Ollama y devuelve la respuesta completa o un mensaje de error.
    Ahora usa context_data que puede ser Few-Shot examples o RAG passages.
    """
    if VARIANTE == "few-shot" and context_data:
        prompt = build_prompt_fewshot(context_data, question)
    elif VARIANTE == "rag" and context_data:
        prompt = build_prompt_rag(context_data, question)
    else:
        # Prompt base si no hay contexto (se usa el PROMPT_BASE, que es PROMPT_TEMPLATE1 en tu código)
        prompt = PROMPT_BASE.format(question=question) # O usa PROMPT_TEMPLATE3 si quieres el pensamiento paso a paso
        
    print(prompt) # Descomenta para ver el prompt completo
    full_response = ""
    
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "options": { "temperature": 0.1 }
        }, timeout=120)

        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                part = json.loads(line)
                if "response" in part:
                    full_response += part["response"]
                if part.get("done"):
                    break
        
        return full_response.strip()
        
    except requests.exceptions.ConnectionError:
        return f"[ERROR: Conexión fallida con Ollama en {OLLAMA_URL}]"
    except requests.exceptions.HTTPError as e:
        return f"[ERROR: HTTP {e}. ¿Modelo '{MODEL}' instalado?]"
    except Exception as e:
        return f"[ERROR: Inesperado: {e}]"
    
ARCHIVO_CORPUS = "corpus.json" 

preguntas_cargadas = [] # Lista para las preguntas de PRUEBA
preguntas_test = []     # Lista para los pasajes de CONTEXTO RAG

try:
    # 1. Cargar DATOS DE PRUEBA (ARCHIVO_JSONL: Línea por línea)
    with open(ARCHIVO_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            preguntas_cargadas.append({
                "número_pregunta": i + 1,
                "pregunta": data.get("question", "")
            })
    
    print(f"✅ Se cargaron {len(preguntas_cargadas)} preguntas de prueba de '{ARCHIVO_JSONL}'.")

    # 2. Cargar DATOS DE CONTEXTO RAG (ARCHIVO_CORPUS: JSON COMPLETO)
    with open(ARCHIVO_CORPUS, "r", encoding="utf-8") as f:
        preguntas_test = json.load(f)
    
    # Verificamos que se haya cargado una lista de pasajes
    if isinstance(preguntas_test, list):
        print(f"✅ Se cargaron {len(preguntas_test)} pasajes (Contexto RAG) de \"{ARCHIVO_CORPUS}\".")
    else:
        print(f"⚠️ ADVERTENCIA: El archivo '{ARCHIVO_CORPUS}' fue cargado, pero no es una lista ({type(preguntas_test)}).")

except FileNotFoundError as e:
    print(f"❌ ERROR: El archivo '{e.filename}' no se encontró. ¡Verifica la ruta!")
    preguntas_cargadas = None
except json.JSONDecodeError as e:
    print(f"❌ ERROR: Fallo al decodificar JSON en '{ARCHIVO_CORPUS}'. Revisa que el archivo sea un JSON válido. Detalle: {e}")
    preguntas_cargadas = None


# --- Verificación de Carga RAG ---

if preguntas_cargadas and preguntas_test and isinstance(preguntas_test, list) and len(preguntas_test) > 0:
    print("\nPrimeras 3 preguntas de PRUEBA cargadas:")
    for item in preguntas_cargadas[:3]:
        print(f"  {item['número_pregunta']}: {item['pregunta'][:50]}...")
    
    ejemplo_completo = preguntas_test[0]
    print("\nVerificación de un ejemplo de CONTEXTO (Pasaje):")
    
    # Asumo la clave 'text' (preferido) o 'texto'
    texto_pasaje = ejemplo_completo.get('text', ejemplo_completo.get('texto', 'N/A'))
    
    print(f"  Pasaje ID: {ejemplo_completo.get('id', 'N/A')}")
    print(f"  Contenido (Inicio): {texto_pasaje[:100]}...")

# Encabezados (aunque no los vamos a escribir, los necesitamos para el DataFrame)
HEADERS = ["número_pregunta", "pregunta", "salida_llm", "nombre_variante"]

if preguntas_cargadas is None:
    print("No se puede ejecutar el bucle porque las preguntas no se cargaron correctamente.")
else:
    print(f"\nIniciando consulta al LLM con la variante RAG...")
    
    es_primera_escritura = True

    # model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    train_data = preguntas_test # El corpus de pasajes
    train_embeddings = None
    
    # --- 1. PREPARACIÓN DE EMBEDDINGS RAG ---
    if VARIANTE == "rag":
        print("entramos y codificamos embeddings del corpus de pasajes")
        # ¡IMPORTANTE! Asegúrate de que la clave 'text' existe en tu corpus.
        text_key = "texto" 
        if train_data and isinstance(train_data, list) and text_key in train_data[0]:
            train_texts = [ex.get(text_key, "") for ex in train_data]
            train_embeddings = model.encode(train_texts, convert_to_tensor=True)
        else:
            print(f"❌ ERROR CRÍTICO: La clave '{text_key}' no se encontró en el corpus para RAG.")
            train_embeddings = None
    
    
    # --- 2. BUCLE DE PROCESAMIENTO ---
    for item in tqdm(preguntas_cargadas, desc="Procesando Preguntas"):
        
        numero_pregunta = item["número_pregunta"]
        question = item["pregunta"]
        
        # Almacenará los pasajes recuperados
        context_data = None 

        if VARIANTE == "rag" and train_embeddings is not None:
            # ➡️ Lógica de Recuperación de Pasajes (RAG)
            test_embedding = model.encode(question, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(test_embedding, train_embeddings)[0]
            top_results = cos_scores.topk(3)
            
            # Los 3 pasajes completos recuperados (diccionarios con 'text', 'id', etc.)
            context_data = [train_data[idx] for idx in top_results[1]]
        
        
        # Se pasa la pregunta y los pasajes recuperados a ask_ollama
        salida_llm = ask_ollama(question, context_data=context_data)
        
        # ... (Resto de la lógica de guardado, sin cambios)
        salida_llm = re.sub(r'\s+', ' ', salida_llm).strip()
        
        df_fila = pd.DataFrame([{
            "número_pregunta": numero_pregunta,
            "pregunta": question,
            "salida_llm": salida_llm,
            "nombre_variante": VARIANTE
        }], columns=HEADERS)
        
        # Define el modo de escritura: 'w' (write/sobrescribir) para la primera, 'a' (append/añadir) después
        modo_escritura = 'w' if es_primera_escritura else 'a'
        
        try:
            # Crea el archivo con encabezados solo en la primera iteración si es necesario
            if es_primera_escritura:
                 pd.DataFrame(columns=HEADERS).to_csv(ARCHIVO_CSV_SALIDA, mode='w', header=True, index=False, encoding='utf-8')
                 modo_escritura = 'a'
            
            df_fila.to_csv(
                ARCHIVO_CSV_SALIDA,
                mode=modo_escritura, # Usa el modo dinámico
                header=False,        
                index=False,
                encoding='utf-8',
                lineterminator='\n'  
            )
            print(f"✅ P{numero_pregunta} guardada correctamente.")
            es_primera_escritura = False # Desactiva la bandera después de la primera escritura
        except Exception as e:
            print(f"\n❌ ERROR al guardar la pregunta {numero_pregunta} en CSV: {e}")
            
        time.sleep(0.1)

    print(f"\n✅ Evaluación finalizada. Revisa el archivo '{ARCHIVO_CSV_SALIDA}'.")

# 💡 NUEVA CELDA A: Definición de la Función de F1 Score
import collections
import string
import re
import numpy as np

def normalize_answer(s):
    """Quita artículos, puntuación y normaliza el texto a minúsculas."""
    def remove_articles(text):
        # Remueve 'a', 'an', 'the' al inicio o entre palabras
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        # Normaliza espacios en blanco
        return ' '.join(text.split())

    def remove_punc(text):
        # Remueve puntuación
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        # Convierte a minúsculas
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_f1_score(generated_answer, expected_answers_list):
    """
    Calcula el F1 Score de la respuesta generada contra una lista de respuestas correctas.
    Retorna el F1 score más alto encontrado.
    """
    if not expected_answers_list or expected_answers_list == ["N/A"]:
        return 0.0

    gen_tokens = normalize_answer(generated_answer).split()
    best_f1 = 0.0
    
    for expected in expected_answers_list:
        exp_tokens = normalize_answer(expected).split()
        
        if not gen_tokens and not exp_tokens:
            return 1.0

        common = collections.Counter(gen_tokens) & collections.Counter(exp_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            f1 = 0.0
        else:
            precision = num_common / len(gen_tokens)
            recall = num_common / len(exp_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
        
        if f1 > best_f1:
            best_f1 = f1

    return best_f1