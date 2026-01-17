# RAG-Embedding-LLM-Text-Generation

An end-to-end **Retrieval-Augmented Generation (RAG)** evaluation pipeline that combines semantic embeddings, vector similarity search, and large language models (LLMs) for question-answering tasks. This project uses Ollama with LLaMA models, sentence transformers for embeddings, and F1-score metrics for performance evaluation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Data Format](#data-format)
- [Workflow](#workflow)
- [Technical Details](#technical-details)

## 🎯 Overview

This project implements a complete RAG system that:

1. **Retrieves** relevant context passages from a knowledge corpus using semantic similarity
2. **Augments** LLM prompts with retrieved context and optional few-shot examples
3. **Generates** answers using locally hosted LLMs via Ollama
4. **Evaluates** generated answers against ground truth using F1-score metrics

The system is designed for factual question-answering tasks with temporal constraints (2018 knowledge cutoff) and supports both **RAG** and **Few-Shot** prompting strategies.

## ✨ Features

- **Semantic Retrieval**: Uses sentence transformers (`all-mpnet-base-v2`) to encode questions and corpus passages into dense vector embeddings
- **Hybrid Retrieval Strategy**: Combines top-K semantically similar passages with random sampling for diverse context
- **Multiple Prompting Modes**:
  - **RAG Mode**: Injects retrieved passages as external knowledge
  - **Few-Shot Mode**: Includes example Q&A pairs as style guides
  - **Combined Mode**: RAG + Few-Shot examples
- **Ollama Integration**: Connects to locally running Ollama API for LLM inference
- **GPU/CPU Support**: Automatically detects and utilizes GPU for embedding computation when available
- **Embedding Caching**: Saves computed embeddings to disk for faster subsequent runs
- **Incremental CSV Output**: Saves results progressively during execution
- **F1-Score Evaluation**: Calculates token-level F1 scores for answer quality assessment

## 🏗️ Architecture

```
┌─────────────────┐
│  Question Input │ (train.jsonl / test.jsonl)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Semantic Embedding (Question)      │
│  (Sentence Transformer)             │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Vector Similarity Search           │
│  (Cosine Similarity)                │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Top-K Passage Retrieval            │
│  + Random Sampling (Hybrid)         │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Prompt Construction                │
│  (Base Instructions + Context       │
│   + Optional Few-Shot Examples)     │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  LLM Generation (Ollama)            │
│  (LLaMA 3.1 / Gemma)                │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Answer Post-Processing             │
│  (Cleaning, Normalization)          │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  F1-Score Evaluation                │
│  (vs Ground Truth)                  │
└─────────────────────────────────────┘
```

## 📦 Requirements

### Software Dependencies

- **Python 3.8+**
- **Ollama** (locally installed and running)
- **LLM Models**: Requires one of the following Ollama models:
  - `llama3.1:8b`
  - `llama3-13b`
  - `gemma3:12b`
  - Or any compatible Ollama model

### Python Packages

```
requests >= 2.32.0
pandas >= 2.2.0
tqdm >= 4.65.0
sentence-transformers >= 2.2.0
numpy >= 1.26.0
torch >= 2.0.0 (for sentence-transformers)
```

## 🚀 Installation

### 1. Install Ollama

Visit [ollama.ai](https://ollama.ai) and follow the installation instructions for your operating system.

### 2. Pull Required LLM Models

```bash
ollama pull llama3.1:8b
# or
ollama pull gemma3:12b
```

### 3. Verify Ollama is Running

```bash
curl http://localhost:11434
```

You should see: `Ollama is running`

### 4. Install Python Dependencies

```bash
pip install requests pandas tqdm sentence-transformers numpy torch
```

Or using the provided requirements:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
RAG-Embedding-LLM-Text-Generation/
│
├── rag-llm.py                 # Main Python script (standalone version)
├── Notebook.ipynb             # Jupyter notebook (detailed version)
├── Notebook_def.ipynb         # Jupyter notebook (definitive version with hybrid retrieval)
│
├── train.jsonl                # Training questions with ground truth answers
├── test.jsonl                 # Test questions for evaluation
├── corpus.json                # Knowledge corpus (passages for RAG context)
│
├── rag.csv                    # Output: Generated answers (RAG mode)
├── rag5.csv                   # Output: Alternative results
├── rag_con_f1_score.csv       # Output: Results with F1 scores
│
└── README.md                  # This file
```

## 💻 Usage

### Method 1: Python Script

```bash
python rag-llm.py
```

The script will:
1. Load questions from `train.jsonl`
2. Load corpus from `corpus.json`
3. Generate embeddings for the corpus
4. Process each question through the RAG pipeline
5. Save results to `rag.csv`
6. Calculate and save F1 scores to `rag_con_f1_score.csv`

### Method 2: Jupyter Notebook

Open and run `Notebook_def.ipynb` for the most complete implementation with:
- Interactive execution
- Detailed explanations
- Step-by-step visualization
- Hybrid retrieval configuration

```bash
jupyter notebook Notebook_def.ipynb
```

### Method 3: Custom Script Execution

Modify the configuration variables at the top of `rag-llm.py` or in the notebook:

```python
MODEL = "llama3.1:8b"          # Ollama model to use
VARIANTE = "rag"                # Mode: "rag", "few-shot", or combined
ARCHIVO_JSONL = "train.jsonl"   # Input questions file
ARCHIVO_CORPUS = "corpus.json"  # Knowledge corpus file
```

## ⚙️ Configuration

### Key Configuration Parameters

#### Model Configuration
```python
MODEL = "llama3.1:8b"          # LLM model identifier in Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
```

#### Variant Selection
```python
VARIANTE = "rag"                # Generation mode
VARIANTE_FS = True              # Enable Few-Shot examples
NUM_FS_EXAMPLES = 3             # Number of few-shot examples
```

#### Retrieval Configuration (in Notebook_def.ipynb)
```python
K_SIMILAR = 25                  # Number of top similar passages
K_RANDOM = 3                    # Number of random passages (hybrid retrieval)
```

#### Generation Parameters
```python
options = {
    "temperature": 0.05         # Low temperature for factual consistency
}
```

### Prompt Configuration

The base prompt enforces:
- **Temporal constraint**: Answers reflect 2018 knowledge cutoff
- **Minimal verbosity**: Essential information only
- **No punctuation**: Answers without periods, commas, or semicolons
- **Full terms**: No abbreviations or acronyms
- **Complete components**: All essential parts of multi-component answers

## 📊 Evaluation Metrics

### F1-Score Calculation

The system uses **token-level F1-score** for evaluation:

1. **Normalization**: Removes articles, punctuation, converts to lowercase
2. **Token Matching**: Finds common tokens between generated and expected answers
3. **Best F1**: Selects highest F1 across multiple acceptable answers
4. **Micro-Averaged**: Calculates average F1 across all questions

**Formula**:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)

where:
Precision = common_tokens / generated_tokens
Recall = common_tokens / expected_tokens
```

### Output Metrics

- **Per-question F1**: Individual F1 score for each question
- **Average F1**: Macro-averaged F1 score across all questions
- **CSV Export**: Results saved with F1 scores for analysis

## 📄 Data Format

### Input Files

#### `train.jsonl` / `test.jsonl`
Each line is a JSON object:
```json
{
  "question": "where did the vietnam war mainly take place",
  "answer": ["Cambodia", "Vietnam", "Laos"]
}
```

#### `corpus.json`
Array of passage objects:
```json
[
  {
    "id": "064_001",
    "numero_pregunta": 64,
    "numero_pasaje": 1,
    "texto": "1815. The Battle of Waterloo was a very important battle..."
  }
]
```

### Output Files

#### `rag.csv`
```csv
número_pregunta,pregunta,salida_llm,nombre_variante
1,"where did...","Cambodia Vietnam Laos",rag
```

#### `rag_con_f1_score.csv`
Includes additional `f1_score_calculado` column with per-question F1 scores.

## 🔄 Workflow

### Step-by-Step Process

1. **Initialization**
   - Verify Ollama connection
   - Load configuration parameters

2. **Data Loading**
   - Load questions from JSONL file
   - Load corpus passages from JSON file
   - Load ground truth answers (for evaluation)

3. **Embedding Generation**
   - Encode all corpus passages using sentence transformer
   - Cache embeddings to disk (`.pt` file)
   - Reuse cached embeddings if available

4. **Question Processing Loop**
   For each question:
   - **Encoding**: Convert question to embedding vector
   - **Retrieval**: 
     - Find top-K semantically similar passages (cosine similarity)
     - Sample random passages (hybrid strategy)
     - Optionally select few-shot examples
   - **Prompt Construction**: Combine base instructions, context passages, and few-shot examples
   - **Generation**: Send prompt to Ollama API, receive streaming response
   - **Post-processing**: Clean and normalize generated answer
   - **Storage**: Append results to CSV file

5. **Evaluation**
   - Load generated answers from CSV
   - Match with ground truth using normalized questions
   - Calculate F1 scores per question
   - Compute average F1 score
   - Export results with metrics

## 🔧 Technical Details

### Semantic Embeddings

- **Model**: `sentence-transformers/all-mpnet-base-v2`
  - 768-dimensional embeddings
  - Optimized for semantic similarity
  - Supports GPU acceleration

### Retrieval Strategy

**Hybrid Approach**:
- **Semantic Retrieval**: Uses cosine similarity to find top-K most relevant passages
- **Random Sampling**: Adds diversity by including random passages
- **Deduplication**: Ensures no passage is selected twice

**Why Hybrid?**
- Semantic retrieval ensures relevance
- Random sampling introduces diversity and prevents overfitting to similar patterns

### Prompt Engineering

The RAG prompt structure:
```
[Base Instructions]
[Few-Shot Examples] (optional)
[External Knowledge: Retrieved Passages]
[Question]
```

This structure:
- Provides clear task instructions
- Shows expected format via examples
- Supplies relevant context
- Presents the specific question

### Performance Optimizations

1. **Embedding Caching**: Pre-computed embeddings saved to disk
2. **GPU Detection**: Automatic GPU utilization when available
3. **Incremental CSV Writing**: Results saved progressively (not all at end)
4. **Streaming API**: Ollama responses streamed for faster feedback
5. **Batch Encoding**: All corpus embeddings computed in single batch

## 🐛 Troubleshooting

### Common Issues

**Ollama Connection Error**
```
ERROR: Conexión fallida con Ollama
```
- Solution: Ensure Ollama is running (`ollama serve`)
- Verify URL: `http://localhost:11434`

**Model Not Found**
```
ERROR: HTTP 404. ¿Modelo 'llama3.1:8b' instalado?
```
- Solution: Pull the model: `ollama pull llama3.1:8b`

**Missing Corpus Key**
```
ERROR CRÍTICO: La clave 'texto' no se encontró en el corpus
```
- Solution: Check `corpus.json` structure. Passages should have `"texto"` or `"text"` field

**Memory Issues**
- Reduce `K_SIMILAR` value (fewer passages retrieved)
- Use CPU instead of GPU if GPU memory is limited
- Process questions in batches

## 📝 Notes

- The system is designed for **factual question-answering** with **2018 knowledge cutoff**
- Answers are **minimal and factual** (no explanations or greetings)
- The prompt enforces **no punctuation** in answers
- **Full names and complete terms** are required (no abbreviations)

## 🔮 Future Enhancements

Potential improvements:
- Support for multiple corpus sources
- Advanced re-ranking strategies
- Multi-hop reasoning support
- Integration with other LLM providers
- Fine-tuning of embedding models
- Web-based evaluation dashboard

## 👤 Author

**Moises Bustillo**

This project demonstrates a complete RAG evaluation pipeline for question-answering tasks using retrieval-augmented generation with semantic embeddings and local LLM inference.

---

For questions or issues, please open an issue on the repository.
