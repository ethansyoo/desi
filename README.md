# chatDESI: An AI-Powered Astronomical Research Assistant

**Live Application Link: [https://askdesi.streamlit.app/](https://askdesi.streamlit.app/)**

---

## 🚀 Overview

**chatDESI** is a sophisticated, multi-model conversational AI designed to assist with astronomical research. It provides a powerful, intuitive interface for interacting with a private library of scientific documents and can generate complex Astronomical Data Query Language (ADQL) queries from natural language prompts.

This application is built with Streamlit and leverages state-of-the-art language models from OpenAI and Anthropic, backed by a robust MongoDB vector database for efficient document retrieval.

## ✨ Key Features

* **Conversational Q&A:** Ask complex questions about your private research papers and get accurate, context-aware answers.
* **Intelligent Document Search:** The system uses a hybrid search strategy to find the most relevant information, whether you're asking about a specific paper or a general concept.
* **ADQL Query Generation:** Describe your data needs in plain English and let the AI generate the precise ADQL query for you.
* **Multi-Model Support:** Seamlessly switch between top-tier AI models from OpenAI (GPT-4o, GPT-3.5) and Anthropic (Claude 3.5 Sonnet, Claude 3 Haiku) to find the best one for your task.
* **Admin Dashboard:** A secure, password-protected admin panel allows for easy uploading and management of the document library.

## 🛠️ Getting Started: Running Locally

Follow these steps to set up and run the chatDESI application on your own machine.

### 1. Prerequisites

Ensure you have the following installed:
* Python (3.9 or higher)
* Git

### 2. Initial Setup

First, clone the repository and set up the Python virtual environment.

```bash
# Clone the repository
git clone [https://github.com/ethansyoo/desi.git](https://github.com/ethansyoo/desi.git)
cd desi

# Create and activate a virtual environment
python3 -m venv chatdesi_venv
source chatdesi_venv/bin/activate