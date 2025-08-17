# Offboarding & Handover Document

This document provides a guide for a new developer taking over the chatDESI project. The goal is to ensure a smooth transition, assuming no prior knowledge of the codebase.

### Project Overview

* **Purpose:** chatDESI is a conversational AI application designed to answer questions about a private collection of scientific papers. It also includes a specialized mode for generating Astronomical Data Query Language (ADQL) queries from natural language.
* **Core Technologies:**
    * **Backend:** Python
    * **Frontend:** Streamlit
    * **Database:** MongoDB (for storing PDF text chunks and embeddings)
    * **AI Models:** OpenAI and Anthropic for language understanding; Sentence-Transformers for text embeddings.

### High-Level Architecture

The application is divided into several logical modules:

1.  **`main.py`:** The main entry point that initializes everything and controls the overall application flow.
2.  **`ui` Module:** Contains all the Streamlit components. `chat_interface.py` and `adql_interface.py` define the two main modes of the app.
3.  **`data` Module:** Handles all data-related operations.
    * `pdf_manager.py` is the most important file here. It manages chunking PDFs, generating embeddings, and searching for relevant documents in the database.
    * `database.py` manages the connection to MongoDB.
4.  **`auth` Module:** Manages API keys and the creation of AI clients for different providers (OpenAI, Anthropic).
5.  **`config` Module:** Contains all the application settings, such as model names, database configuration, and UI defaults.

### How a Chat Query Works (Step-by-Step)

1.  A user enters a message in the Streamlit UI (`chat_interface.py`).
2.  The `_handle_new_message` function is called.
3.  It calls `pdf_manager.find_relevant_docs()` to search the MongoDB database for relevant text chunks.
    * This function first checks if the user's query mentions a specific filename. If so, it retrieves a sample of chunks from that document.
    * If not, it performs a vector search to find chunks that are semantically similar to the query.
4.  The retrieved chunks are passed as context to the AI model (`_generate_chat_response`).
5.  The AI model (e.g., GPT-4o or Claude 3.5 Sonnet) generates a response based on the context.
6.  The response is streamed back to the user's screen.

### Key Credentials & Setup

* All sensitive information (API keys, database strings, passwords) is stored in the `.streamlit/secrets.toml` file. This file is **never** committed to Git.
* To get started, you will need to create your own `secrets.toml` file and populate it with:
    * A MongoDB connection string (from a free Atlas account).
    * An admin password of your choosing.
    * API keys for OpenAI and/or Anthropic.

### Getting Started with Development

1.  **Follow the User Manual:** The best way to start is to follow the `SETUP_GUIDE.md` to get a local version of the application running.
2.  **Key Files to Understand First:**
    * `chatdesi/main.py`: To see how all the pieces are connected.
    * `chatdesi/ui/chat_interface.py`: To understand the main user interaction loop.
    * `chatdesi/data/pdf_manager.py`: To understand the core retrieval logic.
3.  **How to Debug:**
    * The easiest way to debug is to add `st.write()` statements in the code to print out variables and see what's happening.
    * For more complex issues, you can run the Python script with a debugger in a code editor like VS Code.

This project is in a good state, but there is always room for improvement. The `NEXT_STEPS.md` document provides a solid roadmap for future work. Good luck, and feel free to reach out with any questions!