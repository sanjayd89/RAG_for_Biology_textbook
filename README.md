# RAG Implementation

The source code folder contains three solutions

### Solution 1: [Simple RAG Solution](https://github.com/sanjayd89/RAG_for_Biology_textbook/blob/main/src/1.%20Simple%20RAG%20Solution.ipynb)

This has RAG implementation using LLamaindex framework using Huggingface LLM and embeddings

### Solution 2: [Streamlit Solution](https://github.com/sanjayd89/RAG_for_Biology_textbook/blob/main/src/2a.%20Streamlit_app.py)

This contains 2 files viz... 
- 2a. Streamlit_app.py which is to be run to see the User interface
- 2b. Data_Ingestion.py which is used for reading data, converting it into vector embeddings and then storing it locally

### Solution 3: [Alternate RAG Solution](https://github.com/sanjayd89/RAG_for_Biology_textbook/blob/main/src/3.%20Alternate%20RAG%20Solution.ipynb)

- This uses a completely different solution than the Solution 1.
- The text data from pdf is extracted using *Pdfplumber*.
- The data is then divided into chunks and stored in a dataframe alongwith meta information.
- TFIDF vectorizer is used to convert this text data into vectors.
- The same TFIDF model is then used to vectorize the user query.
- Using cosine-similarity from sklearn, the top 5 relevant chunks from dataframe is found.
- This context data along with the user query is then passed to Ollama model for response.
