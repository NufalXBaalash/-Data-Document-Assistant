# 📊 Data & Document Assistant 📝

A powerful Streamlit application that functions as an intelligent assistant for your documents and data files. This tool uses Cohere's LLM capabilities and vector search to help users analyze and understand their documents and datasets through natural conversation.

## ✨ Features

- 📄 **Document Understanding**: Process and analyze text files, PDFs, CSV files, and Excel spreadsheets
- 💬 **Conversational Interface**: Ask questions about your documents and data in natural language
- 📈 **Data Analysis**: Get insights and statistics about your datasets through simple conversation
- 🔍 **Vector Search**: Employs FAISS for efficient semantic search across your documents
- 🖥️ **Interactive UI**: Clean, user-friendly interface with data visualization capabilities
- 🤖 **Multiple Personality Options**: Choose between different assistant personas based on your needs

## 📋 Requirements

- 🐍 Python 3.7+
- 🔑 Cohere API key

## 📦 Dependencies

- 🚀 streamlit
- 🧠 cohere
- 🔢 numpy
- 🐼 pandas
- ⛓️ langchain
- 🔎 faiss-cpu
- 📰 PyPDF2
- 📊 openpyxl

## 🚀 Installation

```bash
git clone https://github.com/yourusername/data-document-assistant.git
cd data-document-assistant
pip install -r requirements.txt
```

Create a `requirements.txt` file with the following content:

```
streamlit>=1.22.0
cohere>=4.0.0
numpy>=1.20.0
pandas>=1.3.0
langchain>=0.0.200
faiss-cpu>=1.7.0
PyPDF2>=2.0.0
openpyxl>=3.0.0
```

## 🎮 Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Access the web interface in your browser (typically at http://localhost:8501)

3. Enter your Cohere API key in the sidebar

4. Upload your documents and data files (supported formats: txt, pdf, csv, xlsx, xls)

5. Click "Process Files" to analyze your documents

6. Start asking questions about your documents and data!

## ⚙️ How It Works

1. **📄 Document Processing**: The application processes various file types and converts them into chunks suitable for embedding.

2. **🧮 Vector Embeddings**: Uses Cohere's embedding models to convert text chunks into vector representations.

3. **🔎 Semantic Search**: Employs FAISS to create a vector database for efficient similarity search.

4. **🤖 Conversational AI**: Leverages Cohere's language models to understand and respond to user queries.

5. **📊 Data Analysis**: For data files, performs automatic analysis and visualization to provide insights.

## 🧩 Key Components

- **🔄 CustomCohereEmbeddings**: A custom implementation for Cohere embeddings to handle both document and query embedding
- **💬 ConversationalRetrievalChain**: A langchain component that maintains conversation context while retrieving relevant information
- **📈 Data Analysis Helpers**: Functions to analyze and summarize dataframes, detect patterns, and provide statistics

## 💡 Example Queries

- "What are the main topics discussed in my documents?"
- "Summarize the key points from the PDF I uploaded."
- "What's the average value in column X of my spreadsheet?"
- "Show me the correlation between columns Y and Z."
- "Are there any missing values in my dataset?"
- "What insights can you give me about this data?"

## 🎨 Customization

You can customize the application through the sidebar:
- Choose different Cohere models (command, command-light, etc.)
- Select different assistant personalities (Friendly & Helpful, Professional & Concise, Data Analyst)
- Work with multiple data files simultaneously

## 📋 Data Capabilities

The assistant can handle various data operations:
- 📊 **Statistical Analysis**: Calculate mean, median, standard deviation and other statistics
- 🔗 **Correlation Analysis**: Identify relationships between variables
- 🧹 **Missing Value Detection**: Find and report on null or missing values
- 📉 **Data Visualization**: Create simple charts and graphs for your data
- 📑 **Data Summaries**: Generate comprehensive overviews of datasets

## 📸 Screenshots

<!-- Add your screenshots here -->
![Application Interface](/path/to/screenshot1.png)
*Main interface of the Data & Document Assistant*

![Data Analysis Example](/path/to/screenshot2.png)
*Example of data analysis visualization*

![Document Query Example](/path/to/screenshot3.png)
*Example of querying documents*

<!-- You can add more screenshots as needed -->

## 🙏 Acknowledgments

This project utilizes several open-source libraries:
- [Streamlit](https://streamlit.io/) - For the interactive web interface
- [Langchain](https://github.com/hwchase17/langchain) - For document processing and conversation chains
- [Cohere](https://cohere.ai/) - For embeddings and language models
- [FAISS](https://github.com/facebookresearch/faiss) - For vector similarity search

---

Built with ❤️ by Ahmed Baalash
