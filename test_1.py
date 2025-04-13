import streamlit as st
import os
import tempfile
import cohere
import numpy as np
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, CSVLoader
from langchain.llms import Cohere
from langchain.embeddings.base import Embeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
import PyPDF2
import glob
from langchain.document_loaders import UnstructuredExcelLoader
from io import StringIO

# Custom Cohere embeddings class to fix the user_agent issue
class CustomCohereEmbeddings(Embeddings):
    def __init__(self, cohere_api_key):
        self.client = cohere.Client(api_key=cohere_api_key)
        
    def embed_documents(self, texts):
        """Embed a list of documents using Cohere API."""
        embeds = self.client.embed(
            texts=texts,
            model='embed-english-v3.0',
            input_type='search_document'
        ).embeddings
        return embeds
    
    def embed_query(self, text):
        """Embed a query using Cohere API."""
        embeds = self.client.embed(
            texts=[text],
            model='embed-english-v3.0',
            input_type='search_query'
        ).embeddings
        return embeds[0]

# Data analysis helper functions
def get_data_summary(df):
    """Generate a summary of the dataframe"""
    summary = []
    summary.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    summary.append("Columns: " + ", ".join(df.columns.tolist()))
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_cols = missing[missing > 0]
        summary.append(f"Missing values found in {len(missing_cols)} columns.")
    
    # Detect column types
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    if num_cols:
        summary.append(f"Numeric columns: {', '.join(num_cols)}")
    if cat_cols:
        summary.append(f"Categorical columns: {', '.join(cat_cols)}")
    if date_cols:
        summary.append(f"Date columns: {', '.join(date_cols)}")
    
    return "\n".join(summary)

def dataframe_to_document(df, filename):
    """Convert a dataframe to a document format that can be used for embeddings"""
    # Generate a text summary of the dataframe
    summary = get_data_summary(df)
    
    # Generate column descriptions
    column_descriptions = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            description = f"Column '{col}' (numeric): range from {df[col].min()} to {df[col].max()}, average: {df[col].mean():.2f}, median: {df[col].median()}"
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            unique_vals = df[col].nunique()
            description = f"Column '{col}' (categorical): {unique_vals} unique values"
            if unique_vals < 10:  # If few unique values, list them
                values = ", ".join([str(x) for x in df[col].unique()[:5]])
                description += f" including {values}" + ("..." if unique_vals > 5 else "")
        else:
            description = f"Column '{col}'"
        column_descriptions.append(description)
    
    # Sample data as text
    sample_rows = df.head(5).to_string()
    
    # Combine all text
    document_text = f"File: {filename}\n\nData Summary:\n{summary}\n\nColumn Details:\n" + "\n".join(column_descriptions) + f"\n\nSample Data:\n{sample_rows}"
    
    from langchain.schema import Document
    return Document(page_content=document_text, metadata={"source": filename, "type": "data"})

# Page configuration
st.set_page_config(page_title="Data & Document Assistant", page_icon="📊")

# Custom CSS to make the interface more friendly
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Data & Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">I can answer questions about your documents and analyze your data</div>', unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
if "chain" not in st.session_state:
    st.session_state.chain = None
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
    
if "current_df" not in st.session_state:
    st.session_state.current_df = None

# Sidebar for configuration
with st.sidebar:
    st.header("Assistant Configuration")
    
    # API Key input
    api_key = st.text_input("Enter Cohere API Key", type="password")
    
    # Model selection
    model_name = st.selectbox(
        "Select Cohere Model",
        ["command", "command-light", "command-nightly", "command-light-nightly"],
        index=0
    )
    
    # Personality settings
    persona = st.selectbox(
        "Assistant Personality",
        ["Friendly & Helpful", "Professional & Concise", "Data Analyst"],
        index=0
    )
    
    # Prompt enhancement based on personality
    persona_prompts = {
        "Friendly & Helpful": "Be warm, friendly and conversational. Use a helpful tone, occasionally ask follow-up questions, and speak as a knowledgeable friend would.",
        "Professional & Concise": "Be professional, direct, and efficient. Focus on delivering precise information without unnecessary details.",
        "Data Analyst": "Act as a data analyst. When discussing data, provide insights, identify patterns, and suggest potential analyses. Be clear and explanatory but focus on the implications of the data."
    }
    
    # File uploader with multiple file types
    uploaded_files = st.file_uploader(
        "Upload your documents and data", 
        accept_multiple_files=True, 
        type=["txt", "pdf", "csv", "xlsx", "xls"]
    )
    
    # Data files selector (for when multiple data files are uploaded)
    if st.session_state.dataframes:
        data_files = list(st.session_state.dataframes.keys())
        selected_data = st.selectbox("Select Data File for Analysis", data_files)
        if selected_data:
            st.session_state.current_df = st.session_state.dataframes[selected_data]
            st.write(f"Selected: {selected_data}")
            st.write(f"Shape: {st.session_state.current_df.shape[0]} rows, {st.session_state.current_df.shape[1]} columns")
    
    # Process uploaded files when user clicks the button
    if st.button("Process Files"):
        if not api_key:
            st.error("Please enter a Cohere API key.")
        elif uploaded_files:
            with st.spinner("Processing your files..."):
                try:
                    # Create a temporary directory
                    temp_dir = tempfile.mkdtemp()
                    
                    # Save uploaded files to the temporary directory
                    for file in uploaded_files:
                        file_path = os.path.join(temp_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                    
                    # Process documents based on file type
                    documents = []
                    
                    # Process text files
                    txt_files = glob.glob(os.path.join(temp_dir, "*.txt"))
                    if txt_files:
                        txt_loader = DirectoryLoader(temp_dir, glob="**/*.txt", loader_cls=TextLoader)
                        documents.extend(txt_loader.load())
                    
                    # Process PDF files
                    pdf_files = glob.glob(os.path.join(temp_dir, "*.pdf"))
                    for pdf_file in pdf_files:
                        pdf_loader = PyPDFLoader(pdf_file)
                        documents.extend(pdf_loader.load())
                    
                    # Process CSV files
                    csv_files = glob.glob(os.path.join(temp_dir, "*.csv"))
                    for csv_file in csv_files:
                        try:
                            # Load as dataframe first
                            filename = os.path.basename(csv_file)
                            df = pd.read_csv(csv_file)
                            st.session_state.dataframes[filename] = df
                            
                            # If no current dataframe is set, set this one
                            if st.session_state.current_df is None:
                                st.session_state.current_df = df
                                
                            # Convert to document for embeddings
                            doc = dataframe_to_document(df, filename)
                            documents.append(doc)
                        except Exception as e:
                            st.error(f"Error processing CSV file {csv_file}: {str(e)}")
                    
                    # Process Excel files
                    excel_files = glob.glob(os.path.join(temp_dir, "*.xlsx")) + glob.glob(os.path.join(temp_dir, "*.xls"))
                    for excel_file in excel_files:
                        try:
                            # Load as dataframe
                            filename = os.path.basename(excel_file)
                            # Read all sheets
                            excel_data = pd.read_excel(excel_file, sheet_name=None)
                            
                            # If multiple sheets, store each separately
                            if len(excel_data) > 1:
                                for sheet_name, df in excel_data.items():
                                    sheet_filename = f"{filename} - {sheet_name}"
                                    st.session_state.dataframes[sheet_filename] = df
                                    
                                    # Convert to document for embeddings
                                    doc = dataframe_to_document(df, sheet_filename)
                                    documents.append(doc)
                                    
                                    # If no current dataframe is set, set the first sheet
                                    if st.session_state.current_df is None:
                                        st.session_state.current_df = df
                            else:
                                # Single sheet
                                sheet_name = list(excel_data.keys())[0]
                                df = excel_data[sheet_name]
                                st.session_state.dataframes[filename] = df
                                
                                # If no current dataframe is set, set this one
                                if st.session_state.current_df is None:
                                    st.session_state.current_df = df
                                    
                                # Convert to document for embeddings
                                doc = dataframe_to_document(df, filename)
                                documents.append(doc)
                        except Exception as e:
                            st.error(f"Error processing Excel file {excel_file}: {str(e)}")
                    
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    
                    # Create embeddings and vectorstore
                    embeddings = CustomCohereEmbeddings(cohere_api_key=api_key)
                    vectorstore = FAISS.from_documents(chunks, embeddings)
                    st.session_state.vectorstore = vectorstore
                    
                    # Initialize Cohere model
                    llm = Cohere(
                        cohere_api_key=api_key, 
                        model=model_name, 
                        temperature=0.7
                    )
                    
                    # Create conversational chain with memory
                    memory = ConversationBufferMemory(
                        memory_key="chat_history", 
                        return_messages=True,
                        output_key="answer"
                    )
                    
                    # Add system message to chat memory
                    system_prompt = persona_prompts[persona]
                    # Add data-specific instructions if data files were processed
                    if st.session_state.dataframes:
                        data_files_info = ", ".join(st.session_state.dataframes.keys())
                        system_prompt += f"\n\nI have access to the following data files: {data_files_info}. When answering data questions, I should reference specific insights from the data and provide clear explanations."
                        
                    memory.chat_memory.messages.append(SystemMessage(content=system_prompt))
                    
                    st.session_state.chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                        memory=memory,
                        return_source_documents=True
                    )
                    
                    doc_count = len([d for d in documents if d.metadata.get("type") != "data"])
                    data_count = len([d for d in documents if d.metadata.get("type") == "data"])
                    
                    success_msg = f"Processed {len(chunks)} text chunks"
                    if doc_count > 0:
                        success_msg += f" from {doc_count} documents"
                    if data_count > 0:
                        success_msg += f" and {data_count} data files"
                        
                    st.success(success_msg)
                    
                    # Add a welcome message to the chat
                    if not st.session_state.messages:
                        welcome_message = "Hello! I've processed your files and I'm ready to help. "
                        if st.session_state.dataframes:
                            data_files = list(st.session_state.dataframes.keys())
                            welcome_message += f"I can analyze your data from {len(data_files)} files: {', '.join(data_files[:3])}"
                            if len(data_files) > 3:
                                welcome_message += f" and {len(data_files) - 3} more."
                            welcome_message += " What would you like to know?"
                        else:
                            welcome_message += "What would you like to know about your documents?"
                            
                        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
                        
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")
        else:
            st.error("Please upload some files first.")

# Data analysis function to run queries on the current dataframe
def analyze_data(query, df):
    """Run data analysis on the dataframe based on the query"""
    result = ""
    try:
        if "describe" in query.lower() or "summary" in query.lower() or "statistics" in query.lower():
            description = df.describe().to_string()
            result = f"Here's a statistical summary of the numerical columns:\n\n{description}"
            
        elif "correlation" in query.lower():
            numeric_df = df.select_dtypes(include=['number'])
            if numeric_df.shape[1] >= 2:
                corr = numeric_df.corr().to_string()
                result = f"Here's the correlation matrix for the numerical columns:\n\n{corr}"
            else:
                result = "The dataset needs at least two numeric columns to calculate correlations."
                
        elif "missing" in query.lower() or "null" in query.lower():
            missing = df.isnull().sum().to_string()
            result = f"Here's the count of missing values in each column:\n\n{missing}"
            
        elif "unique" in query.lower():
            unique_counts = {col: df[col].nunique() for col in df.columns}
            result = "Unique value counts for each column:\n\n" + "\n".join([f"{col}: {count}" for col, count in unique_counts.items()])
            
        else:
            # Basic information about the dataset
            result = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n\n"
            result += f"Columns: {', '.join(df.columns.tolist())}\n\n"
            result += f"Sample data (first 5 rows):\n{df.head(5).to_string()}"
            
        return result
    except Exception as e:
        return f"I encountered an error while analyzing the data: {str(e)}"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Ask me about your documents or data...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()
        
        if st.session_state.chain is not None:
            # Generate response
            with st.spinner("Thinking..."):
                try:
                    # Check if this is a data-specific query
                    data_indicators = ["data", "dataset", "csv", "excel", "xlsx", "dataframe", "column", "row", 
                                     "average", "mean", "median", "correlation", "analyze", "statistics",
                                     "chart", "plot", "graph", "visualization", "missing values", "null"]
                    
                    is_data_query = any(indicator in user_input.lower() for indicator in data_indicators)
                    
                    # Check if this is a conversational query
                    conversational_indicators = ["hello", "hi there", "thanks", "thank you", "how are you", 
                                               "good morning", "what's up", "nice to", "appreciate", "hey"]
                    
                    is_conversational = any(indicator in user_input.lower() for indicator in conversational_indicators)
                    
                    if is_conversational:
                        # For conversational inputs, provide friendly responses
                        friendly_responses = {
                            "hello": "Hello! How can I help you with your files today?",
                            "hi there": "Hi there! What would you like to know about your documents or data?",
                            "thanks": "You're welcome! Is there anything else you'd like to know?",
                            "thank you": "You're welcome! Happy to help with any other questions.",
                            "how are you": "I'm doing well, thanks for asking! I'm ready to help you analyze your files.",
                            "good morning": "Good morning! Ready to explore your documents and data together?",
                            "what's up": "I'm here and ready to assist! What would you like to know?",
                            "hey": "Hey there! How can I help you today?"
                        }
                        
                        # Find matching response or use default
                        for key, response in friendly_responses.items():
                            if key in user_input.lower():
                                answer = response
                                break
                        else:
                            answer = "Hello! I'm your file assistant. What would you like to know about your documents or data?"
                    
                    elif is_data_query and st.session_state.current_df is not None:
                        # For data questions, use a combination of data analysis and the chain
                        df_info = analyze_data(user_input, st.session_state.current_df)
                        
                        # Get context from the chain
                        response = st.session_state.chain({"question": user_input})
                        chain_answer = response["answer"]
                        
                        # Combine the answers
                        if "I don't know" in chain_answer or "I don't have" in chain_answer:
                            # The chain doesn't have a good answer, just use the data analysis
                            answer = f"Based on the current data file '{list(st.session_state.dataframes.keys())[list(st.session_state.dataframes.values()).index(st.session_state.current_df)]}':\n\n{df_info}"
                        else:
                            # Combine both answers
                            answer = f"{chain_answer}\n\nAdditional data insights:\n{df_info}"
                    
                    else:
                        # For document questions, use the chain
                        response = st.session_state.chain({"question": user_input})
                        answer = response["answer"]
                        source_docs = response.get("source_documents", [])
                        
                        if source_docs:
                            # Add sources in a more natural way
                            answer += "\n\nI found this information in these sources:"
                            for i, doc in enumerate(source_docs[:2]):  # Limit to top 2 sources for clarity
                                source = doc.metadata.get('source', 'one of your documents')
                                # Clean up the source path for better readability
                                source = os.path.basename(source) if os.path.isfile(source) else source
                                answer += f"\n• {source}"
                    
                    # Display response
                    response_container.write(answer)
                    
                    # Store the answer in the messages
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_message = f"I'm having trouble answering that. Could you please rephrase your question? (Error: {str(e)})"
                    response_container.write(error_message)
                    
                    # Store the error message
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
        else:
            # Friendly message when no files are loaded
            no_files_message = "I'd love to help you with that, but I need to learn from your files first. Could you please upload and process some documents or data files using the sidebar?"
            response_container.write(no_files_message)
            
            # Store the message 
            st.session_state.messages.append({"role": "assistant", "content": no_files_message})

# Display helpful instructions if the chat is empty
if len(st.session_state.messages) == 0:
    st.markdown("""
    ## Welcome to your Data & Document Assistant!
    
    I can help you have natural conversations about your documents and analyze your data. To get started:
    
    1. Enter your Cohere API key in the sidebar
    2. Upload your files (PDFs, text files, CSVs, Excel spreadsheets)
    3. Click "Process Files" to help me learn from them
    4. Ask me anything about your content or data!
    
    I'll maintain context throughout our conversation and provide helpful, data-driven responses.
    """)

# Add data visualization section if data is loaded
if st.session_state.current_df is not None:
    st.subheader("Data Preview")
    st.dataframe(st.session_state.current_df.head(10))
    
    # Simple visualizations
    st.subheader("Quick Visualizations")
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Numeric columns for histogram
        num_cols = st.session_state.current_df.select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            selected_num_col = st.selectbox("Select column for histogram", num_cols)
            st.bar_chart(st.session_state.current_df[selected_num_col].value_counts())
    
    with viz_col2:
        # Categorical columns for bar chart
        cat_cols = st.session_state.current_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            selected_cat_col = st.selectbox("Select column for bar chart", cat_cols)
            # Limit to top 10 categories
            top_cats = st.session_state.current_df[selected_cat_col].value_counts().head(10)
            st.bar_chart(top_cats)