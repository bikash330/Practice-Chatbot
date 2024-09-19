import os

# from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# from typing_extensions import Concatenate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = ChatGroq(model="llama3-8b-8192")


def load_document(file_path):
    _, file_extension = os.path.splitext(file_path)

    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)

    elif file_extension == ".csv":
        loader = CSVLoader(file_path)

    elif file_extension in [".txt", ".md"]:
        loader = TextLoader(file_path)

    else:
        print("Unsupported file type")

    # Attempt to load the document
    docs = loader.load()

    if docs is None:
        raise ValueError(f"Failed to load document from {file_path}")

    return docs


file_path = "./Traffic.pdf"

try:
    docs = load_document(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(len(splits))

    # Initialize Hugging Face embeddings (no OpenAI API required)
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # Initialize Chroma vectorstore
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    results = rag_chain.invoke(
        {"input": "Implementation of Video Analytics algorithms"}
    )

    results

except Exception as e:
    print(f"Error: {e}")
