from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from src.deepseek_llm import DeepSeekLLM



class FacultyBylawsAgent:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.deepseek = DeepSeekLLM() # Use DeepSeek API (assuming it's Langchain-compatible)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )  # Use open-source embeddings
        self.qa_chain = self._setup_qa_chain()

    def _setup_qa_chain(self):
        # Load and process the PDF
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)

        # Create a vector store
        vectorstore = FAISS.from_documents(texts, self.embeddings)

        # Set up the retriever and QA chain
        retriever = vectorstore.as_retriever()

        return RetrievalQA.from_chain_type(
            llm=self.deepseek,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    def answer_question(self, query):
        result = self.qa_chain.invoke({"query": query})
        return result['result']
