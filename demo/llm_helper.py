from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()


# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
# embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
