import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)

# Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb_file_path = "faiss_index"


def create_vector_db():
    loader = CSVLoader(
        file_path="codebasics_faqs.csv",
        source_column="prompt"
    )

    documents = loader.load()

    vectordb = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    vectordb.save_local(vectordb_file_path)


def get_qa_chain():

    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt_template = """
Given the following context and question,
answer ONLY from the context.

If the answer is not present in the context,
say "I don't know".

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT
        }
    )

    return chain


if __name__ == "__main__":

    create_vector_db()

    chain = get_qa_chain()

    response = chain.invoke({
        "query": "Do you have javascript course?"
    })

    print(response["result"])