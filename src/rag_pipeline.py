# src/rag_pipeline.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline



# 1️⃣ Load embedding model (same as used for indexing)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2️⃣ Load vector store (VERY IMPORTANT)
VECTOR_DB_PATH = "vector_store/chroma"

vectorstore = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embedding_model
)

# 3️⃣ Load LLM
hf_pipeline = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    max_new_tokens=300,
    temperature=0.3
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 4️⃣ Prompt template
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a financial analyst assistant for CrediTrust Financial.
Use ONLY the complaint excerpts below to answer the question.
If the information is insufficient, say you do not have enough information.

Context:
{context}

Question:
{question}

Answer:
"""
)

# 5️⃣ RAG function
def answer_question(question, k=5):
    docs = vectorstore.similarity_search(question, k=k)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PROMPT.format(context=context, question=question)

    answer = llm(prompt)

    sources = [
        {
            "complaint_id": d.metadata.get("complaint_id"),
            "product": d.metadata.get("product_category"),
            "issue": d.metadata.get("issue")
        }
        for d in docs[:2]
    ]

    return answer, sources


# 6️⃣ Test run
if __name__ == "__main__":
    response, sources = answer_question(
        "Why are customers unhappy with credit cards?"
    )

    print("ANSWER:\n", response)
    print("\nSOURCES:")
    for s in sources:
        print(s)
