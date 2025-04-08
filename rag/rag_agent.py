import os
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama

load_dotenv()

# Environment vars
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5433")
PG_DB = os.getenv("PG_DB", "ragdb")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.1")

# Custom Prompt
template = """
You are helpful assistant who provides information about company information to 
potential interest candidates who wants to apply to our company (Stellar Nodes). 
We are a consultancy company in the Netherlands.

# Goal 1: Provide accurate information about company information
In order to provide accurate information to candidates. You have the following
tools at your disposition:

- knowledge_retrieval: Use this tool to retrieve information about Stellar Nodes.

# Goal 2: Collect candidate's information
You should collect candidates information:
- name (is mandatory)
- email(is mandatory)
- position(or role, is mandatory)
- country (is mandatory)
- linkedin(optional)
- github(optional)

You need a chat_id, but you don't collect this information from the user. This is provided in the tool externally.
So don't ask the user for this information. In order to save candidates information. You have the following
tools at your disposition:
- save_candidate_information: Use this tool to save collected candidates information.

So when you start the conversation with the candidate, be sure to ask for personal information. Don't be too rude,
ask information but also provide the information they ask for. For example, if they ask for something specific, provide 
the answer and ask for the missing information. Answer directly and don't ask for consent to ask the questions.

Be concise in your responses. Only provide more detail if asked.
Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# Vector Store setup
embeddings = OllamaEmbeddings(model=LOCAL_MODEL)
connection_string = f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
vectorstore = PGVector(
    connection_string=connection_string,
    embedding_function=embeddings,
    collection_name="knowledge_base_{LOCAL_MODEL}",
)
retriever = vectorstore.as_retriever()

# Build LangChain RAG
llm = Ollama(model=LOCAL_MODEL, base_url="http://localhost:11434", temperature=0)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False,
)

# Create wrapper model to parse response
class Answer(BaseModel):
    answer: str

# Wrap it in a Pydantic Agent
llama_model = OpenAIModel(
    model_name=LOCAL_MODEL,
    provider=OpenAIProvider(base_url="http://localhost:11434/v1")
)

agent = Agent(llama_model, result_type=Answer)

# CLI loop
if __name__ == "__main__":
    print("🤖 Ask about Stellar Nodes! Type 'exit' to quit.\n")
    while True:
        q = input("❓> ")
        if q.lower() in ("exit", "quit"):
            break
        try:
            rag_response = rag_chain.run(q)
            response = agent.run_sync(rag_response)
            print("💬", response.data)
        except Exception as e:
            print("⚠️ Error:", e)
