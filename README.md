# Pydanticai PoC
Proof of Concept of using Pydantic AI as an RAG application

For this project you need to install [Ollama](https://ollama.com/download/windows) and [Postgress](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads) in your machine.

You also need to install [Pgvector](https://dev.to/mehmetakar/install-pgvector-on-windows-6gl)

# Pull the llama3 model into Ollama

```
ollama pull llama3.1
```

Then, to validate the installation run: 

```
ollama list
```

# Install requirements

```
pip install -r requirement.txt
```

# Virtual Environment Variables
Create your env variable based in the .env_template file

Example:
![image](https://github.com/user-attachments/assets/ace4cc0d-4fd2-4634-b522-4c78af2af07c)
