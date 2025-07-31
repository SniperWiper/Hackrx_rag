import os
import aiohttp
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import uvicorn
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import tempfile
import asyncio
from functools import lru_cache

# Settings
class Settings(BaseSettings):
    gemini_api_key: str = ""
    llm_model: str = "gemini-1.5-pro"
    chunk_size: int = 512  # Increased for more context
    chunk_overlap: int = 128  # Increased for better overlap
    api_host: str = "0.0.0.0"
    api_port: int = int(os.getenv("PORT", 8000))
    debug: bool = os.getenv("ENVIRONMENT", "development") != "production"
    data_dir: str = "./data"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
os.makedirs(settings.data_dir, exist_ok=True)

# Models
class BatchQueryRequest(BaseModel):
    documents: str = Field(..., description="PDF blob URL")
    questions: List[str] = Field(..., description="List of questions to process")

class BatchQueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers")

# QueryRetrievalSystem
class QueryRetrievalSystem:
    def __init__(self):
        # Upgrade to a more powerful embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.llm = ChatGoogleGenerativeAI(model=settings.llm_model, google_api_key=settings.gemini_api_key)
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # For re-ranking
        self.prompt_template = PromptTemplate(
            input_variables=["query", "clauses"],
            template="""Based on the provided clauses, answer the query in detail, citing specific parts of the clauses where relevant. Return a JSON object with the following structure:
{{
  \"answer\": \"string\",
  \"meets_criteria\": boolean,
  \"applicable_conditions\": [\"string\"],
  \"rationale\": \"string\",
  \"confidence_score\": number,
  \"supporting_evidence\": [\"string\"]
}}
If the clauses do not fully answer the query, indicate what information is missing. Ensure all fields are present and correctly formatted. Return only valid JSON.

Query: {query}

Clauses: {clauses}"""
        )
        self.llm_chain = self.prompt_template | self.llm
        self.vector_store_cache = {}  # Simple in-memory cache

    async def process_document(self, blob_url: str) -> FAISS:
        # Check cache first
        if blob_url in self.vector_store_cache:
            return self.vector_store_cache[blob_url]

        # Download PDF
        async with aiohttp.ClientSession() as session:
            async with session.get(blob_url, timeout=30) as response:
                response.raise_for_status()
                pdf_bytes = await response.read()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf_path = temp_pdf.name

        # Load and process PDF
        loader = PyMuPDFLoader(temp_pdf_path)
        documents = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(chunks, self.embeddings)

        # Cache the vector store
        self.vector_store_cache[blob_url] = vector_store

        # Clean up
        os.remove(temp_pdf_path)
        return vector_store

    def rerank_chunks(self, query: str, docs: List[Any]) -> List[Any]:
        # Re-rank retrieved chunks using cross-encoder
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.cross_encoder.predict(pairs)
        ranked_docs = sorted(zip(scores, docs), reverse=True, key=lambda x: x[0])
        return [doc for _, doc in ranked_docs[:5]]  # Return top 5 after re-ranking

    async def process_batch_queries(self, document_url: str, questions: List[str]) -> List[str]:
        vector_store = await self.process_document(document_url)
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})  # Retrieve more chunks

        async def process_question(question: str) -> str:
            # Retrieve and re-rank documents
            docs = await retriever.ainvoke(question)
            reranked_docs = self.rerank_chunks(question, docs)
            clauses_text = "\n".join([doc.page_content for doc in reranked_docs])

            # Call LLM
            response = await self.llm_chain.ainvoke({"query": question, "clauses": clauses_text})
            parsed_response = self.parse_llm_response(response)
            return parsed_response.get("answer", "Unable to determine")

        # Process questions in parallel
        answers = await asyncio.gather(*[process_question(question) for question in questions])
        return answers

    def parse_llm_response(self, response: Any) -> Dict[str, Any]:
        if hasattr(response, "content"):
            response = response.content
        response = str(response)
        if response.startswith("```json") and response.endswith("```"):
            response = response[7:-3].strip()
        elif response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"answer": "Invalid response from LLM"}

# FastAPI Setup
app = FastAPI(
    title="LLM-Powered Query-Retrieval System",
    description="Semantic document retrieval and query processing",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

query_system = QueryRetrievalSystem()

@app.post("/hackrx/run", response_model=BatchQueryResponse)
async def process_batch_queries(request: BatchQueryRequest):
    answers = await query_system.process_batch_queries(request.documents, request.questions)
    return BatchQueryResponse(answers=answers)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})

if __name__ == "__main__":
    uvicorn.run("app:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)