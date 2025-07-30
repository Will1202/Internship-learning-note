# 步骤 1：从 PDF 加载并提取文本
```
!pip install llama-index llama-index-llms-gemini pymupdf llama-index-embeddings-huggingface
```
## 从 PDF 加载和提取文本
```
import fitz  # PyMuPDF

# Load PDF document
doc = fitz.open("sample_docs/contract.pdf")

# Extract text from all pages
text = "\n".join([page.get_text() for page in doc])

print(f"Extracted {len(text.split())} words from the PDF.")
```
# 第 2 步：实现查询扩展和重写
通过扩展改进查询处理
```
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage

# Set up Gemini API key
import os
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

# Initialize Gemini LLM
llm = Gemini(model="models/gemini-1.5-flash")

# Define query rewriting function
def rewrite_query(user_query):
    messages = [
        ChatMessage(role="system", content="Rewrite this query for improved retrieval relevance."),
        ChatMessage(role="user", content=user_query),
    ]
    response = llm.chat(messages)
    return response.message.content

# Test query rewriting
query = "What are the penalties for late payments?"
expanded_query = rewrite_query(query)

print(f"Original Query: {query}")
print(f"Expanded Query: {expanded_query}")
```

# 第三步：实现混合检索（关键词+向量检索）
为什么要进行混合检索？
- 基于关键字的搜索（BM25）可找到精确的短语匹配。
- 基于向量的检索（嵌入）可以找到概念上相似的匹配。
混合检索将两者结合起来以获得最准确的搜索结果！
 
对 PDF 执行混合检索
```
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import BM25Retriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import HybridRetriever

# Load documents from the directory
documents = SimpleDirectoryReader("sample_docs").load_data()

# Initialize Hugging Face embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a document store
docstore = SimpleDocumentStore()

# Create a vector index for embedding-based retrieval
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=5)

# Create a BM25 keyword-based retriever
bm25_retriever = BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=5)

# Combine both retrievers into a Hybrid Retriever
hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever, bm25_retriever=bm25_retriever, alpha=0.5
)

# Set up query engine with hybrid retrieval
query_engine = RetrieverQueryEngine(retriever=hybrid_retriever)

# Test hybrid retrieval
query = "What is the refund policy?"
response = query_engine.query(query)

print(response)
```
# 步骤 4：实施重新排名以获得更准确的结果
为什么需要重新排名？
即使经过检索，某些结果仍然比其他结果更相关。重新排序：
按相关性对检索到的结果进行排序。
过滤掉不相关或多余的文本。
 
```
from llama_index.core.retrievers import LLMReranker

# Initialize reranker
reranker = LLMReranker(llm=llm)

# Get the retrieved results
retrieved_chunks = query_engine.query(query, return_results=True)

# Apply reranking
reranked_results = reranker.rerank(query, retrieved_chunks)

print("Top-ranked result:", reranked_results[0].text)
```
