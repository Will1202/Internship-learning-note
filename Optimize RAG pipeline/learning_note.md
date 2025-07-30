# Embeddings(嵌入)
# 什么是嵌入？
嵌入是人工智能理解含义的方式，而不仅仅是匹配单词。
人工智能不会将“滞纳金”或“利率”存储为纯文本，而是将这些短语转换为有意义的数字表示——有点像巨大创意地图上的位置图钉。
-两个相似的想法→它们的向量（数字串）很接近。
-完全不同的想法→它们的向量相距甚远。
<img width="608" height="343" alt="image" src="https://github.com/user-attachments/assets/7c6b5e59-0dad-4bc2-8560-7924170205ec" />
# 嵌入模型
你可以将它们视为紧凑的、预先训练的工具，将日常语言转化为人工智能可以理解的数学。
<img width="1037" height="506" alt="image" src="https://github.com/user-attachments/assets/c663f142-2b7e-4f87-9da0-7e0b56410c6a" />
# Chunking（分块）：处理长文档的关键
<img width="1397" height="230" alt="image" src="https://github.com/user-attachments/assets/31432651-dbad-49e3-af86-5e3c5bc483a3" />
<img width="1472" height="432" alt="image" src="https://github.com/user-attachments/assets/e2b9145d-615a-4a3d-b221-d85b55d7e443" />
<img width="1444" height="484" alt="image" src="https://github.com/user-attachments/assets/18d0a777-d192-43e7-a784-7744c7c43211" />
<img width="1160" height="1489" alt="image" src="https://github.com/user-attachments/assets/17c1f52a-9646-401a-8433-b4bf74c814ab" />
<img width="1265" height="527" alt="image" src="https://github.com/user-attachments/assets/4a72627e-f19d-4cfe-ba8c-ce369746a7fe" />
<img width="1481" height="834" alt="image" src="https://github.com/user-attachments/assets/5fc9574a-29bf-4fc1-84e5-d00e20fe882d" />
<img width="1410" height="537" alt="image" src="https://github.com/user-attachments/assets/a9769c57-4ab7-4d5f-bd7c-d900ec5b0f64" />
<img width="1259" height="844" alt="image" src="https://github.com/user-attachments/assets/ecdbef89-fa43-4f6d-b210-f21842a29ad8" />
<img width="1475" height="794" alt="image" src="https://github.com/user-attachments/assets/a012a118-abbf-4896-9f18-253460d80b58" />


**创建一个人工智能处理文档**
# 步骤 1：安装所需的库
```
!pip install llama-index llama-index-embeddings-huggingface llama-index-llms-gemini
```
# 第 2 步：加载示例文档（纯文本或 PDF）
无需分块即可开始
让我们首先加载一个没有分块的文档，看看会发生什么。
```
from llama_index.core import SimpleDirectoryReader

# Load PDF or text document
documents = SimpleDirectoryReader("sample_docs").load_data()
print(f"Loaded {len(documents)} documents.")
```

# 步骤3：应用不同的分块策略
既然我们已经看到了问题，让我们探索将文档分解成可管理块的不同方法。
 
## 固定长度分块
将文本拆分成大小相等的块（例如，每块 300 个标记）。
最适合结构化文本，但可能会尴尬地切断句子。
```
from llama_index.core.node_parser import SentenceSplitter

splitter_fixed = SentenceSplitter(chunk_size=300, chunk_overlap=0)  # No overlap
chunks_fixed = splitter_fixed.get_nodes_from_documents(documents)
print(f"Total Fixed-Length Chunks Created: {len(chunks_fixed)}")
```
**预期结果**：检索速度快，但句子被截断时可能会丢失上下文。
 
## 重叠分块
重叠的块承载着前一个块的一部分，以保持上下文的完整性。
防止人工智能在检索信息时丢失含义。
```
splitter_overlap = SentenceSplitter(chunk_size=300, chunk_overlap=50)  # 50-token overlap
chunks_overlap = splitter_overlap.get_nodes_from_documents(documents)
print(f"Total Overlapping Chunks Created: {len(chunks_overlap)}")
```
**预期结果**：检索更准确，句子连贯性更顺畅。但也存在一个缺点——由于文本重叠，存储空间使用量会略有增加。
 
## 语义分块（高级）
使用AI 嵌入来查找自然的话题转变并进行相应的分割。
当文档包含多个不相关的部分时，效果最佳。
```
from llama_index.core.node_parser import SemanticSplitter

semantic_splitter = SemanticSplitter()
chunks_semantic = semantic_splitter.get_nodes_from_documents(documents)
print(f"Total Semantic Chunks Created: {len(chunks_semantic)}")
```
**预期结果**：更多上下文感知的词块，以实现更佳的检索效果。但这也存在一个弊端——与其他方法相比，需要额外的处理时间。
 
# 步骤 4：生成用于检索的嵌入
现在我们已经将文档分块，让我们使用嵌入将每个块转换为向量表示。
```
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load embedding model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Apply embeddings
for chunk in chunks_overlap:  # Using Overlapping Chunks for best retrieval
    chunk.embedding = embed_model.get_text_embedding(chunk.text)

print("Embeddings Generated Successfully!")
```
这将加载一个预先训练好的模型，该模型知道如何将文本转换为嵌入。您正在使用一个轻量级但功能强大的模型，名为*MiniLM*。

## 步骤 5：使用 Gemini 存储和检索嵌入
现在我们已经对文本进行了分块和嵌入，让我们存储嵌入并测试文档检索。
```
from llama_index.core import VectorStoreIndex

# Create an index with our embeddings
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Set up query engine
query_engine = index.as_query_engine()

# Test a retrieval query
response = query_engine.query("What is the document about?")
print(response)
```
观察结果：

通过分块和嵌入，AI 可以检索最相关的文本，而不是转储整个文档！


<img width="1056" height="550" alt="image" src="https://github.com/user-attachments/assets/a8c58a50-9187-4e32-8be8-fba812a9629f" />

<img width="1045" height="362" alt="image" src="https://github.com/user-attachments/assets/2f055b12-edc0-4774-bb6c-70e662d7bc11" />

# 检索重排序：排序最佳结果
假设你的人工智能找到了几条相关信息，但并非所有信息都同样有用。重新排序可以解决这个问题。
它重新组织检索到的结果，将最有用的结果放在顶部，因此您的最终答案更有力、更清晰、更准确。

## 重新排序技术
人工智能使用不同的策略来决定哪些检索结果最有帮助：
<img width="1046" height="375" alt="image" src="https://github.com/user-attachments/assets/0461a633-89b0-4767-9100-70e332425df0" />

# 混合检索：结合关键词和向量搜索
好的答案并非单靠一种策略就能得出。关键词搜索（例如谷歌）虽然能提供精准度，但却会忽略细微差别。向量搜索（使用嵌入）能提供含义，但可能会忽略确切的术语。

**混合检索**正是这样做的——合并关键字和向量搜索以提供既准确又上下文丰富的响应。

混合检索的工作原理
- 步骤 1：关键词搜索→AI 查找具有精确词语匹配的部分。
- 第 2 步：向量搜索→AI 检索概念上相似的内容。
- 步骤 3：合并和重新排序→AI 将两者融合以产生最佳响应。
 



