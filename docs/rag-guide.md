# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by retrieving relevant information from an external knowledge base before generating a response.

## Why RAG?

LLMs have several limitations:
- **Knowledge cutoff**: They only know information up to their training date.
- **Hallucination**: They may generate plausible but incorrect information.
- **No domain expertise**: They lack specialized knowledge about your specific domain.

RAG addresses these issues by grounding the LLM's responses in retrieved factual content.

## How RAG Works

### 1. Document Ingestion (Offline)

The knowledge base is prepared through these steps:

1. **Loading**: Documents are loaded from various sources (PDFs, web pages, databases).
2. **Chunking**: Documents are split into smaller, semantically meaningful chunks.
3. **Embedding**: Each chunk is converted into a vector representation using an embedding model.
4. **Indexing**: Vectors are stored in a vector database for efficient similarity search.

### 2. Query Processing (Online)

When a user asks a question:

1. **Query Embedding**: The question is converted into a vector using the same embedding model.
2. **Retrieval**: The vector database finds the most similar document chunks (top-k).
3. **Context Assembly**: Retrieved chunks are combined into a context string.
4. **Generation**: The LLM generates an answer using both the question and the retrieved context.

## Chunking Strategies

Choosing the right chunking strategy significantly affects RAG quality:

- **Fixed-size chunking**: Split by character count. Simple but may break semantic units.
- **Recursive character splitting**: Tries to split at natural boundaries (paragraphs, sentences) while respecting size limits.
- **Semantic chunking**: Uses embedding similarity to determine split points.
- **Document-specific splitting**: Uses document structure (headers, sections) as split boundaries.

## Vector Databases

Popular vector databases for RAG:

| Database | Type | Key Features |
|----------|------|-------------|
| ChromaDB | Embedded | Easy setup, Python-native |
| FAISS | Library | Fast similarity search by Meta |
| Pinecone | Cloud | Managed service, scalable |
| Weaviate | Self-hosted/Cloud | Hybrid search support |
| Milvus | Self-hosted/Cloud | High-performance, distributed |

## Evaluation Metrics

RAG systems should be evaluated on:

- **Retrieval quality**: Are the right documents being retrieved? (Precision@k, Recall@k)
- **Answer faithfulness**: Is the answer grounded in the retrieved context?
- **Answer relevance**: Does the answer actually address the user's question?
- **Latency**: How long does the full pipeline take?

## Advanced Techniques

### Re-ranking

After initial retrieval, a re-ranker model can re-order results for better relevance. Popular re-rankers include Cohere Rerank and BGE-Reranker.

### Hybrid Search

Combining dense retrieval (embedding-based) with sparse retrieval (BM25/keyword-based) often yields better results than either approach alone.

### Query Transformation

Reformulating or expanding the user's query can improve retrieval:
- **HyDE (Hypothetical Document Embeddings)**: Generate a hypothetical answer, then use its embedding for retrieval.
- **Multi-query**: Generate multiple variants of the question and retrieve for each.
