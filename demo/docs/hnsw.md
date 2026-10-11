Hierarchical Navigable Small World

Primary parameters:

1. Max connections (M):
   (connections among nodex)
   low (8-16): smaller index, faster builds
   high(32-64): larger index, higher accuracy

More M = more connections = more memory = better accuracy

2. Search effort (ef):
   Low (32-64): faster search
   High(200+): higher accuracy

more ef = more search effort = slower = better accuracy

We can only priorities two out of accuracy, speed and memory.

Prototype - 16M, 40ef, leads to speed
Production - 16M, 100ef, leads to balanced
High accuracy - 32M, 200ef, leads to accuracy

#pgvector - creating an HNSW index
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m=16, ef_construction = 64);

#At query time, set ef_search

Set hnsw.ef_search = 100 #higher = more accurate, slower

#Chroma - HNSW settings
collection = client.create_collection(
name = 'my_collection',
metadata = {
'hnsw:M': 16,
'hnsw:construction_ef': 100,
'hnsw:search_ef': 50,
}
)

# When and how to scale

if query taking too long -> index is too large for memory -> add more RAM or shard
if insert latency is spiking -> write bottleneck -> scale writes seperately
if getting lots of outmemory errors -> index doesn't fit -> bigger index or shard
if accuracy is droping -> ef search is too low -> increase ef or shard

# Managed vs self-hosted

Factor Managed(Pinecode) Self-Hosted(pgvector)

1. Scaling | Automatic | You manager
2. Ops burden | zero | significant
3. Cost at scale | High | low
4. Control | limited | full

if under 1Million vectors -> single pgvector is fine
else if don't have devops team -> use pinecone
else if cost is primary concern -> self-hosted pgvector
else pincone for convenience

Zero ops capacity? -> manager
Cost matters at scale? -> self-host
