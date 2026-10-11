At large scale, self-hosting saves thousands per month

# Cost optimisation stratergies:

1. Reduce dimension:
   1536 -> 512 dims
   savings: 30-60%
   effort: low
2. Quantisation:
   float32 -> int8
   saving: 50-75%
   effort: medium
3. Batch queries:
   bundle requests
   savings: 10-30%
   effort: low
4. caching
   cache frequent queries
   saving: 10-40%
   effort: medium
5. Right size
   Don't over-provision
   saving: 20-50%
   effort: low

Scale | best choice | Why
< 100K | Chroma / local | Free, simple
100k-1M | Pinecone serverless | low cost, zero ops
1M-10M | pgvector managed | cost-effective
10M+ | pgvector self-hosted | significant saving
