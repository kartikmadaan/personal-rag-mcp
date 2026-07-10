## #1 - Checking the Embeddings

```sql
rag=# SELECT embedding FROM chunks LIMIT 1;
```

```
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 [-0.042613667,-0.048566293,-0.004255321,0.023029238,0.088384695,0.005020686,0.0058553433,-0.014182257,-0.05002319,-0.011246672,-0.037499126,0.035697002,0.041426864,0.04975192,-0.015448844,0.056761194,0.08434716,0.041742176,-0.080812655,-0.021353522,-0.0074074585,0.035720374,-0.08435771,0.019120006,-0.051107194,0.03735525,0.09241376,-0.023635363,-0.019705964,-0.059262205,0.042277783,0.05742157,-0.016578179,0.040847827,-0.11289727,0.040629372,-0.07349565,0.015062049,-0.09145708,-0.019115781,0.04550924,-0.03594301,-0.07824438,0.008037907,0.10376507,0.079611935,-0.0054431623,-0.08831363,0.02768976,0.028009685,-0.16825387,-0.007422999,-0.032012127,0.062660836,-0.0024108319,-0.042702183,0.039154753,-0.03234198,-0.019023178,-0.056549422,-0.029980984,-0.06891991,0.055183113,-0.08938073,-0.024252355,-0.0095952405,-0.05196362,0.06791828,-0.03304897,0.061376706,0.025259444,0.10071151,-0.097406864,0.050429463,-0.07637671,0.02431014,0.046617154,-0.023046661,0.08063704,0.0019672604,-0.0626841,0.08000547,-0.010422996,-0.02472777,0.02660311,-0.006204647,-0.008388936,0.0014220556,0.030333059,0.015085005,0.04096632,0.051340822,0.088607445,-0.022982158,0.05153843,-0.031161567,0.067037895,-0.032374505,0.041620426,0.03029254,0.02076473,0.11413697,0.04827239,0.018443089,-0.03529221,-0.019207845,0.012847152,0.044913705,0.01748758,-0.056125887,-0.03427022,0.005158263,-0.07496568,-0.086000174,0.019814631,-0.07325783,-0.04158613,-0.04736117,-0.023207,0.09701727,-0.010976138,0.014312879,0.042689,0.038167287,-0.053368017,0.07611425,-0.09208623,8.7462775e-33,-0.10767528,-0.0012921905,-0.04485408,0.03936616,-0.049326,-0.025254253,0.021676008,0.0035677687,-0.0085069025,0.067554355,-0.083195,0.067517586,-0.046641994,0.09973227,0.11248744,-0.036603983,0.08526568,0.042687945,-0.01463987,-0.091315515,0.14489086,-0.055048663,0.014869633,0.018732661,0.03864325,-0.023664452,0.06610656,-0.010869013,-0.00017018786,-0.005036811,0.0041991593,-0.012214251,0.037014958,0.060812533,0.00797034,-0.01675774,-0.064851694,-0.030343141,0.0016574564,0.006591885,0.010187928,-0.0047413902,-0.025955072,0.011253938,-0.08452718,0.030758508,0.011768383,-0.033670094,-0.032177757,-0.074943975,0.057083376,-0.027288621,-0.009576532,0.022386372,-0.009889851,0.006411007,-0.04001828,-0.10577593,-0.033626415,-0.0022859827,-0.09559541,0.006701782,0.014610426,0.014981057,-0.05421031,-0.049741242,0.056680977,0.07748574,-0.0013129807,-0.00969361,0.022073874,0.041060485,0.037357885,-0.12703134,0.040119916,-0.049262412,0.05031126,-0.0670142,-0.022348002,0.008533152,-0.005349546,0.008548494,0.034526408,-0.05475758,0.02427632,-0.08546414,0.015465527,-0.037633404,-0.029748771,-0.038570087,-0.09571572,0.05777642,0.010175826,-0.0058324547,-0.018231343,-6.705158e-33,-0.0623081,-0.02303988,0.009123074,0.022356065,0.035980314,0.008379067,-0.113043934,0.0075292,-0.034578543,-0.11704913,-0.092505366,-0.128538,0.090833016,-0.02053982,-0.0114138955,0.10918233,-0.07562761,0.011120728,-0.017738074,0.07226723,-0.095412895,0.022386443,-0.06654544,0.0091867475,-0.035028264,0.0052074427,0.022140251,0.01133691,0.016112523,-0.022103043,-0.03991451,-0.040682793,-0.04188819,-0.00936271
4,0.0016771381,-0.004210369,0.010523266,-0.064299084,0.026818981,0.06450827,0.017134229,0.051835418,-0.05344211,-0.022903768,0.004258541,0.004766775,-0.0656206,0.029275158,0.08381176,0.000304095,-0.045745585,0.015500729,-0.008900341,0.013708359,-0.0026282454,-0.008897123,-0.020136144,0.02739153,0.0072474293,0.00083042,-0.00868261,-0.039452773,0.037466977,0.031891465,0.059727907,-0.032585386,0.036215335,-0.027207045,-0.070488065,-0.015171594,-0.0045279134,-0.00911656,0.018781798,0.020343727,-0.05823543,-0.047269024,-0.008649812,-0.026041914,0.019952813,-0.005779651,0.079084426,0.02282579,0.06758699,0.053488955,0.017519427,0.053353045,-0.0040450017,0.002505252,0.050709877,-0.022423679,-0.019764408,0.014124947,-0.066965654,0.09876816,0.031487733,-5.8574674e-08,-0.054480508,-0.0017500024,0.011306075,-0.024786236,0.042861838,-0.0048003606,-0.0067246035,0.11462182,0.02963718,0.061715256,0.05297358,-0.08319884,-0.10453799,-0.011348354,0.09221691,0.02307884,0.017413458,-0.029539227,0.036700163,0.04699758,0.054210585,0.07168389,0.048502903,0.02567673,0.072698474,-0.13084318,0.017684385,0.06241256,0.044107046,0.0033980822,-0.06458988,0.022733988,0.021631302,0.026509475,0.1010392,0.017382948,0.047598828,0.004208466,-0.084346235,-0.007094536,0.0321281,0.067195565,0.009025624,-0.019474141,0.05389284,0.0075024697,0.038305167,-0.05268749,0.068607464,0.07562726,-0.023137059,-0.016771609,0.047931623,0.10679002,0.05568557,0.044171773,-0.012764979,0.020138264,0.100356966,-0.053401064,-0.003061258,-0.08745871,-0.049451772,0.0091712745]
(1 row)

(END)
```

There are 877 such rows

---

### The Anatomy of One Row

Each row in the chunks table represents one specific, bite-sized piece of text extracted from the PDFs, along with its unique AI "meaning footprint."
877 rows = 877 chunks from a collection of 10 PDFs.

The Table has following columns -

1. ID - Unique ID for the row (p key)
2. Source - file name
3. chunk_id - The location of this collection of words in the file (example 12th chunk)
4. Text - The actual text representation of the chunk. Can be calculated but easier to fetch this way. Also, there is no way to convert from Embedding to Text.
5. Metadata - Details about the file
6. Embedding - The 384 dimensional vector generated strictly from the text in that row.

---

### What it Means to the AI

Each row maps to a single point floating in a 384-dimensional geometric space.

Point = {x1, x2, x3 ..., x384}

Each dimension could mean something like -

- Dim 1. How much is this about technology
- Dim 2. How formal is the tone
- ...
- Dim 384. Is this talking about time or deadlines

Because every text chunk is assigned a specific coordinate in this 384-dimensional room, pgvector can use geometry to find related data.

---

## #2 - Checking the dimensions

```sql
SELECT 
    c.relname AS table_name,
    a.attname AS column_name,
    -- Extracts the exact dimension size (e.g., 1536 or 384) from the type modifier
    a.atttypmod AS vector_dimensions
FROM pg_attribute a
JOIN pg_class c ON a.attrelid = c.oid
WHERE c.relname = 'chunks' AND a.attname = 'embedding';
```

Output -

```
 table_name | column_name | vector_dimensions 
------------+-------------+-------------------
 chunks     | embedding   |               384
(1 row)
```

---

## #3 - Checking table values

```sql
SELECT 
    source, 
    chunk_id, 
    LEFT(text, 100) AS text_preview, 
    jsonb_pretty(metadata::jsonb) AS formatted_metadata
FROM chunks 
LIMIT 5;
```

```
-[ RECORD 1 ]------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
source             | corpus/SIGMOD21_Milvus.pdf
chunk_id           | 0
text_preview       | Milvus: A Purpose-Built Vector Data Management System                                                                                                                                                                                                                                     +
                   | Jianguo Wang*, Xiaomeng Yi, Rentong Guo, Hai J
formatted_metadata | {                                                                                                                                                                                                                                                                                         +
                   |     "page": 0,                                                                                                                                                                                                                                                                            +
                   |     "title": "Milvus: A Purpose-Built Vector Data Management System",                                                                                                                                                                                                                     +
                   |     "author": "Jianguo Wang*, Xiaomeng Yi, Rentong Guo, Hai Jin, Peng Xu, Shengjun Li, Xiangyu Wang, Xiangzhou Guo, Chengming Li, Xiaohai Xu, Kun Yu, Yuxing Yuan, Yinghao Zou, Jiquan Long, Yudong Cai, Zhenxiang Li, Zhifeng Zhang, Yihua Mo, Jun Gu, Ruiyi Jiang, Yi Wei, Charles Xie",+
                   |     "source": "corpus/SIGMOD21_Milvus.pdf",                                                                                                                                                                                                                                               +
                   |     "creator": "LaTeX with acmart 2017/12/14 v1.48 Typesetting articles for the Association for Computing Machinery and hyperref 2020-05-15 v7.00e Hypertext links for LaTeX",                                                                                                            +
                   |     "moddate": "2021-03-25T10:51:05-04:00",                                                                                                                                                                                                                                               +
                   |     "subject": "-  Information systems  ->  Database management system engines; Data access methods;",                                                                                                                                                                                    +
                   |     "trapped": "/False",                                                                                                                                                                                                                                                                  +
                   |     "filetype": ".pdf",                                                                                                                                                                                                                                                                   +
                   |     "keywords": "Vector database; High-dimensional similarity search; Heterogeneous computing; Data science; Machine learning",                                                                                                                                                           +
                   |     "producer": "MiKTeX pdfTeX-1.40.21",                                                                                                                                                                                                                                                  +
                   |     "page_label": "1",                                                                                                                                                                                                                                                                    +
                   |     "total_pages": 14,                                                                                                                                                                                                                                                                    +
                   |     "creationdate": "2021-03-25T10:51:05-04:00",                                                                                                                                                                                                                                          +
                   |     "ptex.fullbanner": "This is MiKTeX-pdfTeX 4.0.1 (1.40.21)"                                                                                                                                                                                                                            +
                   | }
```

---

## #4 - Using the index magic

### The Semantic Index (hnsw)

```sql
CREATE INDEX IF NOT EXISTS chunks_hnsw
   ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

- USING hnsw: Uses the Hierarchical Navigable Small World algorithm. Instead of checking the query vector against every single row in the database (which gets incredibly slow), it jumps through a multi-layered highway graph to instantly find the closest vectors

- (embedding vector_cosine_ops): Tells the database to index the embedding column using Cosine Distance. This means when we query the database in Phase 2, it will calculate the angle between vectors (similarity of meaning) rather than straight-line distance.

- (m = 16): The maximum number of bidirectional connection links established for each new element per graph layer. A value of 16 balances speed and memory

- (ef_construction = 64):  Controls index build quality. A higher number means PostgreSQL spends more time structuring the graph during ingestion for better search accuracy later

### The keyword search index (tsv) : Full-text search (BM 25) by tracking exact keywords

```sql
CREATE INDEX IF NOT EXISTS chunks_tsv
   ON chunks USING gin (tsv);
```

- tsv (Text Search Vector): This column stores the chunk's text preprocessed into a list of clean "tokens" or "lexemes" (e.g., stripping out punctuation, removing words like "and/the", and reducing words like "running" to "run")

- USING gin: Generalized Inverted Index. Think of this like the index at the back of a textbook. It creates a giant mapping table of Word -> List of Chunk IDs where that word appears

- Why it matters: If you search for the exact serial number "X-992-B", your vector index might miss it because serial numbers don't have "semantic meaning." This GIN index will find it instantly

---

## #5 - What this means for retrieval

By creating both, pgvector is fully prepared for Hybrid Search with Reciprocal Rank Fusion (RRF). When a user asks a question, the pipeline will fire two ultra-fast queries simultaneously:

The HNSW index retrieves chunks that match the concept of the question.

The GIN index retrieves chunks that match the exact keywords or codes

## #6 - Checking the retrieval 

Below query compares the first valid chunk to 5 random chunks. The first entry is expected to be a perfect match (sample set is picked from DB). We expect there to be other relevant matches from DB because the chunks are a part of a coherent document.

```sql
rag=# WITH target_chunk AS (
    -- Pick a random valid chunk that has an embedding
    SELECT id, embedding 
    FROM chunks 
    WHERE embedding IS NULL = FALSE 
    LIMIT 1
)
SELECT 
    c.id, 
    c.source, 
    LEFT(c.text, 60) AS snippet,
    -- Cosine distance (0 means identical, 2 means completely opposite)
    (c.embedding <=> t.embedding) AS cosine_distance,
    -- Cosine similarity (Your Python formula)
    (1 - (c.embedding <=> t.embedding)) AS cosine_similarity
FROM chunks c, target_chunk t
ORDER BY cosine_distance ASC
LIMIT 5;
```

```
 id |           source           |                           snippet                            |   cosine_distance   | cosine_similarity  
----+----------------------------+--------------------------------------------------------------+---------------------+--------------------
  1 | corpus/SIGMOD21_Milvus.pdf | Milvus: A Purpose-Built Vector Data Management System       +|                   0 |                  1
    |                            | Jiangu                                                       |                     | 
  4 | corpus/SIGMOD21_Milvus.pdf | hao Zou, Jiquan Long, Yudong Cai, Zhenxiang Li, Zhifeng Zhan | 0.22916015299677084 | 0.7708398470032292
  7 | corpus/SIGMOD21_Milvus.pdf | SIGMOD ’21, June 20–25, 2021, Virtual Event, China Wang et a |  0.3245069376683475 | 0.6754930623316525
  9 | corpus/SIGMOD21_Milvus.pdf | -class                                                      +| 0.36638877887692356 | 0.6336112211230764
    |                            | citizens. (1) Legacy database components such as opti        |                     | 
  5 | corpus/SIGMOD21_Milvus.pdf |  approach in recommender                                    +|  0.3763400130373241 | 0.6236599869626759
    |                            | systems is called vector embedding                           |                     | 
(5 rows)
```

The output shows that self-similarity match for row 1 is as we expected. For the other 4 rows, the cosine distances are <1, this means that clustering is working as expected.

`'<=>'` is the symbol used to compute the cosine distance b/w 2 rows. Cosine distance measures the angle b/w two vectors in the database completely ignoring how long the vectors are. What this means for us is that two phrases/sentences no matter how long each is are semantically similar if their cosine distance is less.

cosine_distance = 1 - cosine_similarity
cosine_similarity = (a.b)/(|a||b|) # Remember the dot product of vectors?
0 indicates same direction = perfect match
1 indicates orthogonal direction = unrelated
2 indicates opposite direction = opposite
```

### Other pgvector operators

- `<->` : This is the euclidean distance operator (straight line distance); best for image vectors or normalised embeddings
- `<#>` : This is the negative inner product; faster to compute than cosine distance but used only if embeddings are normalised to 1.

Below query searches using BM25

```sql
SELECT id, source, LEFT(text, 60) AS snippet,
       ts_rank_cd(tsv, plainto_tsquery('english', 'Jianguo')) as rank
FROM chunks 
WHERE tsv @@ plainto_tsquery('english', 'Jianguo');
```

```sql
rag=# SELECT id, source, LEFT(text, 60) AS snippet,
       ts_rank_cd(tsv, plainto_tsquery('english', 'Jianguo')) as rank
FROM chunks 
WHERE tsv @@ plainto_tsquery('english', 'Jianguo');
```

```
 id |           source           |                        snippet                        | rank 
----+----------------------------+-------------------------------------------------------+------
  1 | corpus/SIGMOD21_Milvus.pdf | Milvus: A Purpose-Built Vector Data Management System+|  0.1
    |                            | Jiangu                                                | 
  3 | corpus/SIGMOD21_Milvus.pdf | \x19 Database management system en-                  +|  0.1
    |                            | gines; Data access methods;                           | 
 84 | corpus/SIGMOD21_Milvus.pdf | on Management of Data (SIGMOD) . 835–850.            +|  0.1
    |                            | [39] Jie Li, Haife                                    | 
 90 | corpus/SIGMOD21_Milvus.pdf |  Fast Accurate Billion-                              +|  0.1
    |                            | point Nearest Neighbor Search on a S                  | 
(4 rows)
```

The output clearly shows matches across the document for 'Jianguo'

## Running a dedicated re-ranking test (test_retrieval.py)

### Query: How does Milvus handle vector data management and scalar filtering?

```
[Dense Top Hit] ID: 63 | Snippet: Milvus: A Purpose-Built Vector Data Management System SIGMOD...
--- Stage 1: Top 5 Hybrid (RRF) Results ---
Rank 1 | Chunk ID: 63 | RRF Score: 0.0164 | Snippet: Milvus: A Purpose-Built Vector Data Management Sys...
Rank 2 | Chunk ID: 10 | RRF Score: 0.0161 | Snippet:  types such as vector similarity search with vario...
Rank 3 | Chunk ID: 13 | RRF Score: 0.0159 | Snippet: Milvus: A Purpose-Built Vector Data Management Sys...
Rank 4 | Chunk ID: 50 | RRF Score: 0.0156 | Snippet: .1 Asynchronous Processing
Milvus is designed to m...
Rank 5 | Chunk ID: 74 | RRF Score: 0.0154 | Snippet:  for billion-scale data and
Vearch is significantl...

--- Stage 2: Top 5 Reranked (Cross-Encoder) Results ---
Rank 1 | Chunk ID: 10 | CE Score: 5.8672 | Snippet:  types such as vector similarity search with vario...
Rank 2 | Chunk ID: 25 | CE Score: 4.4962 | Snippet: Milvus: A Purpose-Built Vector Data Management Sys...
Rank 3 | Chunk ID: 11 | CE Score: 3.8902 | Snippet:  by hundreds of organiza-
tions and institutions w...
Rank 4 | Chunk ID: 2 | CE Score: 3.7277 | Snippet:  data. Milvus sup-
ports easy-to-use application i...
Rank 5 | Chunk ID: 13 | CE Score: 3.5555 | Snippet: Milvus: A Purpose-Built Vector Data Management Sys...
```

## Checking the entire pipeline

```
(.venv) kmadaan@Kartiks-Mac-mini personal-rag-mcp % python -c "from rag.pipeline import query; import json; print(json.dumps(query('what are deep neural networks?'), indent=2))"              
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████| 103/103 [00:00<00:00, 8323.96it/s]
Loading weights: 100%|██████████████| 105/105 [00:00<00:00, 6602.83it/s]
{
  "answer": "Deep neural networks refer to a type of neural network with multiple hidden layers, typically more than five layers [S3]. These networks have become popular due to their unprecedented success in various machine learning tasks [S3]. They are inspired by biological neural networks and have proven to work well for multiple use cases in image processing and recognition [S2].\n\nDeep neural networks can be used for a variety of tasks, including recognizing patterns, anomalies, and making predictions [S2]. Some popular architectures of deep neural networks include Convoluted Neural Networks (CNNs) [S2], which were developed by Yann LeCun and his associates in 1998.\n\nThe success of deep neural networks can be attributed to their ability to learn complex mappings from input to output spaces, making them useful for a wide range of applications in image analytics [S2]. However, there is still limited research on the theoretical perspective of deep neural networks, with only a few publications investigating their complexity and approximation properties [S3].\n\nReferences:\n[S1] E. Dahl, A. Mohamed, N. Jaitly, A. Senior, V . Vanhoucke, P. Nguyen,\nT. Sainath, and B. Kingsbury. Deep neural networks for acoustic modeling in speech recognition.\nIEEE Signal Processing Magazine, 29(6):82\u201397, Nov. 2012.\n[S2] source=corpus/White+paper+1+-+Introduction+to+Image+analytics.pdf\n<context>...</context>\n[S3] source=corpus/1402.1869v2.pdf\n<context>...</context>\n[S4] O. Krause, A. Fischer, T. Glasmachers, and C. Igel. Approximation properties of DBNs with binary\nhidden units and real-valued visible units. In Proceedings of The 30th International Conference\non Machine Learning (ICML\u20192013), 2013.\n[S5] source=corpus/1712.09913v3.pdf",
  "citations": [
    {
      "id": "S1",
      "source": "corpus/1402.1869v2.pdf"
    },
    {
      "id": "S2",
      "source": "corpus/White+paper+1+-+Introduction+to+Image+analytics.pdf"
    },
    {
      "id": "S3",
      "source": "corpus/1402.1869v2.pdf"
    },
    {
      "id": "S4",
      "source": "corpus/1402.1869v2.pdf"
    },
    {
      "id": "S5",
      "source": "corpus/1712.09913v3.pdf"
    }
  ],
  "usage": {
    "completion_tokens": 410,
    "prompt_tokens": 1592,
    "total_tokens": 2002,
    "completion_tokens_details": null,
    "prompt_tokens_details": null
  },
  "latency_ms": 16701
}
```

## Observations

The model literally returned "<context>...</context>" as if it was copying a textbook definition. It got completely confiused with the template styling. This is the downside of using a local model rather than one from open AI/ Google.

`"latency_ms": 16701` -> This is quite high because Ollama had to read the model weights from SSD to VRAM cache as this was a cold start.

## Tweaks

Adding this line to system prompt helped get rid of the XML tags -

Never include raw strings like '<context>' or 'source=' inside your text.

```json
 "answer": "Deep neural networks are a type of neural network that has become increasingly popular due to their unprecedented success in various machine learning tasks [S3]. They consist of multiple layers of artificial neurons, inspired by the structure and function of biological neural networks. The key characteristic of deep neural networks is the presence of many layers, typically more than five, which allows them to learn complex representations of data.\n\nDeep neural networks are general function approximators, meaning they can be applied to a wide range of image and computer vision problems [S2]. They have been shown to excel in tasks such as recognizing patterns, anomalies, and making predictions. Some popular architectures of deep neural networks include Convoluted Neural Networks (CNNs), which were developed by Yann LeCun and his associates in 1998 [S2].\n\nThe success of deep neural networks can be attributed to their ability to learn complex mappings from input data to output space. This is achieved through the use of multiple layers, each with its own set of artificial neurons, which process and transform the input data in a hierarchical manner.\n\nIn recent years, there has been an increasing interest in understanding the theoretical properties of deep neural networks, including their approximation capabilities, complexity bounds, and optimization landscapes [S3, S4, S5]. Researchers have made significant progress in this area, developing new techniques and frameworks for analyzing and improving the performance of deep neural networks.\n\nOverall, deep neural networks have revolutionized the field of machine learning and image analytics, offering unparalleled success in a wide range of applications.",
...
"latency_ms": 13513
```

## Observations

The latency is still high. For a mac mini, this should've taken around 2-3 seconds. 

There are a couple of tweaks that are left - 

1. Prompt caching in OLLAMA - Instead of running pipeline as a script, its better to run a persistent server sessions via MCP/ FAST API so that OLLAMA cache stays warm.
2. Low VRAM Mode Check - OLLAMA is running in low vram mode
3. No acceleration enabled in OLLAMA - Flash attention enablement will lead to a massive boost. "OLLAMA_FLASH_ATTENTION=true ollama serve"

## API support via FAST API

```
(.venv) kmadaan@Kartiks-Mac-mini personal-rag-mcp % curl -s -X POST localhost:8088/query \
 -H 'content-type: application/json' \
 -d '{"q":"how does neural network work?","k":4}' | jq
{
  "answer": "To understand how neural networks work, we need to delve into the basics of these models.\n\nNeural Networks are general function approximations inspired by biological neural networks. They can be applied to any image and computer vision problem where the goal is to learn a complex mapping from input to output space (S1).\n\nThe top tasks that can be accomplished with neural networks include recognizing patterns, anomalies, and making predictions (S1). Convoluted Neural Networks (CNNs) are one of the most popular architectures used in image analytics. They were developed by Yann LeCun and his associates in 1998 and use back propagation in a feedforward net with multiple hidden layers, many maps of replicated units in each layer, and pooling the outputs of nearby replicated units (S1).\n\nHowever, understanding how neural networks work requires more than just knowing their architectures. Visualizations have been proposed to help answer questions about why neural networks work, such as minimizing highly non-convex neural loss functions and generalization properties (S2). The study explores how different network architecture choices affect the loss landscape and proposes a simple \"filter normalization\" scheme for side-by-side comparisons of different minima found during training.\n\nNeural networks have also been shown to be robust to quantization, meaning they can be quantized to lower bit-widths with relatively small impact on accuracy (S3). However, low-bit-width quantization introduces noise that can lead to a drop in accuracy. The study introduces the state-of-the-art in neural network quantization, discussing hardware and practical considerations, as well as two different regimes of quantizing neural networks: Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).\n\nIn terms of architecture, Convoluted Neural Networks (CNNs) are primarily used in image processing and classification but can also be used across other input types such as sound and audio. They involve convoluted layers instead of traditional normal layers, where not all nodes are connected to all nodes. These layers have a tendency to shrink when they become deeper. CNNs also involve pooling layers, such as max pooling, which filters out details and creates a downsampled feature map.\n\nRecurrent Neural Networks (RNNs) were originally introduced by Jeffrey Elman in 1990 and are basically perceptrons with connections between passes and through time. They combine two key properties: distributed hidden state that allows them to store information about the past efficiently, and non-linear dynamics that allow them to update their hidden state in a complicated manner.\n\nIn summary, neural networks work by learning complex mappings from input to output space through general function approximations inspired by biological neural networks. Understanding how they work requires knowledge of architectures such as CNNs and RNNs, as well as the principles behind quantization and visualization methods for loss functions.\n\nI don't know based on the provided context.",
  "citations": [
    {
      "id": "S1",
      "source": "corpus/White+paper+1+-+Introduction+to+Image+analytics.pdf"
    },
    {
      "id": "S2",
      "source": "corpus/1712.09913v3.pdf"
    },
    {
      "id": "S3",
      "source": "corpus/2106.08295v1.pdf"
    },
    {
      "id": "S4",
      "source": "corpus/White+paper+1+-+Introduction+to+Image+analytics.pdf"
    }
  ],
  "usage": {
    "completion_tokens": 579,
    "prompt_tokens": 1630,
    "total_tokens": 2209,
    "completion_tokens_details": null,
    "prompt_tokens_details": null
  },
  "latency_ms": 21155
}
```

## Observations

The latency is still high. The reason for this seems to be prompt cache eviction. The tokens generated seem to be higher than the context window limit. 

## Tweak 

Increase the context memory (by sacrificing VRAM) so that the model can retain a larger cache.

