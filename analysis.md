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

By creating both, pgvector is fully prepared for Hybrid Search with Reciprocal Rank Fusion (RRF). When a user asks a question, the pipeline will fire two ultra-fast queries simultaneously:The HNSW index retrieves chunks that match the concept of the question.The GIN index retrieves chunks that match the exact keywords or codes
