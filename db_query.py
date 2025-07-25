import argparse
from sentence_transformers import SentenceTransformer
import pinecone
from openai import OpenAI
import json
import gzip

### CONFIG ###
PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
PINECONE_ENV = "YOUR_REGION"
INDEX_NAME = "dna-hmm-kozak"
LLM_MODEL = "gpt-4o"
TOKEN_LIMIT = 3500   # leave room for question + response (~500 tokens buffer)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
openai_client = OpenAI()

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

### HELPERS ###
def embed(text):
    return embed_model.encode(text).tolist()

def load_large_array(blob):
    return json.loads(gzip.decompress(blob).decode("utf-8"))

def estimate_tokens(text):
    """Rough token estimate: ~4 chars/token"""
    return len(text) // 4

def retrieve_dynamic(query_text, max_k=20, min_score=0.75, organism=None):
    """
    Retrieve up to max_k results, but stop early if score drops below min_score.
    """
    query_vector = embed(query_text)

    filter_ = {"organism": organism} if organism else None

    results = index.query(
        vector=query_vector,
        top_k=max_k,
        include_metadata=True,
        filter=filter_
    )

    selected = []
    for i, r in enumerate(results.matches, 1):
        if r.score < min_score:
            print(f"[!] Stopping at rank {i} — score {r.score:.2f} < min_score {min_score}")
            break
        selected.append(r)

    return selected

def build_context(records, token_limit=TOKEN_LIMIT):
    """
    Build a context string that stays under token_limit.
    """
    context = ""
    tokens = 0
    for i, r in enumerate(records, 1):
        meta = r.metadata
        consensus = json.loads(meta['consensus_kozak'])
        posterior = load_large_array(meta['posterior_compressed'])

        snippet = (
            f"--- Record {i} ---\n"
            f"ID: {r.id}\n"
            f"Organism: {meta['organism']}\n"
            f"Sequence (start): {meta['sequence'][:50]}...\n"
            f"Consensus Kozak: {consensus}\n"
            f"Posterior (first 10): {posterior[:10]}\n\n"
        )
        snippet_tokens = estimate_tokens(snippet)
        if tokens + snippet_tokens > token_limit:
            print(f"[!] Stopping at {i-1} records to stay under token limit.")
            break
        context += snippet
        tokens += snippet_tokens

    print(f"[+] Context: ~{tokens} tokens")
    return context

def ask_llm(user_question, context):
    prompt = f"""
You are a bioinformatics assistant with access to a database of DNA sequences, their forward–backward HMM results (posterior probabilities for locations of the start codon), and annotations on were the start codon was previously assumed to be located.

Here are some relevant records:
{context}

Question:
{user_question}

Answer the question based only on the records above. If the answer is unclear, say so explicitly. Provide reasoning when appropriate.
"""
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful bioinformatics expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

### MAIN ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="Natural language question about the dataset")
    parser.add_argument("--organism", type=str, help="Optional: filter by organism")
    parser.add_argument("--max_k", type=int, default=1000, help="Maximum records to consider")
    parser.add_argument("--min_score", type=float, default=0.75, help="Stop if score falls below this")
    args = parser.parse_args()

    print(f"[+] Retrieving context for question: {args.question}")
    if args.organism:
        print(f"[+] Filtering by organism: {args.organism}")

    records = retrieve_dynamic(
        args.question,
        max_k=args.max_k,
        min_score=args.min_score,
        organism=args.organism
    )

    if not records:
        print("[!] No relevant records found above score threshold.")
        exit(0)

    context = build_context(records, token_limit=TOKEN_LIMIT)

    print(f"[+] Asking LLM...")
    answer = ask_llm(args.question, context)

    print("\n[+] Answer:\n")
    print(answer)
