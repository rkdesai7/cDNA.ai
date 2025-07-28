import os
import json
import gzip
import pinecone
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough
from langchain_core.runnables.graph import StateGraph
from langchain_core.output_parsers import StrOutputParser

### CONFIG ###
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "YOUR_PINECONE_API_KEY"
PINECONE_ENV = os.getenv("PINECONE_ENV") or "YOUR_REGION"
INDEX_NAME = "dna-hmm-kozak"
LLM_MODEL = "gpt-4o"
MODEL_NAME = "all-MiniLM-L6-v2"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

embedding_model = SentenceTransformer(model_name=MODEL_NAME)
vectorstore = LC_Pinecone(index, embedding_model.encode, "text")
llm = ChatOpenAI(model_name=LLM_MODEL)

### Prompt Template ###
prompt_template = PromptTemplate.from_template("""
You are a helpful bioinformatics assistant. You have access to DNA sequences, their forward–backward HMM posterior probabilities for Kozak/not‑Kozak states, and annotated consensus Kozak regions.

{context}

{question}
""")

output_parser = StrOutputParser()

### FASTA Parser ###
def parse_fasta(fasta_path):
    with open(fasta_path, "r") as f:
        header = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    yield (header, ''.join(seq))
                header = line[1:]
                seq = []
            else:
                seq.append(line)
        if header:
            yield (header, ''.join(seq))

### Compression ###
def decode_posterior(blob):
    return json.loads(gzip.decompress(blob).decode("utf-8"))

def encode_posterior(array):
    return gzip.compress(json.dumps(array).encode("utf-8"))

### Indexing Function ###
def index_fasta(fasta_path, organism="unknown"):
    for seq_id, sequence in parse_fasta(fasta_path):
        summary = f"Sequence {seq_id} from {organism}: {sequence[:20]}..."
        vector = embedding_model.encode(summary).tolist()
        posterior = [[0.8, 0.2]] * len(sequence)  # MOCK POSTERIOR
        consensus = [[5, 12]]                    # MOCK KOZAK REGION

        index.upsert([
            (f"{organism}_{seq_id}", vector, {
                "sequence": sequence,
                "posterior_compressed": encode_posterior(posterior),
                "organism": organism,
                "consensus_kozak": json.dumps(consensus)
            })
        ])
    print(f"✅ Indexed sequences from {fasta_path}")

### Prediction Detection ###
def detect_prediction(user_input):
    lowered = user_input.lower()
    if "predict" in lowered and "sequence:" in lowered:
        parts = user_input.split("sequence:")
        return parts[-1].strip()
    return None

### LangGraph Definition ###
def build_graph():
    def retrieve_fn(state):
        query = state["question"]
        filter_ = {"organism": state["organism"]} if state.get("organism") else None
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15, "filter": filter_})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(d.page_content if hasattr(d, "page_content") else str(d.metadata) for d in docs)
        return {"context": context, **state}

    def prompt_fn(state):
        prompt = prompt_template.format(**state)
        return {"prompt": prompt, **state}

    def llm_fn(state):
        return {"response": llm.invoke(state["prompt"]), **state}

    graph = StateGraph()
    graph.add_node("retrieve", RunnableLambda(retrieve_fn))
    graph.add_node("prompt", RunnableLambda(prompt_fn))
    graph.add_node("llm", RunnableLambda(llm_fn))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "prompt")
    graph.add_edge("prompt", "llm")
    graph.set_finish_point("llm")

    return graph.compile()

### Run Loop ###
graph = build_graph()

print("\nBioinformatics Assistant with LangGraph (type 'exit' to quit, or 'upload <path>' to index a FASTA file)")
while True:
    user_input = input("\nPrompt: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        print("Exiting.")
        break

    if user_input.lower().startswith("upload"):
        _, file_path = user_input.split(maxsplit=1)
        organism = input("Organism name: ").strip()
        if os.path.exists(file_path):
            index_fasta(file_path, organism=organism)
        else:
            print("[!] File not found.")
        continue

    organism = input("Organism filter (or leave blank): ").strip()
    new_sequence = detect_prediction(user_input)
    if new_sequence:
        question = f"Here is a new sequence from {organism or 'unknown'}:\n{new_sequence}\n\nBased on the records above, predict the Kozak region(s) and explain."
    else:
        question = user_input

    result = graph.invoke({"question": question, "organism": organism})
    print("\nAnswer:\n")
    print(result["response"])

- add recursivecharactertext splitter to split documents
	- textsplitter
- then embed contents of each split and insert into vector store: vector_store
- Lang graph?

