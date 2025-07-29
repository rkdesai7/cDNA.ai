import os
import json
import gzip
import subprocess
import tempfile
import pinecone
import streamlit as st
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.graph import StateGraph
from langchain_core.output_parsers import StrOutputParser

# --- CONFIG ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or "YOUR_PINECONE_API_KEY"
PINECONE_ENV = os.getenv("PINECONE_ENV") or "YOUR_REGION"
INDEX_NAME = "dna-hmm-kozak"
MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    return LC_Pinecone(index, get_embedding_model().encode, "text")

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatOpenAI(model_name=LLM_MODEL)

embedding_model = get_embedding_model()
vectorstore = get_vectorstore()
llm = get_llm()

prompt_template = PromptTemplate.from_template("""
You are a helpful bioinformatics assistant. You have access to DNA sequences, their forward–backward HMM posterior probabilities for Kozak/not‑Kozak states, and annotated consensus Kozak regions.

{context}

{question}
""")

# --- UTILS ---
def parse_fasta(text):
    lines = text.splitlines()
    header, seq = None, []
    for line in lines:
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

def encode_posterior(array):
    return gzip.compress(json.dumps(array).encode("utf-8"))

def decode_posterior(blob):
    return json.loads(gzip.decompress(blob).decode("utf-8"))

def generate_hmm_model(fasta_path, organism):
    os.makedirs("models", exist_ok=True)
    output = f"models/model_{organism}.json"
    subprocess.run(["python", "gen_HMM.py", "--input", fasta_path, "--output", output], check=True)
    return output

def run_forward_backward(seq, model_path):
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".fa") as f:
        f.write(f">query\n{seq}\n")
        fasta_path = f.name
    result = subprocess.run(
        ["python", "forward_backward.py", "--input", fasta_path, "--model", model_path],
        capture_output=True, text=True
    )
    os.remove(fasta_path)
    if result.returncode != 0:
        raise RuntimeError(f"HMM failed: {result.stderr}")
    data = json.loads(result.stdout)
    return data["posterior"], data.get("consensus", [])

def index_fasta(text, organism, update_hmm):
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".fa") as f:
        f.write(text)
        fasta_path = f.name

    if update_hmm or not os.path.exists(f"models/model_{organism}.json"):
        hmm_model_path = generate_hmm_model(fasta_path, organism)
    else:
        hmm_model_path = f"models/model_{organism}.json"

    for seq_id, sequence in parse_fasta(text.splitlines()):
        posterior, consensus = run_forward_backward(sequence, hmm_model_path)
        summary = f"Sequence {seq_id} from {organism}: {sequence[:20]}..."
        vector = embedding_model.encode(summary).tolist()
        index.upsert([
            (f"{organism}_{seq_id}", vector, {
                "sequence": sequence,
                "posterior_compressed": encode_posterior(posterior),
                "organism": organism,
                "consensus_kozak": json.dumps(consensus)
            })
        ])

    os.remove(fasta_path)
    return "✅ Indexed successfully."

# --- LangGraph ---
def build_graph():
    def retrieve_fn(state):
        query = state["question"]
        filter_ = {"organism": state["organism"]} if state.get("organism") else None
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10, "filter": filter_})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(str(d.metadata) for d in docs)
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

# --- Streamlit UI ---
st.set_page_config(page_title="Biochat Assistant", layout="wide")
st.title("🧬 Kozak Sequence Assistant")

with st.sidebar:
    st.header("📤 Upload FASTA")
    uploaded_file = st.file_uploader("Upload .fa file", type=["fa", "fasta"])
    organism = st.text_input("Organism", value="E_coli")
    update_hmm = st.checkbox("Regenerate HMM model", value=False)
    if uploaded_file and st.button("Index Sequences"):
        text = uploaded_file.read().decode()
        message = index_fasta(text, organism, update_hmm)
        st.success(message)

st.subheader("💬 Ask a Question or Prediction")
org = st.text_input("Filter by Organism", value="E_coli")
prompt = st.text_area("Your Question", height=150)
submit = st.button("Submit")

if submit and prompt:
    graph = build_graph()
    state = {"question": prompt, "organism": org}
    result = graph.invoke(state)
    st.markdown("### 🧠 LLM Answer")
    st.success(result["response"])

    st.markdown("### 🔍 Posterior Probability Plots")
    ids = [k for k in index.describe_index_stats()["namespaces"].get("", {}) if org in k]
    if ids:
        results = index.fetch(ids=ids)
        for k, v in results.items():
            metadata = v["metadata"]
            if metadata.get("organism") == org:
                posterior = decode_posterior(metadata["posterior_compressed"])
                kozak_probs = [p[0] for p in posterior]
                plt.figure(figsize=(8, 2))
                plt.plot(kozak_probs, label=f"{k}")
                plt.title(f"{k}")
                plt.xlabel("Position")
                plt.ylabel("P(Kozak)")
                plt.ylim(0, 1)
                plt.legend()
                st.pyplot(plt)
