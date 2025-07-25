! pip install sentence-transformers pinecone-client
from sentence_transformers import SentenceTransformer
from HMM import forward_backward, parse_fasta, get_consensus_annotation

import argparse
import pinecone
import gzip
import json

parser = argparse.ArgumentParser(desription="Uploads sequences and state probabilities to PineconeDB")
parser.add_argument("fasta", type=str, help="Path to fasta file containing sequences you want to analyze")
parser.add_argument("-API_KEY", type=str, default="",help="PineconeDB API Key")
parser.add_argument("-REGION", type=str, default="", help="PineconeDB Environment")
parser.add_argument("-index", type=str, default="dna-hmm-kozak", help="PinceconeDB index name")
parser.add_argument("-organism", type=str, default="human", help="type of organism that .fa file comes from")
parser.add_argument("-seq_type", type=str, default="cDNA", help="type of genetic sequence (cDNA, full transcript, etc.)")

arg = parser.parse_args()

#Initialize embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

#Init Pinecone
pinecone.init(api_key=arg.API_KEY, environment=arg.REGION)
index_name = arg.index

#Create index if name does not exist
if index_name not in pinecone.list_indexes(): pinecone.create_index(index_name, dimension=384, metric="cosine")
index = pinecone.Index(index_name)

def compress_array(array):
	"""Compress and serialize large arrays"""
	compressed = gzip.compress(json.dumps(array).encode('utf-8'))
	return compressed

def load_array(array):
	"""Decompress and load arrays"""
	return json.loads(gzip.decompress(array).decode('utf-8'))
    
def summarize(seq_obj, length, consensus):
    """Summary text for embedding"""
    text = (
        f"Sequence {seq_obj.id} from {seq_obj.organism} with a length of {length} nucleotides. "
        f"Start codon assumed to be at {consensus}.")
    return text

for seq in readfasta(arg.fasta):
    sequence = seq[1]
    name = seq[0]
    hmm_result = forward_backward(seq.sequence)
    consensus = get_consensus_location(seq.sequence)
    length = seq.length
    
    #summary
    summary_text = summarize(seq, length, consensus)
    
    #embedding
    vector = embed_model.encode(summary_text).tolist()
    
    #metadata
    metadata = {
        "organism": arg.organism,
        "sequence": sequence,
        "id": name,
        "assumed_start_codon": json.dumps(consensus),
        "sequence_type": arg.seq_type,
    }
    
    #Payload (compress probability array)
    payload = {
        "sequence": seq.sequence,
        "states": hmm_result.states,
        "posterior_compressed": save_large_array(hmm_result.posterior),
    }
    
    #Combine metadata and payload
    full_data = {**metadata, **payload}
    
    #Upsert into pinecone
    index.upsert([
        (f"{seq.organism}_{seq.id}", vector, full_data)
    ])
        
        
        
        
        

