import argparse
import json

parser = argparse.ArgumentParser(desription="Builds an HMM for identifying different parts of a cDNA")
parser.add_argument("5utr", type=str, help="Path to .fa containing 5' utr's")
parser.add_argument("3utr", type=str, help="Path to .fa containing 3' utr's")
parser.add_argument("cds", type=str, help="Path to .fa containing the coding sequence")
arg = parser.parse_args()

def compile_data(5utr, 3utr, cds):
    """Builds full cDNA sequence containing specified labels"""
    utr5s = readfasta(5utr)
    utr3s = readfasta(3utr)
    cds = readfasta(cds)
    
    dataset = []
    ids = set(utr5s) & set(cds) & set(utr3s)
    print(f"Found {len(ids)} whole cDNA sequences")
    
    for id in ids:
        utr5 = utr5s[id]
        utr3 = utr3s[id]
        cd = cds[id]
        
        cDNA = utr5 + cd + utr3
        labels = (["5utr"]*len(utr5) + ["start_codon"] * 3 + ["cds"] * len(cd[3:]) + ["3utr"] * len(utr3))
        
        assert len(cDNA) == len(labels), f"Mismatch for {id}"
        
        dataset.append((list(cDNA), labels))
        
    return dataset
    
def readfasta(file):

    """Reads fasta files"""
    seqs = {}
    with open(filename, 'r') as f:
        id = None
        seq = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.starswith(">"):
                if id is not None: seqs[id] = "".join(seq)
                id = line[1:].split()[0]
                seq = []
            else: seq.append(line.upper())
        if id is not None: seqs[id] = "".join(seq)
        
    return sequences
    
def train_HMM(dataset):
    """builds and HMM"""
    states = ["5utr", "start_codon", "cds", "3utr"]
    observations = ["A", "C", "G", "T"]
    
    #Initialize counts
    emissions = {s: {o: for o in observations} for s in states}
    transitions = {}
    starts = {s: 0 for s in states}
    state_lengths = {s: 0 for s in states}
    
    for seq, labels in dataset:
        prev_state = None
        starts[labels[0]] += 1
        
        for obs, state in zip(seq, labels):
            emissions[state][obs] += 1
            
            if prev_state is not None and state != prev_state:
                key = (prev_state, state)
                if key not in transitions:
                    transitions[key] = 0
                transitions[key] += 1
                
            prev_state = state
            state_lengths[state] += 1
    
    #normalize
    emission_probs = {}
    for state in states:
        total = sum(emissions[state][o] for o in observations)
        if total > 0: emission_probs[state] = {o: emissions[state][0]/total for o in observations}
        else: emission_probs[state] = {o: 0.0 for o in observations}
        
    transition_probs = {}
    for (s1, s2) in transitions:
        total_from_s1 = sum(
            count for (from_state, _), count in transitions.items() if from_state == s1
        )
        if total_from_s1 > 0:transition_probs[(s1, s2)] = transitions[(s1, s2)] / total_from_s1
        else: transition_probs[(s1, s2)] = 0.0

    # Normalize start probs
    total_starts = sum(starts[s] for s in states)
    if total_starts > 0: start_probs = {s: starts[s] / total_starts for s in states}
    else: start_probs = {s: 0.0 for s in states}
    
    #convert to JSON format
    hmm_model = {
        "states": states,
        "start": {},
        "transitions": {},
        "emissions": {}
    }

    # Start probabilities
    for state in states: hmm_model["start"][state] = start_probs.get(state, 0.0)

    # Transition probabilities
    for state_from in states:
        hmm_model["transitions"][state_from] = {}
        for state_to in states:
            hmm_model["transitions"][state_from][state_to] = transition_probs.get(
                (state_from, state_to), 0.0
            )

    # Emission probabilities
    for state in states:
        hmm_model["emissions"][state] = {}
        for obs in observations:
            hmm_model["emissions"][state][obs] = emission_probs.get(state, {}).get(obs, 0.0)
    
    with open("hmm_model.json", "w") as f: json.dump(hmm_model, f, indent=4)
    return hmm_model
    
dataset = compile_data(arg.5utr, arg.3utr, arg.cds)
HMM = train_HMM(dataset)
