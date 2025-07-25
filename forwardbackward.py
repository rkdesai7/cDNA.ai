import argparse
import json
import math
import numpy as np
import pandas as pd
import sys
import gzip

def readfasta(filename):
	
	name = None
	seqs = []
	
	fp = None
	if filename.endswith('.gz'): fp = gzip.open(filename, 'rt')
	elif filename == '-': fp = sys.stdin
	else: fp = open(filename)
	
	while True:
		line = fp.readline()
		if line == '': break
		line = line.rstrip()
		if line.startswith('>'):
			if len(seqs) > 0:
				seq = ''.join(seqs)
				yield(name, seq)
				name = line[1:]
				seqs = []
			else:
				name = line[1:]
		else:
			seqs.append(line)
	yield(name, ''.join(seqs))
	fp.close()

def log_dict(probs_dict):
    """tool for converting values in a dictionary to log probabilities"""
    log_probs = {}
    for from_state, to_probs in probs_dict.items():
        log_to_probs = {}
        for to_state, prob in to_probs.items():
            if prob > 0: log_prob = math.log(prob)
            else:        log_prob = -99
            log_to_probs[to_state] = log_prob
        log_probs[from_state] = log_to_probs
    
    return log_probs

def logsumexp(log_a, log_b):
    """tool to add probabilities in log space"""
    max_val = max(log_a, log_b)
    return max_val + math.log(np.exp(-abs(log_a-log_b)))

def
def HMM_loader(json_path):
    """reads HMM .json file into format/log-scale to be used in the alg"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    #convert to log-scale
    states = data["states"]
    transitions = log_dict(data["transitions"])
    emits = log_dict(data["emissions"])
    orders = []
    for i in states:
        dict = emits[i]
        order = len(list(dict.keys())[0]-1
        orders.append(order)
        
    return emits, orders, transitions, states
    
def forward_backward(sequence, emits, orders, transitions, states):
    """Run forward backward on a sequence"""
    
    #Forward fill
    forward_matrix = []
    p = math.log(1/len(states))
    for i in range(len(states)): forward_matrix.append([p])
    temp_emit = math.log(.25)
    #fill matrix
    for i in range(1, len(sequence)+1):
        for j in range(len(states)):
            sum = -np.inf
            for k in range(len(states)):
                if i < orders[j] + 1:
                    prev = forward_matrix[k][i-1]
                    prob = prev + transitions[states[k]][states[j]] + temp_emit
                    sum  = logsumexp(sum, prob)
                else:
                    seq = sequence[i-orders[j]-1:i]
                    prev = forward_matrix[k][i-1]
                    prob = prev+transitions[states[k]][states[j]]+emits[states[j]][seq]
                    sum  = logsumexp(sum, prob)
                    
            forward_matrix[j].append(sum)
    for i in forward_matrix: i = i.pop(0)
    
    #Backward Fill
    backward_matrix = []
    for i in range(len(states)):
        temp = [0]*(len(sequence))
        backward_matrix.append(temp)
    #fill matrix
    for i in range(len(sequence), -1, -1):
        if i == len(sequence): continue
        elif (i+1)==len(sequence):
            for j in backward_matrix:j[i] = 0
        else: 
            for j in range(len(states)):
                sum = -np.inf
                for k in range(len(states)):
                    if (i+1) < (orders[k] + 1):
                        prev = backward_matrix[k][i+1]
                        prob = transitions[states[j]][states[k]]]+prev+temp_emit
                        sum  = logsumexp(sum, prob)
                    else:
                        seq = sequence[i-(orders[k]-1):i+2]
                        prev = backward_matrix[k][i+1]
                        prob = transitions[states[j]][states[k]]+prev+emits[states[k]][seq]
                        sum  = logsumexp(sum, prob)
                backward_matrix[j][i] = sum
                
    #Decode
    true_probs = []
    for i in range(len(states)):
        true_probs.append([])
    for i in range(len(forward_matrix[0])):
        denominator = 0
        for k in range(len(states)):
            denominator = logsumexp(denominator, forward_matrix[k][i]+backward_matrix[k][i])
        for j in range(len(states)):
            numerator = forward_matrix[j][i]+backward_matrix[j][i]
            prob = numerator - denominator
            true_probs[j].append(prob)
    
    return true_probs
    
def run_forward_backward(file_path, json_path):
    """Run entire process"""
    emits, orders, transitions, states = HMM_loader(json_path)
    for seq in readfasta(file_path): forward_backward(seq, emits, orders, transitions, states)
 