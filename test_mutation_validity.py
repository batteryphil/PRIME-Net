import numpy as np

def generate_valid_rpn(vocab, max_len, rng):
    leaves = [t for t in vocab if (0 <= t <= 12) or t >= 50]
    unaries = [t for t in vocab if 24 <= t <= 30]
    binaries = [t for t in vocab if 20 <= t <= 23]
    
    seq = [rng.choice(leaves)]
    
    while len(seq) < max_len:
        if rng.random() < 0.2: # 20% chance to stop early
            break
            
        if rng.random() < 0.4 and len(seq) + 1 <= max_len: # Add unary
            seq.append(rng.choice(unaries))
        elif len(seq) + 2 <= max_len: # Add leaf and binary
            seq.append(rng.choice(leaves))
            seq.append(rng.choice(binaries))
        else:
            break
            
    padded = np.full(max_len, -1, dtype=np.int32)
    padded[:len(seq)] = seq
    return padded

rng = np.random.default_rng(42)
num_vars = 5
base_vocab = list(range(num_vars)) + [10, 11, 12] + list(range(20, 24)) + list(range(24, 31)) + [-1]
comp_vocab = np.array(base_vocab, dtype=np.int32)

def is_valid_rpn(rpn):
    sp = 0
    active = 0
    for token in rpn:
        if token == -1:
            continue
        active += 1
        if 0 <= token <= 12 or token >= 50:
            sp += 1
        elif 20 <= token <= 23:
            if sp < 2:
                return False
            sp -= 1
        elif 24 <= token <= 30:
            if sp < 1:
                return False
        else:
            return False
    return sp == 1 and active > 0

# Test 10,000 unguided random mutations
n_trials = 10000
invalid_count = 0

for _ in range(n_trials):
    parent = generate_valid_rpn(comp_vocab, 15, rng)
    child = parent.copy()
    mut_idx = rng.integers(0, 15)
    child[mut_idx] = rng.choice(comp_vocab)
    if not is_valid_rpn(child):
        invalid_count += 1

print(f"Total mutations: {n_trials}")
print(f"Invalid mutations: {invalid_count} ({invalid_count / n_trials * 100:.2f}%)")

def category_preserving_mutation(parent, vocab, rng):
    leaves = [t for t in vocab if (0 <= t <= 12) or t >= 50]
    unaries = [t for t in vocab if 24 <= t <= 30]
    binaries = [t for t in vocab if 20 <= t <= 23]

    active_indices = [i for i, t in enumerate(parent) if t != -1]
    if not active_indices:
        return parent.copy()

    mut_idx = rng.choice(active_indices)
    tok = parent[mut_idx]
    child = parent.copy()

    if (0 <= tok <= 12) or tok >= 50:
        child[mut_idx] = rng.choice(leaves)
    elif 20 <= tok <= 23:
        child[mut_idx] = rng.choice(binaries)
    elif 24 <= tok <= 30:
        child[mut_idx] = rng.choice(unaries)
    
    return child

# Test 10,000 category-preserving mutations
invalid_guarded_count = 0
for _ in range(n_trials):
    parent = generate_valid_rpn(comp_vocab, 15, rng)
    child = category_preserving_mutation(parent, comp_vocab, rng)
    if not is_valid_rpn(child):
        invalid_guarded_count += 1

print(f"Guarded mutations: {n_trials}")
print(f"Invalid guarded mutations: {invalid_guarded_count} ({invalid_guarded_count / n_trials * 100:.2f}%)")

def extract_random_subtree(seq, rng):
    valid_indices = [i for i, t in enumerate(seq) if t != -1]
    if not valid_indices: return None
    start_idx = rng.choice(valid_indices)
    req = 1
    for i in range(start_idx, -1, -1):
        t = seq[i]
        if t == -1: continue
        elif 20 <= t <= 23: req += 1
        elif 24 <= t <= 30: req += 0
        else: req -= 1
        if req == 0:
            return i, start_idx
    return None

def subtree_mutation(parent, vocab, max_len, rng):
    bounds = extract_random_subtree(parent, rng)
    if not bounds:
        return category_preserving_mutation(parent, vocab, rng)
    start_idx, end_idx = bounds
    
    active = [t for t in parent if t != -1]
    budget = max_len - (len(active) - (end_idx - start_idx + 1))
    if budget < 1:
        return category_preserving_mutation(parent, vocab, rng)
        
    sub_len = rng.integers(1, min(budget + 1, 7))
    new_sub = generate_valid_rpn(vocab, sub_len, rng)
    new_sub_active = [t for t in new_sub if t != -1]
    
    child = []
    for i in range(start_idx):
        if parent[i] != -1: child.append(parent[i])
    child.extend(new_sub_active)
    for i in range(end_idx + 1, len(parent)):
        if parent[i] != -1: child.append(parent[i])
        
    if len(child) > max_len or len(child) == 0:
        return category_preserving_mutation(parent, vocab, rng)
        
    padded = np.full(max_len, -1, dtype=np.int32)
    padded[:len(child)] = child
    return padded

# Test combined guarded mutation
invalid_combined = 0
for _ in range(n_trials):
    parent = generate_valid_rpn(comp_vocab, 15, rng)
    if rng.random() < 0.5:
        child = category_preserving_mutation(parent, comp_vocab, rng)
    else:
        child = subtree_mutation(parent, comp_vocab, 15, rng)
    if not is_valid_rpn(child):
        invalid_combined += 1

print(f"Combined guarded mutations: {n_trials}")
print(f"Invalid combined mutations: {invalid_combined} ({invalid_combined / n_trials * 100:.2f}%)")
