import numpy as np

def get_valid_subtrees(seq):
    subtrees = []
    n = len(seq)
    for i in range(n):
        if seq[i] >= 12 and seq[i] != 19:
            continue
            
        stack_depth = 0
        valid = True
        
        for j in range(i, n):
            token = seq[j]
            if token == 19:
                pass
            elif token < 12:
                stack_depth += 1
            elif token >= 12 and token <= 15:
                stack_depth -= 1
                if stack_depth <= 0:
                    valid = False
                    break
            elif token >= 16 and token <= 18:
                if stack_depth <= 0:
                    valid = False
                    break
                    
            if valid and stack_depth == 1:
                # We only want composite sub-trees (length >= 3)
                clean_seq = [t for t in seq[i:j+1] if t != 19]
                if len(clean_seq) >= 3:
                    subtrees.append(clean_seq)
                    
    # Deduplicate raw sequences
    unique = list(set([tuple(x) for x in subtrees]))
    return [list(x) for x in unique]

def canonicalize_subtree(clean_seq):
    stack = []
    
    def format_token(t):
        if t < 6: return f"X{t+1}"
        elif t < 12: return f"C{t-6}"
        elif t == 12: return "*"
        elif t == 13: return "/"
        elif t == 14: return "+"
        elif t == 15: return "-"
        elif t == 16: return "sin"
        elif t == 17: return "cos"
        elif t == 18: return "exp"
        return str(t)
        
    for token in clean_seq:
        if token < 12:
            stack.append(format_token(token))
        elif token >= 12 and token <= 15:
            right = stack.pop()
            left = stack.pop()
            
            # Commutative operators
            if token == 12 or token == 14:
                if right < left:
                    left, right = right, left
                    
            op = format_token(token)
            stack.append(f"({left}{op}{right})")
        elif token >= 16 and token <= 18:
            operand = stack.pop()
            op = format_token(token)
            stack.append(f"{op}({operand})")
            
    return stack[0]

def test_extraction():
    print("================================================================================")
    print(" PHASE II-A: SUB-TREE EXTRACTION & HASHING")
    print("================================================================================\n")
    
    archive = {}
    
    def add_to_archive(seq, name):
        print(f"[*] Processing {name}: {seq}")
        subtrees = get_valid_subtrees(seq)
        for st in subtrees:
            hash_key = canonicalize_subtree(st)
            if hash_key not in archive:
                archive[hash_key] = {"raw_seq": st, "count": 1}
                print(f"    -> [NEW] Added Sub-tree: {hash_key} (Raw: {st})")
            else:
                archive[hash_key]["count"] += 1
                print(f"    -> [DEDUPE] Incremented Count: {hash_key}")
        print("-" * 60)
        
    # Sequence 1: X1 * X2 (Raw)
    add_to_archive([0, 1, 12, 19, 19], "X1*X2")
    
    # Sequence 2: X2 * X1 (Commutative Equivalent)
    add_to_archive([19, 1, 0, 12, 19], "X2*X1")
    
    # Sequence 3: X3 * X3 + X1 * X2 (Complex composite)
    # X3 X3 * X1 X2 * +
    # [2, 2, 12, 0, 1, 12, 14]
    add_to_archive([2, 2, 12, 0, 1, 12, 14], "Composite A")
    
    # Sequence 4: Full Newton (X1*X2)/(X3*X3)
    # [0, 1, 12, 2, 2, 12, 13]
    add_to_archive([0, 1, 12, 2, 2, 12, 13], "Full Newton")
    
    print("\nFINAL ARCHIVE STATE:")
    print("-" * 60)
    for k, v in archive.items():
        print(f"Hash: {k:<20} | Count: {v['count']} | Raw: {v['raw_seq']}")

if __name__ == "__main__":
    test_extraction()
