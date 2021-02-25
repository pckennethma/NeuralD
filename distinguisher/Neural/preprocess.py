from os import write
import numpy as np
import copy, sys, pickle
from tqdm import tqdm

def get_longest_common_subseq(data):
    substr = []
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and is_subseq_of_any(data[0][i:i+j], data):
                    substr = data[0][i:i+j]
    return substr

def is_subseq_of_any(find, data):
    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if not is_subseq(find, data[i]):
            return False
    return True

# Will also return True if possible_subseq == seq.
def is_subseq(possible_subseq, seq):
    if len(possible_subseq) > len(seq):
        return False
    def get_length_n_slices(n):
        for i in range(len(seq) + 1 - n):
            yield seq[i:i+n]
    for slyce in get_length_n_slices(len(possible_subseq)):
        if slyce == possible_subseq:
            return True
    return False

def first_preprocess(examples, sample_num=30):
    substitution = {} # special token -> lengthy substring
    spec_count = 0
    is_splitted = False
    if np.mean([len(e.split()) for e in examples]) > 200:
        is_splitted = True
        new_exp = []
        for e in examples:
            e_list = e.split()
            for i in range(len(e_list)//150):
                start = i * 150
                new_exp.append(" ".join(e_list[start:start+150]))
        old_example = examples
        examples = new_exp
        sample_num *= 3
        print(examples[:5])
    if sample_num * 3 < len(examples):
        triplets = np.random.choice(examples, size=(sample_num, 3), replace=False)
    else:
        triplets = np.random.choice(examples, size=(sample_num, 3))

    for triplet in tqdm(triplets):
        new_t = []
        for seq in triplet:
            for k, v in substitution.items():
                if v in seq:
                    seq = seq.replace(v, " " + k + " ")
            new_t.append(seq.split())
        
        common = get_longest_common_subseq(new_t)
        if len(common) < 20: continue
        substitution[f"special_tok{spec_count}"] = " ".join(common).strip()
        spec_count += 1
    substitution = raw_substitution(substitution)
    print(len(substitution))
    new_examples = []
    if is_splitted:
        examples = old_example
    for e in tqdm(examples):
        for k, v in substitution.items():
            e = e.replace(v, k)
        new_examples.append(e)
    return new_examples, substitution

def preprocess(examples, substitution):
    new_examples = []
    for e in tqdm(examples):
        for k, v in substitution.items():
            e = e.replace(v, k)
        new_examples.append(e)
    return new_examples

def raw_substitution(subst: dict):

    def lookup(token: str):
        if "special_tok" not in subst[token]:
            return subst[token]
        else:
            seq = []
            for tok in subst[token].split():
                if "special_tok" not in tok:
                    seq.append(tok)
                else:
                    seq.append(lookup(tok))
            return " ".join(seq).strip()
    return {k:lookup(k) for k in subst}


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "first":
        with open(sys.argv[1]) as f:
            lines = f.readlines()
        raw_examples = [(l[:11], l[11:]) for l in lines]
        raw_text = [i[1].strip() for i in raw_examples]
        raw_labels = [i[0] for i in raw_examples]
        examples, subs = first_preprocess(raw_text, 1000)
        with open(sys.argv[1], "w") as f:
            for i in range(len(examples)):
                label = raw_labels[i]
                text = examples[i]
                f.write(f"{label} {text}\n")
        with open(sys.argv[2]) as f:
            lines = f.readlines()
        raw_examples = [(l[:11], l[11:]) for l in lines]
        raw_text = [i[1].strip() for i in raw_examples]
        raw_labels = [i[0] for i in raw_examples]
        examples = preprocess(raw_text, subs)
        with open(sys.argv[2], "w") as f:
            for i in range(len(examples)):
                label = raw_labels[i]
                text = examples[i]
                f.write(f"{label} {text}\n")
        with open("subs.bin", "wb") as f:
            pickle.dump(subs, f)
    else:
        with open("subs.bin", "rb") as f:
            subs = pickle.load(f)
        with open(sys.argv[1]) as f:
            lines = f.readlines()
        raw_examples = [(l[:11], l[11:]) for l in lines]
        raw_text = [i[1] for i in raw_examples]
        raw_labels = [i[0] for i in raw_examples]
        examples = preprocess(raw_text, subs)
        with open(sys.argv[1], "w") as f:
            for i in range(len(examples)):
                label = raw_labels[i]
                text = examples[i]
                f.write(f"{label} {text}\n")
        with open(sys.argv[2]) as f:
            lines = f.readlines()
        raw_examples = [(l[:11], l[11:]) for l in lines]
        raw_text = [i[1] for i in raw_examples]
        raw_labels = [i[0] for i in raw_examples]
        examples = preprocess(raw_text, subs)
        with open(sys.argv[2], "w") as f:
            for i in range(len(examples)):
                label = raw_labels[i]
                text = examples[i]
                f.write(f"{label} {text}\n")