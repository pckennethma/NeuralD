import os, time, sys
import numpy as np

def slice(file_path, new_file_path, start, end):
    conf_file = os.path.join(file_path, "config.json")
    new_conf_file = os.path.join(new_file_path, "config.json")
    os.system(f"cp {conf_file} {new_conf_file}")
    train_file_path = os.path.join(file_path, "train.txt")
    val_file_path = os.path.join(file_path, "val.txt")
    with open(train_file_path) as f:
        lines = f.readlines()
    train_examples = [(l[:11], l[11:].strip().split()) for l in lines]
    sliced_train_example = [(i[0], i[1][start: min(len(i[1])-1, end)]) for i in train_examples]

    with open(val_file_path) as f:
        lines = f.readlines()
    val_examples = [(l[:11], l[11:].strip().split()) for l in lines]
    sliced_val_example = [(i[0], i[1][start: min(len(i[1])-1, end)]) for i in val_examples]

    new_train_file_path = os.path.join(new_file_path, "train.txt")
    new_val_file_path = os.path.join(new_file_path, "val.txt")
    with open(new_train_file_path, "w") as f:
        for e in sliced_train_example:
            seq = " ".join(e[1])
            f.write(f"{e[0]} {seq}\n")
    with open(new_val_file_path, "w") as f:
        for e in sliced_val_example:
            seq = " ".join(e[1])
            f.write(f"{e[0]} {seq}\n")
    
    return len(sliced_val_example)

def train_val(file_path):
    abs_path = os.path.abspath(file_path)
    os.system(f"nvidia-docker run -it --rm -v {abs_path}:/data oram-model python train.py")
    result_path = os.path.join(abs_path, "result.txt")
    if os.path.exists(result_path):
        time.sleep(0.1) # avoid file racing
        with open(result_path) as f:
            precision = float(f.read().split(",")[1]) / 100
        return precision
    raise FileNotFoundError()

def get_max(file_path):
    train_file_path = os.path.join(file_path, "train.txt")
    with open(train_file_path) as f:
        lines = f.readlines()
    train_examples = [len(l[11:].strip().split()) for l in lines]
    return max(train_examples)

def delta_debugging(file_path):
    start, end = 0, get_max(file_path)
    delta = 8
    new_path = os.path.join(file_path, "new")
    all_new_path = os.path.join(new_path, "*")

    if not os.path.exists(new_path):
        os.mkdir(new_path)
    round = 0
    while end - start > delta:
        print(f"Round: #{round}: {start}, {end}")
        round += 1
        mid = (start + end) // 2
        val_num = slice(file_path, new_path, start, mid)
        p1 = train_val(new_path)
        os.system(f"rm -f {all_new_path}")

        val_num = slice(file_path, new_path, mid, end)
        p2 = train_val(new_path)
        os.system(f"rm -f {all_new_path}")

        if p1 < 0.5 + np.sqrt(1.498/val_num) and p2 < 0.5 + np.sqrt(1.498/val_num):
            break
        if p1 > 0.5 + np.sqrt(1.498/val_num) and p2 > 0.5 + np.sqrt(1.498/val_num):
            end = mid
        if p1 > 0.5 + np.sqrt(1.498/val_num) and p2 < 0.5 + np.sqrt(1.498/val_num):
            end = mid
        if p1 < 0.5 + np.sqrt(1.498/val_num) and p2 > 0.5 + np.sqrt(1.498/val_num):
            start = mid
    
    return start, end

if __name__ == "__main__":
    # new_path = os.path.join(sys.argv[1], "new")
    # slice(sys.argv[1], new_path, 0, 0.5)
    start, end = delta_debugging(sys.argv[1])
    print(start, end)
