import numpy as np
import os, time, sys
from tqdm import tqdm

from util.fuzzer_util import TopkHeap, BlackBoxFuzzerConfig, encode_input_blackbox, random_pool, levenshtein

def get_prioritized_pair(rand_pool, maxsize):
    heap = TopkHeap(maxsize)
    for i in tqdm(range(len(rand_pool))):
        for j in range(i+1, len(rand_pool)):
            dist = levenshtein(rand_pool[i], rand_pool[j])
            heap.Push((dist, (rand_pool[i], rand_pool[j])))
    return heap.TopK()

def extend_dataset(input_pair, candidate_file, file_path, num, is_train=True):
    config_path1 = os.path.join(file_path, "config1.txt")
    config_path2 = os.path.join(file_path, "config2.txt")
    with open(config_path1, "w") as f:
        f.write(input_pair[0])
    with open(config_path2, "w") as f:
        f.write(input_pair[1])

    out_path = os.path.join(file_path, "train.txt" if is_train else "val.txt")
    command1 = f"./{candidate_file} {config_path1} {out_path} __label__1"
    command2 = f"./{candidate_file} {config_path2} {out_path} __label__2"
    
    for i in tqdm(range(num)):
        os.system(command1)
        os.system(command2)

def train_and_val(model:str, file_path="temp"):
    train_file_path = os.path.join(file_path, "train.txt")
    val_file_path = os.path.join(file_path, "val.txt")
    if os.path.exists(os.path.join(file_path, "model.bin")):
        os.system(f"python distinguisher/Neural/preprocess.py {train_file_path} {val_file_path} ")
    else: 
        os.system(f"python distinguisher/Neural/preprocess.py {train_file_path} {val_file_path} first")
    if model == "CNN":
        abs_path = os.path.abspath(file_path)
        os.system(f"nvidia-docker run -it --rm -v {abs_path}:/data oram-model python train.py")
        result_path = os.path.join(abs_path, "result.txt")
        if os.path.exists(result_path):
            time.sleep(0.1) # avoid file racing
            with open(result_path) as f:
                precision = float(f.read().split(",")[1]) / 100
            return precision
        raise FileNotFoundError()
    else:
        raise NotImplementedError()


def flush_obfuscated_sequence(file_path="temp"):
    os.system(f"rm -f {file_path}/*.txt")
    os.system(f"rm -f {file_path}/*.bin")

def main(config: BlackBoxFuzzerConfig):
    start = time.time()
    rand_pool = random_pool(config.random_generator, config.random_generator_args, config.random_input_pool_size)
    rand_pool = list(map(encode_input_blackbox, rand_pool))
    heap = get_prioritized_pair(rand_pool, maxsize=config.maximal_trail_num)

    file_path = config.file_path
    candidate_solution_file = config.candidate_solution_file
    train_num = config.train_num

    assert os.path.exists(candidate_solution_file)

    is_nonbliviousness = False

    while len(heap) != 0:
        val_num = config.init_val_num
        curr_val_num = val_num
        dist, input_pair = heap.pop(0)
        print(input_pair, "dist:", dist)
        extend_dataset(input_pair, candidate_solution_file, file_path, num=train_num, is_train=True)
        extend_dataset(input_pair, candidate_solution_file, file_path, num=val_num, is_train=False)
        while curr_val_num < 400000:
            precision = train_and_val("CNN", file_path=file_path)
            if precision - 0.5 < 1e-5:
                print(f"indistinguishable w.h.p. on {input_pair} with accuracy of {precision}")
                break
            # p = 0.7
            elif precision > 0.5 + np.sqrt(0.602/curr_val_num):
                print(f"distinguishable w.h.p. on {input_pair} with accuracy of {precision}")
                model_path = os.path.join(file_path, "model.bin")
                print(f"model saved at {model_path}")
                is_nonbliviousness = True
                break
            else:
                # TODO: prograssively increase 
                print(f"extend dataset and train again with accuracy of {precision}")
                extend_dataset(input_pair, candidate_solution_file, file_path, num=int(0.1*train_num), is_train=True)
                extend_dataset(input_pair, candidate_solution_file, file_path, num=int(curr_val_num), is_train=False)
                curr_val_num += curr_val_num
        if is_nonbliviousness:
            break
        else:
            flush_obfuscated_sequence(file_path)
            
        
    if not is_nonbliviousness: print("timeout")
    print(time.time()-start)


if __name__ == "__main__":
    config = BlackBoxFuzzerConfig(sys.argv[1])
    main(config)