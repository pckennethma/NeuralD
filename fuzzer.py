import numpy as np
import os, importlib, sys, time
from tqdm import tqdm

from Instrumentation.Instrumentation import instrumentation
from Instrumentation.util import flush, dump_hist
from distinguisher import fastText
from util.fuzzer_util import TopkHeap, FuzzerConfig, decode_input_str, encode_input, prepare_instrumentation_env, random_pool

def dynamic_analysis(execution_file, execution_inputs, lib_files, vm_args={}, white_list={}):
    prepare_instrumentation_env(lib_files=lib_files)
    # instrumentation
    max_bb_no = instrumentation(execution_file, f"Instrumentation/temp/InstVM.py",  white_list)
    VirtualMemory = importlib.import_module("Instrumentation.temp.InstVM").VirtualMemory

    hist_result = {}
    flush(max_bb_no)
    for execution_input in tqdm(execution_inputs):
        # generate execution input in str format
        execution_input_str = encode_input(execution_input)
        if execution_input_str in hist_result: continue
        # repreat 10 times
        for i in range(10):
            vm = VirtualMemory(**vm_args)
            for access_entry in execution_input:
                if access_entry[0] == "R":
                    vm.read(access_entry[1])
                else:
                    vm.write(access_entry[1])
        hist_result[execution_input_str] = dump_hist()
        flush(max_bb_no)
    
    os.system("rm -rf Instrumentation/temp/*")
    return hist_result

def get_prioritized_pair(hist_result, maxsize = 20):
    heap = TopkHeap(maxsize)
    inputs = list(hist_result.keys())
    for i in range(len(inputs)):
        for j in range(i+1, len(inputs)):
            dist = np.sum(np.abs(hist_result[inputs[i]] - hist_result[inputs[j]]))
            heap.Push((dist, (decode_input_str(inputs[i]), decode_input_str(inputs[j]))))
    return heap.TopK()

def _synthesize_obfuscated_sequence(input, vm, vm_args, label, file_path, num):
    for i in tqdm(range(num)):
        VirtualMemory = importlib.import_module(vm).VirtualMemory
        temp_vm = VirtualMemory(**vm_args)
        for access_entry in input:
            if access_entry[0] == "R": temp_vm.read(access_entry[1])
            else: temp_vm.write(access_entry[1])
        temp_vm.dump_seq(label, file_path)

def extend_dataset(input_pair, vm, vm_args, file_path="temp", num=10_000, is_train=False):
    if is_train:
        data_path = os.path.join(file_path, "train.txt")
    else:
        data_path = os.path.join(file_path, "val.txt")
    _synthesize_obfuscated_sequence(input_pair[0], vm, vm_args, "__label__1", data_path, num)
    _synthesize_obfuscated_sequence(input_pair[1], vm, vm_args, "__label__2", data_path, num)

def flush_obfuscated_sequence(file_path="temp"):
    os.system(f"rm -f {file_path}/*.txt")
    os.system(f"rm -f {file_path}/*.bin")

def train_and_val(model:str, file_path="temp"):
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

def main(config: FuzzerConfig):
    start = time.time()
    rand_pool = random_pool(config.random_generator, config.random_generator_args, config.random_input_pool_size)
    lib_files = config.candidate_solution_lib_files
    candidate_solution_file = config.candidate_solution_file
    candidate_solution = config.candidate_solution_module
    inst_white_list = config.instrumentation_white_list
    vm_args = config.vm_args

    file_path = config.file_path
    train_num = config.train_num

    is_nonbliviousness = False

    hist_result = dynamic_analysis(candidate_solution_file, rand_pool, lib_files, vm_args, inst_white_list)
    heap = get_prioritized_pair(hist_result, maxsize=config.maximal_trail_num)
    
    while len(heap) != 0:
        val_num = config.init_val_num
        curr_val_num = val_num

        dist, input_pair = heap.pop(0)
        print(input_pair, "dist:", dist)
        extend_dataset(input_pair, candidate_solution, vm_args, num=train_num, is_train=True)
        extend_dataset(input_pair, candidate_solution, vm_args, num=val_num, is_train=False)
        while curr_val_num < 20000:
            precision = train_and_val("CNN", file_path=file_path)
            if precision <= 0.5:
                print(f"indistinguishable w.h.p. on {input_pair} with accuracy of {precision}")
                break
            elif precision > 0.5 + np.sqrt(1.498/curr_val_num):
                print(f"distinguishable w.h.p. on {input_pair} with accuracy of {precision}")
                model_path = os.path.join(file_path, "model.bin")
                print(f"model saved at {model_path}")
                is_nonbliviousness = True
                break
            else:
                # TODO: prograssively increase 
                print(f"extend dataset and train again with accuracy of {precision}")
                extend_dataset(input_pair, candidate_solution, vm_args, num=int(0.3*train_num), is_train=True)
                extend_dataset(input_pair, candidate_solution, vm_args, num=int(1.5*val_num), is_train=False)
                curr_val_num += int(1.5*val_num)
        if is_nonbliviousness:
            break
        else:
            flush_obfuscated_sequence(file_path)
            # break
        
    if not is_nonbliviousness: print("timeout")
    print(time.time()-start)
    

if __name__ == "__main__":
    config = FuzzerConfig(sys.argv[1]) 
    main(config)