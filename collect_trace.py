import tqdm, importlib, sys
from util.fuzzer_util import TopkHeap, FuzzerConfig, decode_input_str, encode_input, prepare_instrumentation_env, random_pool


def _synthesize_trace(input, vm, vm_args, label, file_path):
    VirtualMemory = importlib.import_module(vm).VirtualMemory
    temp_vm = VirtualMemory(**vm_args)
    for access_entry in input:
        if access_entry[0] == "R": temp_vm.read(access_entry[1])
        else: temp_vm.write(access_entry[1])
    temp_vm.dump_seq(label, file_path)


def main(config: FuzzerConfig, input_pair):
    lib_files = config.candidate_solution_lib_files
    candidate_solution_file = config.candidate_solution_file
    candidate_solution = config.candidate_solution_module
    inst_white_list = config.instrumentation_white_list
    vm_args = config.vm_args

    file_path = config.file_path

    _synthesize_trace(input_pair[0], candidate_solution, vm_args, "1", "temp/debug1.txt")
    _synthesize_trace(input_pair[1], candidate_solution, vm_args, "2", "temp/debug2.txt")

if __name__ == "__main__":
    config = FuzzerConfig(sys.argv[1]) 
    # Pract Sq Root
    input_pair = ([('W', 3), ('W', 3), ('W', 1), ('W', 1), ('W', 3), ('W', 3)], [('R', 0), ('R', 1), ('R', 0), ('R', 2), ('R', 0), ('R', 1)])
    # Buggy Sq Root
    input_pair = ([('R', 2), ('R', 3), ('R', 1), ('R', 3)], [('W', 0), ('W', 0), ('W', 2), ('W', 2)])
    main(config, input_pair)
