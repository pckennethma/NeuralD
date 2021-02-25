import heapq, os, json
import numpy as np

class TopkHeap:
    def __init__(self, k):
        self.k = k
        self.data = []

    def Push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0][0]
            if elem[0] > topk_small:
                heapq.heapreplace(self.data, elem)

    def TopK(self):
        return [x for x in reversed([heapq.heappop(self.data) for x in range(len(self.data))])]

class FuzzerConfig:
    def __init__(self, config_file) -> None:
        self.config_file = config_file
        with open(config_file) as f:
            config = json.load(f)
        self.candidate_solution_file = config["candidate_solution_file"]
        self.candidate_solution_module = config["candidate_solution_module"]
        self.candidate_solution_lib_files = config["candidate_solution_lib_files"]
        self.instrumentation_white_list = config["instrumentation_white_list"]
        self.vm_args = config["vm_args"]
        self.random_generator =  config["random_generator"]
        self.random_generator_args =  config["random_generator_args"]
        self.random_input_pool_size = config["random_input_pool_size"]
        
        self.maximal_trail_num = config["maximal_trail_num"]
        self.file_path = config["file_path"]

        self.train_num = config["train_num"]
        self.init_val_num = config["init_val_num"]

class BlackBoxFuzzerConfig:
    def __init__(self, config_file) -> None:
        self.config_file = config_file
        with open(config_file) as f:
            config = json.load(f)
        self.candidate_solution_file = config["candidate_solution_file"]
        self.random_generator =  config["random_generator"]
        self.random_generator_args =  config["random_generator_args"]
        self.random_input_pool_size = config["random_input_pool_size"]
        
        self.maximal_trail_num = config["maximal_trail_num"]
        self.file_path = config["file_path"]

        self.train_num = config["train_num"]
        self.init_val_num = config["init_val_num"]

def decode_input_str(input_str):
    # input_str format: R0;R5;R0;
    # input format: [('R', 0), ('R', 5), ('R', 0)]
    input_str_split = [i for i in input_str.split(";") if i != ""]
    input_list = []
    for i in input_str_split:
        operation = i[0]
        addr = int(i[1:])
        input_list.append((operation, addr))
    return input_list

def encode_input(input_list):
    execution_input_str = ""
    for access_entry in input_list:
        execution_input_str += access_entry[0] + str(access_entry[1]) + ";"
    return execution_input_str

def encode_input_blackbox(input_list):
    execution_input_str = ""
    for access_entry in input_list:
        execution_input_str += access_entry[0] + " " + str(access_entry[1]) + " "
    return execution_input_str.strip()

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def prepare_instrumentation_env(lib_files):
    # prepare environment
    os.system("rm -rf Instrumentation/temp/*")
    os.system("mkdir Instrumentation/temp/util")
    for file_path in lib_files: os.system(f"cp util/{file_path} Instrumentation/temp/util/")
    os.system("touch Instrumentation/temp/__init__.py")
    os.system("touch Instrumentation/temp/util/__init__.py")
    os.system("cp Instrumentation/util.py Instrumentation/temp/")

def random_input(size, length):
    operations = np.random.choice(["R", "W"], size=length)
    address = np.random.choice(list(range(size)), size=length)
    return [(operations[i],address[i]) for i in range(length)]

def random_pool(random_generator, random_generator_args, random_input_pool_size):
    if random_generator == "default":
        return [random_input(**random_generator_args) for i in range(random_input_pool_size)]
    else:
        raise NotImplementedError()

def merge_file(file1, file2, file3):
    lines = []
    with open(file1) as f:
        for l in f.readlines():
            lines.append("__label__1 " + l)
    with open(file2) as f:
        for l in f.readlines():
            lines.append("__label__2 " + l)
    with open(file3, "w") as f:
        f.writelines(lines)