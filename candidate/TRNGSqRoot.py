import numpy as np
from util.solution_util import *
from logging import debug, info, error
import logging, time
# import Crypto.Random.random as crypto_random 
import random

logging.basicConfig(level=logging.ERROR)
strong_random = random.SystemRandom()



class Permutation:
    def ConstructRandomOracle(size: int):
        # construct random 
        rmax = int(size ** np.log2(size))
        rand_num = np.arange(0,rmax,dtype=int)
        # strong_random.shuffle(rand_num)
        # strong_random.shuffle(rand_num)
        # strong_random.shuffle(rand_num)
        # np.random.seed(strong_random.randint(0, 10_0000_0000))
        oracle = np.random.choice(rand_num, size=size, replace=False)
        return oracle
    
    # as memory access pattern of tag assignment is independent w.r.t. inputs, 
    # we use perfect r/w operation
    def AssignTag(tags, memory):
        debug(f"Assign Tags to {len(tags)} blocks.")
        for idx, tag in enumerate(tags):
            mem_unit = memory.read(idx)
            assert mem_unit[0] == MemoryUnitType.NonEmpty # ensure the assigned memory unit is non-empty
            _, mem_data = mem_unit
            mem_data["tag"] = tag
            memory.write_mem_data(idx, mem_data)
    
    def SortByTag(memory):
        # in original Square Root Algorithm, Batcher’s Sorting Network is used
        # to obliviously sort items. here, we use ideal sort to sort elements 
        # where detailed memory access pattern is not recorded.
        debug(f"Perform Ideal Sort on Memory Units by Tag (Ascending)")
        memory.perfect_inplace_sort_by_tag()

class SqRootProtocol:
    def __init__(self, size: int):
        # ensure size is a square number
        assert int(np.sqrt(size))**2 == size
        self.virtual_memory_size = size
        self.square_root = int(np.sqrt(size))
        self.epoch = 0
        self.initialize_memory(size)


    def initialize_memory(self, size: int):
        info("Initialize Memory at the First Epoch")
        self.real_memory_size = size + 2 * self.square_root
        self.permuted_memory_size = size + self.square_root
        self.block_list = list(range(size)) + [-1 for i in range(self.square_root)]
        self.real_memory = RealMemory(self.real_memory_size, self.block_list)
        self.tags = Permutation.ConstructRandomOracle(self.permuted_memory_size)
        self.sorted_tag = sorted(self.tags)
        Permutation.AssignTag(self.tags, self.real_memory)
        Permutation.SortByTag(self.real_memory)
    
    # prioritize non-dummy blocks and 
    def sort(self):
        # blocks updated in current epoch
        updated_blocks = {}
        shelter_start_idx = self.permuted_memory_size
        shelter_end_idx = self.real_memory_size - 1
        curr_pointer = shelter_start_idx
        while curr_pointer <= shelter_end_idx:
            mem_unit = self.real_memory.perfect_read(curr_pointer)
            if mem_unit[0] == MemoryUnitType.NonEmpty:
                _, mem_data = mem_unit
                bID = mem_data["blockID"]
                if -1 != bID:
                    updated_blocks[bID] = curr_pointer
            curr_pointer += 1
        # mark updated blocks in permuted memory as dummy
        permuted_start_idx = 0
        permuted_end_idx = self.permuted_memory_size - 1
        curr_pointer = permuted_start_idx
        while curr_pointer <= permuted_end_idx:
            mem_unit = self.real_memory.read(curr_pointer)
            if mem_unit[0] == MemoryUnitType.NonEmpty:
                _, mem_data = mem_unit
                bID = mem_data["blockID"]
                if bID in updated_blocks:
                    self.real_memory.write(curr_pointer, -1)
                else:
                    self.real_memory.write_nothing(curr_pointer)
            else:
                self.real_memory.write_nothing(curr_pointer)
            curr_pointer += 1
        # sort real blocks
        debug("Sort Blocks by Block ID")
        def sorter(mem_unit):
            if mem_unit[0] == MemoryUnitType.Empty:
                return self.real_memory_size + 2
            elif mem_unit[1]["blockID"] == -1:
                return self.real_memory_size + 1
            else:
                return mem_unit[1]["blockID"]
        self.real_memory.perfect_sort(sorter)        

    def clear_shelter(self):
        debug("Clear Shelter")
        shelter_start_idx = self.permuted_memory_size
        shelter_end_idx = self.real_memory_size - 1
        curr_pointer = shelter_start_idx
        while curr_pointer <= shelter_end_idx:
            self.real_memory.clear_memory(curr_pointer)
            curr_pointer += 1

    # reshuffle memory
    def reshuffle(self):
        info("Reshuffle Memory")
        info("Sort")
        self.sort()
        info("Clear Shelter")
        self.clear_shelter()
        self.tags = Permutation.ConstructRandomOracle(self.permuted_memory_size)
        self.sorted_tag = sorted(self.tags)
        info("Assign Tag")
        Permutation.AssignTag(self.tags, self.real_memory)
        info("Sort By Tag")
        Permutation.SortByTag(self.real_memory)

    # step 1
    def scan_shelter(self, bID):
        shelter_start_idx = self.permuted_memory_size
        shelter_end_idx = self.real_memory_size - 1
        curr_pointer = shelter_start_idx
        is_found = False
        while curr_pointer <= shelter_end_idx:
            mem_unit = self.real_memory.read(curr_pointer)
            if mem_unit[0] == MemoryUnitType.NonEmpty:
                _, mem_data = mem_unit
                if bID == mem_data["blockID"]:
                    is_found = True
            curr_pointer += 1
        return is_found
    
    # step 3
    def cache_retrieved_data_to_shelter(self, bID):
        debug("Write Retrieved Data to Shelter")
        shelter_start_idx = self.permuted_memory_size
        shelter_end_idx = self.real_memory_size - 1
        curr_pointer = shelter_start_idx
        successful_write = False
        while curr_pointer <= shelter_end_idx:
            mem_unit = self.real_memory.read(curr_pointer)
            if mem_unit[0] == MemoryUnitType.Empty and not successful_write:
                self.real_memory.write(curr_pointer, bID)
                successful_write = True
            else:
                self.real_memory.write_nothing(curr_pointer)
            curr_pointer += 1
        assert successful_write

    # step 2
    def retrieve_from_permuted_memory(self, bID):
        assert bID < self.permuted_memory_size
        tag = self.tags[bID]
        addr, mem_data = self.binary_search(tag)
        return addr, mem_data
        
    # binary search for retrieve memory from permuted memory
    def binary_search(self, tag):
        def inner_search(memory, start, end, tag):
            if start <= end:
                mid = (start + end) // 2
                # read memory
                mem_unit = memory.read(mid)
                assert mem_unit[0] != MemoryUnitType.Empty
                assert "tag" in mem_unit[1]
                mem_data = mem_unit[1]
                curr_tag = mem_data["tag"]
                if curr_tag == tag:
                    return mid, mem_data
                elif curr_tag > tag:
                    return inner_search(memory, start, mid - 1, tag)
                else:
                    return inner_search(memory, mid + 1, end, tag)
            else:
                error(f"Tag#{tag} not found via Binary Search.")
                assert False
        return inner_search(self.real_memory, 0, self.permuted_memory_size-1, tag)

class VirtualMemory:
    def __init__(self, size):
        self.protocol = SqRootProtocol(size)
        self.size = size
        self.square_root = int(np.sqrt(size))
        self.counter = 0

    def read(self, bID):
        assert 0 <= bID <= self.size
        is_found = self.protocol.scan_shelter(bID)
        if is_found:
            self.protocol.retrieve_from_permuted_memory(self.size+self.counter)
            self.protocol.cache_retrieved_data_to_shelter(-1)
        else:
            try:
                mem_type, mem_data = self.protocol.retrieve_from_permuted_memory(bID)
                assert bID == mem_data["blockID"]
            except:
                self.protocol.real_memory.dump_memory()
                print(mem_type, mem_data,bID, self.protocol.tags)
                assert False
            self.protocol.cache_retrieved_data_to_shelter(bID)
        self.counter += 1
        if self.counter % self.square_root == 0:
            self.protocol.reshuffle()
            self.counter = 0
    
    def write(self, bID):
        assert 0 <= bID <= self.size
        is_found = self.protocol.scan_shelter(bID)
        if is_found:
            self.protocol.retrieve_from_permuted_memory(self.size+self.counter)
            self.protocol.cache_retrieved_data_to_shelter(-1)
        else:
            try:
                mem_type, mem_data = self.protocol.retrieve_from_permuted_memory(bID)
                assert bID == mem_data["blockID"]
            except:
                self.protocol.real_memory.dump_memory()
                print(mem_type, mem_data, bID)
                assert False
            self.protocol.cache_retrieved_data_to_shelter(bID)
        self.counter += 1
        if self.counter % self.square_root == 0:
            self.protocol.reshuffle()
            self.counter = 0
    
    def dump_seq(self, label, file_name):
        self.protocol.real_memory.dump_memory_access_sequence_with_label(label, file_name)

if __name__ == "__main__":
    vm = VirtualMemory(4)
    while True:
        command = input("Command?")
        if command == "D":
            vm.protocol.real_memory.dump_memory()
            for a in vm.protocol.real_memory.memory_access_seq:
                print(a)
        if command == "R":
            bID = input("Block ID?")
            vm.read(int(bID))
        if command == "W":
            bID = input("Block ID?")
            vm.write(int(bID))
        