import numpy as np
from util.solution_util import *
from logging import debug, info, error
import logging, time

logging.basicConfig(level=logging.ERROR)

class CuckooSimulator:
    def __init__(self, table_size, keys, hash_func1, hash_func2):
        self.hash_table = [-1 for i in range(table_size)]
        self.key_num = len(keys)
        self.keys = keys
        self.hash_func1 = hash_func1
        self.hash_func2 = hash_func2
        self.has_cycle = self.insert()
    
    def move(self, key, curr_loc, starting_key, is_first_time):
        if starting_key == key and not is_first_time:
            return False 
        location1 = self.hash_func1[key]
        location2 = self.hash_func2[key]
        if curr_loc == location1:
            location = location2
        else:
            location = location1
        if self.hash_table[location] == -1:
            self.hash_table[location] = key 
            return True
        else:
            move_key = self.hash_table[location]
            self.hash_table[location] = key 
            return self.move(move_key, location, starting_key, False)

    def insert(self):
        for key in self.keys:
            location = self.hash_func1[key]
            if self.hash_table[location] == -1:
                self.hash_table[location] = key 
            else:
                move_key = self.hash_table[location]
                self.hash_table[location] = key 
                rlt = self.move(move_key, location, key, True)
                if not rlt:
                    return False 
        return True
        

def construct_hash_func(virtual_memory__size, level):
    dummy_size = 2**(level+1) 
    dont_care_size = 2**(level+1)
    output_size = dummy_size * 4
    rand_num = np.arange(0,output_size,dtype=int)
    hash_func = np.random.choice(rand_num, size=virtual_memory__size+dummy_size+dont_care_size)
    return hash_func

class CuckooProtocol:
    def __init__(self, potential_size):
        self.virtual_memory_size = potential_size 
        self.level_num = int(np.ceil(np.log2(potential_size)) + 1)
        self.real_memory_size = sum([4*2**(i+1) for i in range(self.level_num)])
        self.real_memory = RealMemory(self.real_memory_size)
        self.hash_funcs1 = {i:construct_hash_func(potential_size, i) for i in range(self.level_num)}
        self.hash_funcs2 = {i:construct_hash_func(potential_size, i) for i in range(self.level_num)}
    
    def get_level(self, level):
        level_start = sum([4*2**(i+1) for i in range(level)])
        level_end = level_start + 4*2**(level+1)
        return level_start, level_end
    
    def scan_level1(self, bID):
        l1_start, l1_end = self.get_level(0)
        is_found = False
        for addr in range(l1_start, l1_end):
            mem_unit = self.real_memory.read(addr)
            if mem_unit[0] is MemoryUnitType.NonEmpty and bID == mem_unit[1]["blockID"]:
                is_found = True
        return is_found
    
    def retrieve_higher_level(self, level, bID):
        level_start, _ = self.get_level(0)
        is_found = False
        addr1 = level_start + self.hash_funcs1[level][bID]
        addr2 = level_start + self.hash_funcs2[level][bID]
        mem_unit = self.real_memory.read(addr1)
        if mem_unit[0] is MemoryUnitType.NonEmpty and bID == mem_unit[1]["blockID"]:
            is_found = True
        mem_unit = self.real_memory.read(addr2)
        if mem_unit[0] is MemoryUnitType.NonEmpty and bID == mem_unit[1]["blockID"]:
            is_found = True
        return is_found
    
    def write_back(self, bID):
        l1_start, l1_end = self.get_level(0)
        is_write = False
        for addr in range(l1_start, l1_end):
            mem_unit = self.real_memory.read(addr)
            if mem_unit[0] is MemoryUnitType.NonEmpty and bID == mem_unit[1]["blockID"]:
                self.real_memory.write(addr, bID)
                is_write = True
            elif mem_unit[0] is MemoryUnitType.Empty and not is_write:
                self.real_memory.write(addr, bID)
                is_write = True
            else:
                self.real_memory.write_nothing(addr)
        return is_write
    
    def move_down(self, level):
        blocks = []
        duplicates = []
        curr_start, curr_end = self.get_level(level)
        for addr in range(curr_start, curr_end):
            mem_unit = self.real_memory.read(addr)
            if mem_unit[0] is MemoryUnitType.NonEmpty and mem_unit[1]["blockID"] < self.virtual_memory_size:
                blocks.append(mem_unit[1]["blockID"])
            self.real_memory.clear_memory(addr)
        
        next_start, next_end = self.get_level(level+1)

        if level + 1 == self.level_num:
            self.real_memory.allocate_more(4*2**(level+2))
            self.level_num += 1
            self.real_memory_size += 4*2**(level+2)
        else:
            for addr in range(next_start, next_end):
                mem_unit = self.real_memory.read(addr)
                if mem_unit[0] is MemoryUnitType.NonEmpty and mem_unit[1]["blockID"] < self.virtual_memory_size:
                    if mem_unit[1]["blockID"] in blocks:
                        pass
                    else: 
                        blocks.append(mem_unit[1]["blockID"])
                self.real_memory.clear_memory(addr)
        self.hash_funcs1[level] = construct_hash_func(self.virtual_memory_size, level)
        self.hash_funcs2[level] = construct_hash_func(self.virtual_memory_size, level)

        # pad dummy block
        to_be_insert = list(blocks) + [self.virtual_memory_size + dummy_offset for dummy_offset in range(2**(level+2))]
        # pad don't-care block
        to_be_insert += [ self.virtual_memory_size + i + 2**(level+2) for i in range(2*2**(level+2) - len(to_be_insert))]

        has_cycle = True
        while has_cycle:
            self.hash_funcs1[level+1] = construct_hash_func(self.virtual_memory_size, level+1)
            self.hash_funcs2[level+1] = construct_hash_func(self.virtual_memory_size, level+1)
            cuckoo_simulator = CuckooSimulator(4*2**(level+2), to_be_insert, self.hash_funcs1[level+1], self.hash_funcs2[level+1])
            has_cycle = cuckoo_simulator.has_cycle
            # if has_cycle:
            #     print(4*2**(level+2), len(to_be_insert))
            #     print("has cycle")
        
        
        for addr, block in enumerate(cuckoo_simulator.hash_table):
            if block != -1:
                self.real_memory.write(addr + next_start, block)


class VirtualMemory:
    def __init__(self, size):
        self.protocol = CuckooProtocol(size)
        self.block_num = size
        self.req_num = 0
    
    def read(self, bID:int):
        is_found = self.protocol.scan_level1(bID)
        for level in range(1, self.protocol.level_num):
            if is_found:
                dummy_index = self.block_num + (self.req_num % (2**(level+1)))
                self.protocol.retrieve_higher_level(level, dummy_index)
            else:
                is_found = self.protocol.retrieve_higher_level(level, bID)
        is_write = self.protocol.write_back(bID)
        assert is_write
        self.req_num += 1
        self.reshuffle()
    
    def write(self, bID:int):
        is_found = self.protocol.scan_level1(bID)
        for level in range(1, self.protocol.level_num):
            if is_found:
                dummy_index = self.block_num + (self.req_num % (2**(level+1)))
                self.protocol.retrieve_higher_level(level, dummy_index)
            else:
                is_found = self.protocol.retrieve_higher_level(level, bID)
        is_write = self.protocol.write_back(bID)
        assert is_write
        self.req_num += 1
        self.reshuffle()
    
    def dump_seq(self, file_name):
        self.protocol.real_memory.dump_memory_access_sequence(file_name)
    
    def reshuffle(self):
        for level in range(self.protocol.level_num):
            if self.req_num % (2 ** (level + 1)) == 0:
                self.protocol.move_down(level)
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