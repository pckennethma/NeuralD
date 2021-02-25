import numpy as np
from util.solution_util import *
from logging import debug, info, error
import logging, time

logging.basicConfig(level=logging.ERROR)

def construct_hash_func(input_size, output_size):
    rand_num = np.arange(0,output_size,dtype=int)
    hash_func = np.random.choice(rand_num, size=input_size)
    return hash_func

class HierarchicalProtocol:
    def __init__(self, size):
        self.virtual_memory_size = size
        self.bucket_size = int(np.ceil(np.log2(size)) * 1)
        self.level_num = int(np.ceil(np.log2(size)) + 1)
        self.real_memory_size = np.sum([self.bucket_size * 2 ** (i+1) for i in range(self.level_num)])
        self.real_memory = RealMemory(self.real_memory_size)
        self.hash_funcs = {level: construct_hash_func(self.virtual_memory_size ,2 ** (level+1)) for level in range(self.level_num)}
        self.allocate_initial()

    def get_level_range(self, level_index):
        assert self.level_num > level_index
        start = int(np.sum([self.bucket_size * 2 ** (i+1) for i in range(level_index)]))
        end = int(start + self.bucket_size * 2 ** (level_index+1))
        return start, end
    
    def get_bucket_range(self, level_index, bucket_index):
        assert 2 ** (level_index+1) > bucket_index
        level_start, _ = self.get_level_range(level_index)
        return level_start, level_start + (bucket_index + 1) * self.bucket_size
    
    def allocate_initial(self):
        block_list = range(self.virtual_memory_size)
        allocate_level = self.level_num-1
        
        hash_func = self.hash_funcs[allocate_level]
        no_collsion = np.max(np.bincount(hash_func)) <= self.bucket_size
        while not no_collsion:
            self.hash_funcs[allocate_level] = construct_hash_func(self.virtual_memory_size ,2 ** (allocate_level+1))
            hash_func = self.hash_funcs[allocate_level]
            no_collsion = np.max(np.bincount(hash_func)) <= self.bucket_size

        for block in block_list:
            bucket_id = hash_func[block]
            bucket_start, bucket_end = self.get_bucket_range(allocate_level, bucket_id)
            for addr in range(bucket_start, bucket_end):
                mem_unit = self.real_memory.perfect_read(addr)
                if mem_unit[0] is MemoryUnitType.Empty:
                    self.real_memory.perfect_write(addr, block)
                    break
    
    def scan_level1(self, bID):
        l1_start, l1_end = self.get_level_range(0)
        is_found = False 
        for addr in range(l1_start, l1_end):
            mem_unit = self.real_memory.read(addr)
            if mem_unit[0] is MemoryUnitType.NonEmpty and bID == mem_unit[1]["blockID"]:
                is_found = True
        return is_found
    
    def retrive_higher_level(self, level, bID):
        hash_func = self.hash_funcs[level]
        bucketID = hash_func[bID]
        bucket_start, bucket_end = self.get_bucket_range(level, bucketID)
        is_found = False 
        for addr in range(bucket_start, bucket_end):
            mem_unit = self.real_memory.read(addr)
            if mem_unit[0] is MemoryUnitType.NonEmpty and bID == mem_unit[1]["blockID"]:
                is_found = True
        return is_found
    
    def write_back(self, bID):
        hash_func = self.hash_funcs[0]
        bucketID = hash_func[bID]
        bucket_start, bucket_end = self.get_bucket_range(0, bucketID)
        is_write = False 
        for addr in range(bucket_start, bucket_end):
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
    
    def handle_collsion(self, level, block_to_add):
        self.real_memory.insert_access_entry(MemoryAccess.Rehash, "l"+str(level))
        start_addr, end_addr = self.get_level_range(level)
        stored_block = [block_to_add]
        for addr in range(start_addr, end_addr):
            mem_unit = self.real_memory.perfect_read(addr)
            if mem_unit[0] is MemoryUnitType.NonEmpty:
                stored_block.append(mem_unit[1]["blockID"])
        is_resolved = False

        new_hash_func = None
        while not is_resolved:
            new_hash_func = construct_hash_func(self.virtual_memory_size, 2 ** (level + 1))
            assigned_bucket = [new_hash_func[bID] for bID in stored_block]
            bincount = np.bincount(assigned_bucket)
            is_resolved = np.max(bincount) <= self.bucket_size
        assert new_hash_func is not None

        self.hash_funcs[level] = new_hash_func
        self.clear_level(level)
        for bID in stored_block:
            bucketID = new_hash_func[bID]
            bucket_start, bucket_end = self.get_bucket_range(0, bucketID)
            is_write = False 
            for addr in range(bucket_start, bucket_end):
                mem_unit = self.real_memory.perfect_read(addr)
                if mem_unit[0] is MemoryUnitType.Empty:
                    self.real_memory.perfect_write(addr, bID)
                    is_write = True
                    break
                if mem_unit[0] is MemoryUnitType.NonEmpty and mem_unit[1]["blockID"] == bID:
                    self.real_memory.perfect_write(addr, bID)
                    is_write = True
                    break
            assert is_write
    
    # move level to level+1
    def move_down(self, level):
        assert level < self.level_num
        next_level = level + 1
        new_level_size = self.bucket_size * 2 ** next_level
        new_level_bucket_size = 2 ** (next_level + 1)
        
        self.real_memory.insert_access_entry(MemoryAccess.Move_Down, "l"+str(next_level))
        if next_level == self.level_num:
            self.level_num += 1
            self.real_memory_size += new_level_size
            self.real_memory.allocate_more(new_level_size)
            
        curr_start, curr_end = self.get_level_range(level)
        next_start, next_end = self.get_level_range(next_level)
        
        self.real_memory.allocate_temp(next_end - curr_start)
        for addr in range(curr_start, curr_end):
            mem_unit = self.real_memory.perfect_read(addr)
            if mem_unit[0] is MemoryUnitType.NonEmpty:
                self.real_memory.write_temp(addr-curr_start, mem_unit[1]["blockID"], level)
            else:
                self.real_memory.write_nothing_temp(addr-curr_start)
        for addr in range(next_start, next_end):
            mem_unit = self.real_memory.perfect_read(addr)
            if mem_unit[0] is MemoryUnitType.NonEmpty:
                self.real_memory.write_temp(addr-curr_start, mem_unit[1]["blockID"], next_level)
            else:
                self.real_memory.write_nothing_temp(addr-curr_start)
        
        def step2_osort(mem_unit):
            if mem_unit[0] == MemoryUnitType.Empty:
                return self.virtual_memory_size 
            elif mem_unit[1]["tag"] == level:
                return mem_unit[1]["blockID"]
            elif mem_unit[1]["tag"] == next_level:
                return mem_unit[1]["blockID"] + 0.1
            else:
                raise NotImplementedError()
        
        self.real_memory.sort_temp(step2_osort)
        
        # remove old blocks
        prev_bID = -1
        nonempty_num = 0
        blocks = []
        for addr in range(0, next_end-curr_start):
            mem_unit = self.real_memory.read_temp(addr)
            if mem_unit[0] == MemoryUnitType.NonEmpty:
                if mem_unit[1]["blockID"] == prev_bID:
                    self.real_memory.clear_temp(addr)
                else:
                    self.real_memory.write_nothing_temp(addr)
                    prev_bID = mem_unit[1]["blockID"]
                    blocks.append(prev_bID)
                    nonempty_num += 1
            else:
                self.real_memory.write_nothing_temp(addr)
        
        # add dummy
        no_collsion_on_non_dummy = False 
        next_hash_func = construct_hash_func(self.virtual_memory_size, new_level_bucket_size)
        while not no_collsion_on_non_dummy:
            bucket = [next_hash_func[bID] for bID in blocks]
            no_collsion_on_non_dummy = np.max(np.bincount(bucket)) <= self.bucket_size
            if not no_collsion_on_non_dummy:
                next_hash_func = construct_hash_func(self.virtual_memory_size, new_level_bucket_size)


        dummy_num = 2 ** (next_level + 2) - nonempty_num
        assert dummy_num + len(blocks) <= new_level_bucket_size * self.bucket_size
        bucket_bin = np.bincount([next_hash_func[bID] for bID in blocks] + [new_level_bucket_size-1])
        bucket_bin[new_level_bucket_size-1] -= 1

        def random_bucket(bucket_bin):
            bucket_id = np.random.randint(new_level_bucket_size)
            count = 0
            while bucket_bin[bucket_id] >= self.bucket_size:
                if count > 1000:
                    raise RuntimeError()
                bucket_id = np.random.randint(new_level_bucket_size)
                count += 1
            return bucket_id

        for addr in range(0, next_end-curr_start):
            mem_unit = self.real_memory.read_temp(addr)
            if dummy_num != 0 and mem_unit[0] == MemoryUnitType.Empty:
                rand_bucket = random_bucket(bucket_bin)
                bucket_bin[rand_bucket] += 1
                self.real_memory.write_temp(addr, -1, rand_bucket)
                dummy_num -= 1
            else:
                self.real_memory.write_nothing_temp(addr)
        
        # map to bucket
        bucket = []
        for addr in range(0, next_end-curr_start):
            mem_unit = self.real_memory.read_temp(addr)
            if mem_unit[0] == MemoryUnitType.NonEmpty:
                if mem_unit[1]["blockID"] == -1:
                    bucketID = mem_unit[1]["tag"]
                    bucket.append(bucketID)
                else:
                    bucketID = next_hash_func[mem_unit[1]["blockID"]]
                    bucket.append(bucketID)
                self.real_memory.write_temp(addr, mem_unit[1]["blockID"], bucketID)
        assert np.max(np.bincount(bucket)) <= self.bucket_size

        self.hash_funcs[next_level] = next_hash_func
        def step6_osort(mem_unit):
            if mem_unit[0] == MemoryUnitType.Empty:
                return new_level_bucket_size
            else:
                return mem_unit[1]["tag"] 
        self.real_memory.sort_temp(step6_osort)
        bucket_bin = self.bucket_size - np.bincount(bucket)
        def get_first_non_zero(b_bin):
            for i in range(len(b_bin)):
                if b_bin[i] != 0:
                    return i
            return -1
        for addr in range(0, next_end-curr_start):
            bucketID = get_first_non_zero(bucket_bin)
            mem_unit = self.real_memory.read_temp(addr)
            if mem_unit[0] == MemoryUnitType.Empty and bucketID != -1:
                bucket_bin[bucketID] -= 1
                self.real_memory.write_temp(addr, -2, bucketID)
            else:
                self.real_memory.write_nothing_temp(addr)
        
        def step9_osort(mem_unit):
            # actually un tagged
            if mem_unit[0] == MemoryUnitType.Empty:
                return new_level_bucket_size
            elif mem_unit[1]["blockID"] == -2:
                return mem_unit[1]["tag"] + 0.1
            else:
                return mem_unit[1]["tag"]
        self.real_memory.sort_temp(step9_osort)
        for addr in range(0, next_end-curr_start):
            mem_unit = self.real_memory.read_temp(addr)
            if mem_unit[0] == MemoryUnitType.NonEmpty and mem_unit[1]["blockID"] == -2:
                self.real_memory.clear_temp(addr)
            else:
                self.real_memory.write_nothing_temp(addr)
        def step11_osort(mem_unit):
            if mem_unit[0] == MemoryUnitType.Empty:
                return new_level_bucket_size
            else:
                return mem_unit[1]["tag"]
        self.real_memory.sort_temp(step11_osort)
        for addr in range(0, next_end-curr_start):
            mem_unit = self.real_memory.read_temp(addr)
            if mem_unit[0] == MemoryUnitType.NonEmpty and mem_unit[1]["blockID"] == -1:
                self.real_memory.clear_temp(addr)
            else:
                self.real_memory.write_nothing_temp(addr)
        self.real_memory.copy_temp_to_inner_memory(next_start, next_end)
        self.clear_level(level)
        

    # clear all memory units in the level
    def clear_level(self, level):
        start, end = self.get_level_range(level)
        for addr in range(start, end):
            self.real_memory.perfect_claer(addr)

class VirtualMemory:
    def __init__(self, size) -> None:
        self.protocol = HierarchicalProtocol(size)
        self.block_num = size
        self.req_num = 0
        np.random.seed(int(time.time()*1000000) % 1000000)
    
    def read(self, bID:int):
        is_found = self.protocol.scan_level1(bID)
        for level in range(1, self.protocol.level_num):
            if is_found:
                dummy_bid = np.random.randint(self.block_num)
                is_found = self.protocol.retrive_higher_level(level, dummy_bid)
            else:
                is_found = self.protocol.retrive_higher_level(level, bID)
        is_write = self.protocol.write_back(bID)
        if not is_write: 
            self.protocol.handle_collsion(0, bID)
        self.req_num += 1
        self.reshuffle()
    
    def write(self, bID:int):
        is_found = self.protocol.scan_level1(bID)
        for level in range(1, self.protocol.level_num):
            if is_found:
                dummy_bid = np.random.randint(self.block_num)
                is_found = self.protocol.retrive_higher_level(level, dummy_bid)
            else:
                is_found = self.protocol.retrive_higher_level(level, bID)
        is_write = self.protocol.write_back(bID)
        if not is_write: 
            self.protocol.handle_collsion(0, bID)
        self.req_num += 1
        self.reshuffle()
    
    def dump_seq(self, label, file_name):
        self.protocol.real_memory.dump_memory_access_sequence_with_label(label, file_name)
    
    def reshuffle(self):
        for level in range(self.protocol.level_num, -1, -1):
            if self.req_num % (2 ** (level + 1)) == 0:
                self.protocol.move_down(level)
            else:
                pass


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