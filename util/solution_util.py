import random
from enum import Enum
from typing import List
from copy import copy
import numpy as np
import traceback

class MemoryUnitType(Enum):
    Empty = 1
    NonEmpty = 2

class MemoryAccess(Enum):
    Read = 1
    Write = 2
    Sort = 3
    DeleteTemp = 4
    
    Allocate = 6
    Move_Down = 7
    Scan = 8
    Rehash = 9
    CopyTemp = 10

class MemorySeq:
    def __init__(self, debug=False) -> None:
        self.seq = []
        self.debug = debug
        if debug:
            self.call_stacks = []
    
    def append(self, entry):
        self.seq.append(entry)
        if self.debug:
            self.call_stacks.append(traceback.format_stack())

class RealMemory:
    def __init__(self, memory_size: int, block_list=None):
        self.global_writer_counter = 0
        if block_list: self.initialize_memory(memory_size, block_list)
        else:self.initialize_empty_memory(memory_size)
        self.memory_size = memory_size
        self.memory_access_seq = MemorySeq(debug=True)
        #self.memory_access_seq = MemorySeq()
        
    def initialize_memory(self, memory_size: int, block_list: List[int]):
        self._inner_memory = []
        for addr in range(memory_size):
            self._inner_memory.append((MemoryUnitType.Empty, ))
        for addr, blockID in enumerate(block_list):
            memory_data = {
                "lastModified": 0,
                "blockID": blockID,
            }
            self._inner_memory[addr] = (MemoryUnitType.NonEmpty, memory_data)
    
    def initialize_empty_memory(self, memory_size: int):
        self._inner_memory = []
        for addr in range(memory_size):
            self._inner_memory.append((MemoryUnitType.Empty, ))
            
    def read(self, addr: int):
        self.memory_access_seq.append((MemoryAccess.Read, addr))
        return self._inner_memory[addr]

    def write(self, addr: int, blockID):
        self.memory_access_seq.append((MemoryAccess.Write, addr))
        self.global_writer_counter += 1
        mem_data = {
            "lastModified": self.global_writer_counter,
            "blockID": blockID,
        }
        self._inner_memory[addr] = (MemoryUnitType.NonEmpty, mem_data)
    
    def write_mem_data(self, addr: int, mem_data):
        self.memory_access_seq.append((MemoryAccess.Write, addr))
        self.global_writer_counter += 1
        mem_data["lastModified"] = self.global_writer_counter
        self._inner_memory[addr] = (MemoryUnitType.NonEmpty, mem_data)
    
    # a handy method for write nothing
    # note that, to adversary, write(addr, bID) and write_nothing(addr) is indistinguishable
    def write_nothing(self, addr: int):
        self.memory_access_seq.append((MemoryAccess.Write, addr))
    
    def clear_memory(self, addr: int):
        self.memory_access_seq.append((MemoryAccess.Write, addr))
        self._inner_memory[addr] = (MemoryUnitType.Empty, )
    
    def allocate_more(self, alloc_mem_size: int):
        self.memory_access_seq.append((MemoryAccess.Allocate, alloc_mem_size))
        for addr in range(alloc_mem_size):
            self._inner_memory.append((MemoryUnitType.Empty, ))
    
    # only allowed for some operations which are known to be 
    # ideally oblivious from server
    def perfect_read(self, addr: int):
        return self._inner_memory[addr]
    
    def perfect_write(self, addr: int, blockID):
        self.global_writer_counter += 1
        mem_data = {
            "lastModified": self.global_writer_counter,
            "blockID": blockID,
        }
        self._inner_memory[addr] = (MemoryUnitType.NonEmpty, mem_data)
    
    def perfect_write_mem_data(self, addr: int, mem_data):
        self.global_writer_counter += 1
        mem_data["lastModified"] = self.global_writer_counter
        self._inner_memory[addr] = (MemoryUnitType.NonEmpty, mem_data)
    
    def perfect_claer(self, addr:int):
        self._inner_memory[addr] = (MemoryUnitType.Empty, )
    
    # perform ideally oblivious sort
    def perfect_sort(self, sort_key):
        self._inner_memory.sort(key=sort_key)
    
    # perform ideal sort by tag
    # mainly for square root solution
    def perfect_inplace_sort_by_tag(self):
        # since real_size = virt_size^2 + 2*virt_size
        # we have virt_size = sqrt(real_size + 1) - 1
        virt_memory_size = (np.sqrt(self.memory_size+1) - 1) ** 2
        # ensure virt_memory_size is a square number
        assert int(np.sqrt(virt_memory_size)) ** 2 == virt_memory_size
        # slice the permuted_memory from the whole memory
        permuted_memory_size = int(virt_memory_size + np.sqrt(virt_memory_size))
        permuted_memory = copy(self._inner_memory[:permuted_memory_size])
        sheltered_memory = copy(self._inner_memory[permuted_memory_size:])
        assert len([i for i in permuted_memory if i[0] == MemoryUnitType.Empty or "tag" not in i[1]])==0 # ensure all units in permuted_memory is valid
        # perfect sort that memory is modified but no informatin can be inferred
        sorted_permuted_memory = sorted(permuted_memory, key=lambda mem_unit: mem_unit[1]["tag"])
        self._inner_memory = sorted_permuted_memory + sheltered_memory
        self.memory_access_seq.append((MemoryAccess.Sort, ))

    def get_inner_memory(self):
        return copy(self._inner_memory)
    
    def set_inner_memory(self, _inner_memory):
        self._inner_memory = _inner_memory
    
    def insert_access_entry(self, access_type, access_detail):
        if access_detail:
            self.memory_access_seq.append((access_type, access_detail))
        else:
            self.memory_access_seq.append((access_type, ))
    
    def allocate_temp(self, size:int):
        self.memory_access_seq.append((MemoryAccess.Allocate, "temp"))
        self.temp_memory = []
        for i in range(size):
            self.temp_memory.append((MemoryUnitType.Empty, ))
    
    def write_temp(self, addr: int, blockID, tag):
        self.global_writer_counter += 1
        mem_data = {
            "lastModified": self.global_writer_counter,
            "blockID": blockID,
            "tag": tag
        }
        self.temp_memory[addr] = (MemoryUnitType.NonEmpty, mem_data)
    
    def write_nothing_temp(self, addr: int):
        pass

    def read_temp(self, addr: int):
        return self.temp_memory[addr]
    
    def sort_temp(self, sorter):
        self.memory_access_seq.append((MemoryAccess.Sort, "temp"))
        self.temp_memory.sort(key=sorter)
    
    def delete_temp(self):
        self.memory_access_seq.append((MemoryAccess.DeleteTemp, "t"+str(addr)))
        del self.temp_memory

    def clear_temp(self, addr:int):
        self.temp_memory[addr] = (MemoryUnitType.Empty, )
    
    def copy_temp_to_inner_memory(self, start, end):
        self.memory_access_seq.append((MemoryAccess.CopyTemp, ))
        self._inner_memory = self._inner_memory[:start] + self.temp_memory[:end-start] + self._inner_memory[end:]

    def dump_memory(self, memory=None):
        print("Address｜Block ID|Last Modified|Tag")
        print("-------------------------------")
        for addr, mem_unit in enumerate(self._inner_memory if memory is None else memory):
            if mem_unit[0] == MemoryUnitType.Empty:
                print("%-7s|%-8s|%-14s|%s" % (addr, "Empty", "N/A", "N/A"))
            else:
                print("%-7s|%-8s|%-14s|%s" % (addr, mem_unit[1]["blockID"], mem_unit[1]["lastModified"], "N/A" if "tag" not in mem_unit[1] else mem_unit[1]["tag"]))

    def dump_memory_access_sequence(self, file_name):
        '''
            Read = 1
            Write = 2
            Sort = 3
            DeleteTemp = 4
            Bucket = 5
            Allocate = 6
            Move_Down = 7
            Scan = 8
            Rehash = 9
            CopyTemp = 10
        '''
        seq = ""
        for access_entry in self.memory_access_seq:
            if len(access_entry) == 1: access_entry = (access_entry[0], "")
            if access_entry[0] == MemoryAccess.Read:
                seq += "read " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.Write:
                seq += "write " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.DeleteTemp:
                seq += "delete_temp " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.Move_Down:
                seq += "move_down " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.Allocate:
                seq += "allocate " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.Scan:
                seq += "scan " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.Rehash:
                seq += "rehash " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.CopyTemp:
                seq += "copy_temp " + str(access_entry[1]) + " "
            else:
                seq += "sort "
        with open(file_name, "a") as f:
            f.write(seq.strip()+"\n")
    
    def dump_memory_access_sequence_with_label(self, label, file_name):
        seq = label + " "
        for access_entry in self.memory_access_seq.seq:
            if len(access_entry) == 1: access_entry = (access_entry[0], "")
            if access_entry[0] == MemoryAccess.Read:
                seq += "read " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.Write:
                seq += "write " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.DeleteTemp:
                seq += "delete_temp " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.Move_Down:
                seq += "move_down " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.Allocate:
                seq += "allocate " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.Scan:
                seq += "scan " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.Rehash:
                seq += "rehash " + str(access_entry[1]) + " "
            elif access_entry[0] == MemoryAccess.CopyTemp:
                seq += "copy_temp " + str(access_entry[1]) + " "
            else:
                seq += "sort "
        with open(file_name, "a") as f:
            f.write(seq.strip()+"\n")
        if self.memory_access_seq.debug:
            traces = ""
            counter = 0
            for idx, trace in enumerate(self.memory_access_seq.call_stacks):
                access_entry = self.memory_access_seq.seq[idx]
                next_counter = counter + len(access_entry)
                if len(access_entry) == 1: access_entry = (access_entry[0], "")
                if access_entry[0] == MemoryAccess.Read:
                    action = "read " + str(access_entry[1]) + " "
                elif access_entry[0] == MemoryAccess.Write:
                    action =  "write " + str(access_entry[1]) + " "
                elif access_entry[0] == MemoryAccess.DeleteTemp:
                    action =  "delete_temp " + str(access_entry[1]) + " "
                elif access_entry[0] == MemoryAccess.Move_Down:
                    action =  "move_down " + str(access_entry[1]) + " "
                elif access_entry[0] == MemoryAccess.Allocate:
                    action =  "allocate " + str(access_entry[1]) + " "
                elif access_entry[0] == MemoryAccess.Scan:
                    action =  "scan " + str(access_entry[1]) + " "
                elif access_entry[0] == MemoryAccess.Rehash:
                    action =  "rehash " + str(access_entry[1]) + " "
                elif access_entry[0] == MemoryAccess.CopyTemp:
                    action =  "copy_temp " + str(access_entry[1]) + " "
                else:
                    action =  "sort "
                if len(access_entry) == 1: access_entry = (access_entry[0], "")
                t = f"{counter}-{next_counter}: {action}\n"
                counter = next_counter
                for call_func in trace:
                    t += call_func
                traces += t
            with open(file_name+".debug", "w") as f:
                f.write(traces)
