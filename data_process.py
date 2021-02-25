# for Lethe and PathOHeap
import sys

train = []
val = []
with open(sys.argv[1]) as f:
    lines = ["__l"+l.strip()+"\n" for l in f.readline().split("__l") if l.strip() != ""]
    train += lines[:len(lines)//5]
    val += lines[len(lines)//5:]
with open(sys.argv[2]) as f:
    lines = ["__l"+l.strip()+"\n" for l in f.readline().split("__l") if l.strip() != ""]
    train += lines[:len(lines)//5]
    val += lines[len(lines)//5:]

with open("temp/train.txt", "w") as f:
    f.writelines(train)
with open("temp/val.txt", "w") as f:
    f.writelines(val)