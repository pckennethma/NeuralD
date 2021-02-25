import fasttext, os

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

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))
    return p

def train(file_path):
    merge_file(os.path.join(file_path, "1.txt"), os.path.join(file_path, "2.txt"), os.path.join(file_path, "train.txt"))
    return fasttext.train_supervised(os.path.join(file_path, "train.txt"))

def validate(model, file_path):
    merge_file(os.path.join(file_path, "1-val.txt"), os.path.join(file_path, "2-val.txt"), os.path.join(file_path, "val.txt"))
    return print_results(*model.test(os.path.join(file_path, 'val.txt')))

if __name__ == "__main__":
    merge_file("1.txt", "2.txt", "train.txt")
    merge_file("1-val.txt", "2-val.txt", "val.txt")
    model = fasttext.train_supervised('train.txt')
    print_results(*model.test('val.txt'))
