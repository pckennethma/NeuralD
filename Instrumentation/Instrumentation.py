import pasta, ast
import sys 
counter = 0

def instrumentation(src_file, dest_file, white_list={}):
    for k in white_list:
        white_list[k] = set(white_list[k])
    with open(src_file) as f:
        source_code = f.read()
    tree = pasta.parse(source_code)
    tree.body.insert(0, pasta.parse("from ..util import instrumentation_visited\n"))
    
    instrumentation_node_types = set([ "If", "FunctionDef", "For", "While", "With", "Try", "Except", "IfExp" ])

    def in_while_list(tree):
        if not hasattr(tree, "name"): return False
        name = tree.name
        class_name = tree.__class__.__name__
        if class_name in white_list and name in white_list[class_name]:
            return True
        else: return False

    def deep_first_insertion(tree):
        global counter
        if in_while_list(tree): return
        if hasattr(tree, "body"):
            for subtree in tree.body:
                deep_first_insertion(subtree)
            if tree.__class__.__name__ in instrumentation_node_types:
                func_call_node = ast.Call(func=ast.Name("instrumentation_visited", ast.Load()), args=[ast.Constant(value=counter)], keywords=[])
                tree.body.insert(0,ast.Expr(func_call_node))
                counter += 1
        if hasattr(tree, "orelse") and len(tree.orelse) > 0:
            for subtree in tree.orelse:
                deep_first_insertion(subtree)
            if tree.orelse[0].__class__.__name__ == "If": return
            func_call_node = ast.Call(func=ast.Name("instrumentation_visited", ast.Load()), args=[ast.Constant(value=counter)], keywords=[])
            tree.orelse.insert(0,ast.Expr(func_call_node))
            counter += 1
    deep_first_insertion(tree)
    ast.fix_missing_locations(tree)
    transformed_code = pasta.dump(tree)
    with open(dest_file, "w")  as f:
        f.write(transformed_code)
    return counter

if __name__ == "__main__":
    instrumentation("../buggySqRoot1.py", "./buggySqRoot1Inst.py", {"FunctionDef": {"ConstructRandomOracle"}})