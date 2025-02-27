import pydot
import networkx as nx
import random
from pydot import graph_from_dot_file
import json
import re
import random
import networkx as nx

def load_dot_file(dot_file_path):
    print(dot_file_path)
    (graph,) = pydot.graph_from_dot_file(dot_file_path)
    if graph is None:
        raise ValueError(f"无法加载DOT文件：{dot_file_path}")
        return None

    nx_graph = nx.nx_pydot.from_pydot(graph)
    return nx_graph

def clean_label(code):
    code = code[code.index(",") + 1:-2].split('\\n')[0]
    code = code.strip() 
    code = code.replace("static", "")
    code = ','.join(dict.fromkeys(code.split(',')))
    code = re.sub(r'\s+', ' ', code)     
    return code

def adjust_node_weight(node_label, keywords):

    weight = 1
    for keyword in keywords:
        if keyword in node_label:
            weight += 100  
    return weight

def generate_path_with_score(G, source, target, max_length, keywords):
    path = [source]
    visited_count = {source: 1}  
    current = source
    path_score = 0  

    while current != target:
        if len(path) >= max_length:  
            return None, 0  

        neighbors = list(G.neighbors(current))
        if not neighbors:  
            return None, 0


        available_neighbors = [n for n in neighbors if visited_count.get(n, 0) < 2]
        if not available_neighbors: 
            return None, 0

 
        weights = []
        for n in available_neighbors:
            node_label = G.nodes[n].get('label', '')  
            weight = adjust_node_weight(node_label, keywords)  
            weights.append(weight)

        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        current = random.choices(available_neighbors, weights=probabilities, k=1)[0]

        node_label = G.nodes[current].get('label', '')
        path_score += adjust_node_weight(node_label, keywords)

        visited_count[current] = visited_count.get(current, 0) + 1
        path.append(current)

    return path, path_score

#路径生成
def get_three_high_score_paths(G, source, target, keywords):
    num_nodes = len(G)
    max_length = int(1 * num_nodes)  
    all_paths = []
    finally_path=[]
    attempts = 0
    max_attempts = 50

    while attempts < max_attempts and len(all_paths) < 3:
        path, path_score = generate_path_with_score(G, source, target, max_length, keywords)
        if path and len(path) <= max_length and path not in [p for p, _ in all_paths]:  
            all_paths.append((path, path_score))
        attempts += 1


    all_paths = sorted(all_paths, key=lambda x: x[1], reverse=True)
    for i, (path, score) in enumerate(all_paths[:3], start=1):
        print(f"Path {i}: Score: {score}")
        finally_path.append(path)
   
    # print(finally_path)

    return finally_path

#路径简化
def remove_contained_strings(strings):
    filtered_strings = []
    for i in range(len(strings) - 1, -1, -1):
        current_string = strings[i]
        if not any(current_string in s and current_string != s for s in filtered_strings):
        # if not any(current_string in s for s in filtered_strings):
            filtered_strings.append(current_string)
    return filtered_strings[::-1]

def find_method_paths(dot_file_path):
    G = load_dot_file(dot_file_path)
    nodes = list(G.nodes())
    source = None
    target = None
    for node in nodes:
        label = G.nodes[node].get('label', '')
        if label.startswith('"(METHOD') and not source:
            source = node 
        if label.startswith('"(METHOD_RETURN') and not target:
            target = node  

    if source is None or target is None:
        print("No valid source or target nodes found based on labels.")
        return None, None, None
    
    cfg_edges = []
    for u, v, data in G.edges(data=True):
        cfg_edges.append((u, v))
            #print(u, v)

    cfg_G = nx.DiGraph()
    cfg_G.add_nodes_from(G.nodes())
    cfg_G.add_edges_from(cfg_edges)

    keywords=['strcpy','wcscpy','strcat','wcscat','gets', # 字符串操作函数没有边界检查，容易导致缓冲区溢出。
         'malloc','calloc','realloc','free',         # 内存分配或释放不当可能导致内存泄漏、双重释放或未初始化内存使用
         'printf','fprintf','sprintf',                  # 格式化字符串函数未正确指定格式化参数，可能导致信息泄漏或代码注入
         'fopen','freopen','fgets',                  # 文件操作时未正确检查文件路径、权限或返回值，可能导致文件遍历漏洞
         'vsprintf','strncpy', 		#指针未初始化
         'memcpy', 'memmove','memcmp', 'snprintf',  #参数长度错误
         'alloca', 'strtol','atoi', '<<', '>>','>=','<=','/','-','+'
         'fopen', 'scanf', 'fgets', 'fread','fwrite','sizeof','NULL','assert']
    paths = get_three_high_score_paths(G, source, target, keywords)


    label_paths = []
    for path in paths:
        code = [clean_label(G.nodes[node].get('label', '')) for node in path]
        label_paths.append(code)
    return source, target, label_paths
def process_dot_files(input_json_path, output_json_path):
    with open(input_json_path, 'r') as infile:
        data = json.load(infile)


    output_data = []
    for item in data:
        dot_file_path = item.get('dot') 
        #print(dot_file_path)
        target = item.get('target')
        func_id = item.get('id')
        # func =item.get('func')
        func =item.get('func')
        if not dot_file_path :
            print('11')
            continue  
        source_node, target_node, func_paths = find_method_paths(dot_file_path)
        all_path=[]
        for path in func_paths:
            path=remove_contained_strings(path)
            clean_path = '; '.join(path) 
            all_path.append(clean_path)
        if source_node and target_node:
            output_data.append({
                'id': func_id,
                'target': target,
                'path': all_path,  
                'func' : func
            })

        with open(output_json_path, 'w') as outfile:
            json.dump(output_data, outfile,indent=4)
    print(f"Output saved to {output_json_path}")


input_json_path = "/mnt/e/zzz/finally_data/devign/part_2.json"  
output_json_path = "/mnt/e/zzz/finally_data/devign/path-key-3-random/smi-key/part_2_cfg.json"  
# input_json_path = "/mnt/e/zzz/finally_data/reveal/part_2.json" 
# output_json_path = "/mnt/e/zzz/finally_data/reveal/path-key-2-random/random1/part_2_cfg.json"  
process_dot_files(input_json_path, output_json_path)