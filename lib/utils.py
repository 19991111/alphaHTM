import ast
from lib.model_api import LLM, get_tokenizer
import tiktoken

def processed_topic(topic):
    topic = topic.lower()
    topic = topic.strip()
    return topic

def calculate_tokens(message,tokenizer):
    encoded_input = tokenizer(message, return_tensors='pt')
    num_tokens = len(encoded_input['input_ids'][0])
    return num_tokens

def is_dict_string(s):
    try:
        obj = ast.literal_eval(s)
        return isinstance(obj, dict)
    except (ValueError, SyntaxError):
        return False
    
def write_id(file_path,id):
    with open(file_path,"a+",encoding="utf-8") as f:
        f.write(str(id)+"\n")
        f.close()

def assign_id_for_topic(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    final_id = lines[-1].strip()
    return int(final_id) + 1

def is_legal_generation(topic):
    if topic == "<eog>" or topic == "None":
        return False
    else:
        return True
    
def get_topic(topic_queue,doc,model_ckpt,temperature,top_p,max_tokens):
    if len(topic_queue) == 0:
        prompt = open("./prompt/primary_topics_generation.txt", "r").read()
        prompt = prompt.format(Document=doc)
    else:
        topic_queue_str = ",".join(topic_queue)
        prompt = open("./prompt/forward_do_generate.txt", "r").read()
        prompt = prompt.format(Topics=topic_queue_str, Document=doc)
    
    llm = LLM(model_ckpt)
    topic = llm.local_LLM_generate(prompt=prompt,
                                   temperature=temperature,
                                   top_p = top_p,
                                   max_tokens = max_tokens)
    if is_dict_string(topic):
        topic_dict = ast.literal_eval(topic)
        legal_topic = topic_dict.get("topic","None")
    else:
        legal_topic = "None"

    return legal_topic
    
def backward_get_topic(reversed_simulated_trajectories,doc,model_ckpt,temperature,top_p,max_tokens):
    prompt = open("./prompt/backward_do_generate.txt", "r").read()
    topics_str = ",".join(reversed_simulated_trajectories)
    prompt = prompt.format(Topics=topics_str,Document=doc)

    llm = LLM(model_ckpt)
    topic = llm.local_LLM_generate(prompt=prompt,
                                   temperature=temperature,
                                   top_p = top_p,
                                   max_tokens = max_tokens)
    
    if is_dict_string(topic):
        topic_dict = ast.literal_eval(topic)
        legal_topic = topic_dict.get("topic","None")
    else:
        legal_topic = "None"
    return legal_topic


def is_terminal(topic,trajectories):
    if (topic == "<eog>") or (topic == "None") or (processed_topic(topic) in trajectories):
        return True
    else:
        return False

def get_best_exploit(node):
    node_exploits = [node.reward / node.visits if node.visits > 0 else float('inf') for node in node.childs]
    best_index = max(range(len(node_exploits)), key=node_exploits.__getitem__)
    best_node = node.childs[best_index]
    return best_node

def drop_replicate(topic_list):
    topic_list = list(set(topic_list))
    return topic_list

def truncate(prompt,doc,max_token=4096):
    encoding = tiktoken.get_encoding("cl100k_base")
    prompt_len = calculate_tokens(prompt)
    doc_len = calculate_tokens(doc)
    doc_token = encoding.encode(doc)
    if prompt_len > max_token:
        max_doc_len = max_token - (prompt_len-doc_len)
        doc_token = doc_token[: max_doc_len - 1003]
        truncated_doc = encoding.decode(doc_token)
    else:
        truncated_doc = doc
    return truncated_doc



        


