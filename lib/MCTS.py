from lib.utils import *
from lib.do_eval import *
import math
from lib.policy import POLICY

class STATE:
    def __init__(self,id,topic,topic_queue=[],visits=0,reward=0,parent = None,childs = [], is_terminal = False) -> None:
        self.id = id
        self.topic = topic
        self.parent = parent
        self.childs = childs
        self.topic_queue = topic_queue
        self.visits = visits
        self.reward = reward
        self.is_terminal = is_terminal

    def is_leaf(self):
        return len(self.childs) == 0
    
    def init_root(self):
        while self.parent is not None:
            self = self.parent
        return self
    
    def state_to_dict(self):
        return {
            "id": self.id,
            "topic": self.topic,
            "topic_queue": self.topic_queue,
            "visits": self.visits,
            "reward": self.reward,
            "parent_id": self.parent.id if self.parent else None,
            "child_ids": [child.id for child in self.childs]
        }

    def opt_select(self,C=2):
        best_score = -float('inf')
        best_child = self
        for child in self.childs:
            if child.is_terminal:
                continue
            else:
                if child.visits == 0:
                    uct_socre = float('inf')
                else:
                    exploit = child.reward / child.visits
                    explore = math.sqrt(math.log(self.visits) / child.visits)
                    uct_socre = exploit + C * explore
                if uct_socre > best_score:
                    best_score = uct_socre
                    best_child = child
        return best_child
    
    def get_best_path(self):
        best_path = []
        current_node = self
        while True: 
            if current_node.is_leaf():
                break
            else:
                best_node = get_best_exploit(current_node)
                best_path.append(best_node.topic)
                current_node = best_node
        return best_path
    
    def backpropagate(self,reward):
        self.reward += reward
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(reward)
    
    def rollout(self,
                document,
                candidate_space,
                model_ckpt,
                temperature,
                top_p,
                max_tokens
                ):
        simulated_trajectories = []
        trajectories = self.topic_queue.copy()
        while True:
            candidate_topics = [get_topic(topic_queue = trajectories,
                                          doc = document,
                                          model_ckpt = model_ckpt,
                                          temperature = temperature,
                                          top_p = top_p,
                                          max_tokens = max_tokens) for i in range(candidate_space)]
            candidate_topics = drop_replicate(candidate_topics)
            if len(candidate_topics) > 0:
                policy = POLICY(candidate_topics)
                topic = policy.basic_policy()
                if is_terminal(topic,trajectories):
                    break
                else:
                    topic = processed_topic(topic)
                    simulated_trajectories.append(topic)
                    trajectories.append(topic)
            else:
                break
        eval_ct = EVALAUATION(simulated_trajectories = simulated_trajectories,
                              trajectories = trajectories,
                              document = document)
        reward = eval_ct.evaluate_candidate_trajectories(model_ckpt,temperature,top_p,max_tokens)
        return reward

    def child_expansion(self,
                document,
                candidate_space,
                model_ckpt,
                temperature,
                top_p,
                max_tokens):   
        file_path = "id.txt"
        topics_list = []
        for i in range(candidate_space):
            candidate_topic = get_topic(topic_queue = self.topic_queue,
                                          doc = document,
                                          model_ckpt = model_ckpt,
                                          temperature = temperature,
                                          top_p = top_p,
                                          max_tokens = max_tokens)
            
            candidate_id = assign_id_for_topic(file_path)
            candidate_topic = processed_topic(candidate_topic)

            if candidate_topic not in topics_list and candidate_topic not in self.topic_queue and is_legal_generation(candidate_topic):
                write_id(file_path,candidate_id)
                topics_list.append(candidate_topic)
                # generate childs
                child = STATE(
                    id=candidate_id, 
                    topic=candidate_topic, 
                    topic_queue=self.topic_queue + [candidate_topic],
                    parent=self,
                    childs=[]
                )
                self.childs.append(child)

        
class MCTS():
    def __init__(self,doc,candidate_space,model_ckpt,temperature,top_p,max_tokens):
        self.doc = doc
        self.candidate_space = candidate_space
        self.model_ckpt = model_ckpt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def get_leaf_node(self,current_node):
        while True:
            if not current_node.is_leaf():  
                if current_node == current_node.opt_select():
                    current_node.is_terminal = True
                    current_node = current_node.init_root()
                    if current_node.is_terminal:
                        break
                else:
                    current_node = current_node.opt_select()
            else:
                break
        return current_node

    def tree_search(self):
        current_node = STATE(id=0,topic=None,topic_queue=[],visits=1,reward=0,parent = None,childs = [], is_terminal = False)
        count = 0
        while True:
            current_node = current_node.init_root()
            if current_node.is_terminal == False and count<16:
                count += 1
                current_node = self.get_leaf_node(current_node)
                if current_node.visits == 0:
                    reward = current_node.rollout(document=self.doc,
                                                    candidate_space=self.candidate_space,
                                                    model_ckpt = self.model_ckpt,
                                                    temperature = self.temperature,
                                                    top_p = self.top_p,
                                                    max_tokens = self.max_tokens
                                                    )
                    current_node.backpropagate(reward)
                else:
                    current_node.child_expansion(document=self.doc,
                                                    candidate_space=self.candidate_space,
                                                    model_ckpt = self.model_ckpt,
                                                    temperature = self.temperature,
                                                    top_p = self.top_p,
                                                    max_tokens = self.max_tokens
                                                    )
                    if len(current_node.childs)>0:
                        current_node = current_node.opt_select()
                        reward = current_node.rollout(document=self.doc,
                                                    candidate_space=self.candidate_space,
                                                    model_ckpt = self.model_ckpt,
                                                    temperature = self.temperature,
                                                    top_p = self.top_p,
                                                    max_tokens = self.max_tokens
                                                    )
                        current_node.backpropagate(reward)
                    else:
                        current_node.is_terminal = True
            else:
                current_node = current_node.init_root()
                return current_node.get_best_path()



