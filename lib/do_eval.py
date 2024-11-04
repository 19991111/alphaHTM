from lib.utils import backward_get_topic,is_terminal,processed_topic
from lib.model_api import LLM
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from lib.policy import POLICY

def DTW_reward(list1,list2):
    distance, _ = fastdtw(list1, list2, dist=cosine)
    reward = 1 - distance/(1+distance)
    return reward

class EVALAUATION():
    def __init__(self,candidate_space=2,simulated_trajectories=[],trajectories=[],document=None):
        self.simulated_trajectories = simulated_trajectories
        self.candidate_space = candidate_space
        self.trajectories = trajectories
        self.document = document
    
    def drop_replicate(self,topic_list):
        topic_list[:] = filter(lambda x: x is not None, topic_list)
        topic_list = list(set(topic_list))
        return topic_list
    
    def get_forward_truth_topics(self):
        return [item for item in self.trajectories.copy() if item not in self.simulated_trajectories.copy()]
    
    def reverse_generation(self,model_ckpt,temperature,top_p,max_tokens):
        backward_simulated_trajectories = []
        reversed_simulated_trajectories = self.simulated_trajectories[::-1].copy()
        if self.document is not None:
            while True:
                candidate_topics = [backward_get_topic(reversed_simulated_trajectories = reversed_simulated_trajectories,
                                                    model_ckpt = model_ckpt,
                                                    doc = self.document,
                                                    temperature = temperature,
                                                    top_p = top_p,
                                                    max_tokens = max_tokens) for i in range(self.candidate_space)]
                
                candidate_topics = self.drop_replicate(candidate_topics)
                if len(candidate_topics) > 0:
                    policy = POLICY(candidate_topics)
                    topic = policy.basic_policy()
                    if is_terminal(topic,reversed_simulated_trajectories):
                        break
                    else:
                        topic = processed_topic(topic)
                        backward_simulated_trajectories.append(topic)
                        reversed_simulated_trajectories = reversed_simulated_trajectories + [topic]
                else:
                    return []
            reversed_backward_topics = backward_simulated_trajectories[::-1]
            return reversed_backward_topics        
        else:
            return None

    def evaluate_candidate_trajectories(self,model_ckpt,temperature,top_p,max_tokens):
        forward_truth_topics = self.get_forward_truth_topics()
        reversed_backward_topics = self.reverse_generation(model_ckpt,temperature,top_p,max_tokens)
        if reversed_backward_topics is not None and len(reversed_backward_topics)>0 and len(forward_truth_topics)>0:
            llm = LLM(model_ckpt)
            forward_truth_topics_embedding = llm.get_topics_embedding(forward_truth_topics)
            reversed_backward_topics_embedding = llm.get_topics_embedding(reversed_backward_topics)
            return DTW_reward(forward_truth_topics_embedding,reversed_backward_topics_embedding)
        else:
            return 0


# 学习一下in-context learning和LLM如何利用MCTS进行训练
class Models():
    def __init__(self,model):
        
        pass

    def policy_network(self):
        pass
    def value_network(self):
        pass

    def overall_network(self):
        pass    



