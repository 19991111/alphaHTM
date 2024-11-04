from lib.utils import get_topic,processed_topic,is_legal_generation


class BENCHMARK():
    def __init__(self,doc,model_ckpt,temperature,top_p,max_tokens) -> None:
        self.doc = doc
        self.model_ckpt = model_ckpt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
    
    def only_llm_benchmark(self):
        topic_queue = []
        while True:
            topic = get_topic(topic_queue,self.doc,self.model_ckpt,self.temperature,self.top_p,self.max_tokens)
            if  processed_topic(topic) not in topic_queue and is_legal_generation(topic):
                topic = processed_topic(topic)
                topic_queue.append(topic)
            else:
                break
        return topic_queue
    





        
