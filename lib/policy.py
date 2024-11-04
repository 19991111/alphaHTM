import random
class POLICY():
    def __init__(self,candidate_topics):
        self.candidate_topics = candidate_topics

    def basic_policy(self):
        return random.choice(self.candidate_topics)