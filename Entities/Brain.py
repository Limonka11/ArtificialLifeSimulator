import random
class Brain:
    def __init__(self, input_dim, output_dim, algorithm):
        self.algorithm = algorithm
        self.input_dim = input_dim
        self.output_dim = output_dim

class RandBrain(Brain):
    def __init__(self, input_dim=1087, output_dim=5, learning_rate=0.001, train_freq=20,
                 load_model=False, training=True):
        super().__init__(input_dim, output_dim, "Rand")

    def act(self, state, n_epi):
        return random.randint(0, 3)

    def train(self):
        pass

    def learn(self, age, dead, action, state, reward, state_prime, done, n_epi):
        pass

    def mutate(self):
        pass