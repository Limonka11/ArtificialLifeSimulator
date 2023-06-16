import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override

class TorchRNNModel(RecurrentNetwork, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 fc_size=28,
                 lstm_state_size=32):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self.obs_size = get_preprocessor(obs_space)(obs_space).size
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size

        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=2)

        self.conv4 = nn.Conv1d(1, 3, kernel_size=3)
        self.conv5 = nn.Conv1d(3, 6, kernel_size=3, stride=2)

        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        # self.fc2 = nn.Linear(self.fc_size, (self.fc_size)//2)
        # self.fc3 = nn.Linear(10, 5)
        self.lstm = nn.LSTM(
            12, self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        h = [
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    # @override(RecurrentNetwork)
    # def forward_rnn(self, inputs, state, seq_lens):
    #     """Feeds `inputs` (B x T x ..) through the Gru Unit.
    #     Returns the resulting outputs as a sequence (B x T x ...).
    #     Values are stored in self._cur_value in simple (B) shape (where B
    #     contains both the B and T dims!).
    #     Returns:
    #         NN Outputs (B x T x ...) as sequence.
    #         The state batches as a List of two items (c- and h-states).
    #     """
    #     #print("HMM", inputs.size())
    #     x, y = torch.split(inputs, [49, 10], dim=2)
    #     x = nn.functional.relu(self.fc1(inputs))
    #     x = nn.functional.relu(self.fc2(x))

    #     y = nn.functional.relu(self.fc3(y))

    #     #print("HM: ", x.size() , "hm: ", y.size())
    #     x = torch.cat((x, y), dim=2)
    #     print("WTF:", x.shape)
    #     #print("HM: ", x.size())
    #     self._features, [h, c] = self.lstm(
    #         x, [torch.unsqueeze(state[0], 0),
    #             torch.unsqueeze(state[1], 0)])
    #     action_out = self.action_branch(self._features)
    #     return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.
        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).
        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        x, y = torch.split(inputs, [49, 10], dim=2)
        x = x[:, :, None,  :]
        x = torch.reshape(x, (-1, 1, 7, 7))

        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))

        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.reshape(x, (-1, inputs.size(dim=1), 6))

        y = torch.reshape(y, (-1, 1, 10))
        y = nn.functional.relu(self.conv4(y))
        y = nn.functional.max_pool1d(y, 2)
        y = nn.functional.relu(self.conv5(y))

        y = torch.permute(y, (0, 2, 1))
        y = torch.reshape(y, (-1, inputs.size(dim=1), 6))

        #print("HM: ", x.size() , "hm: ", y.size())
        x = torch.cat((x, y), dim=2)
        #print("HM: ", x.size())
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0),
                torch.unsqueeze(state[1], 0)])
        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]