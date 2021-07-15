from torch.nn.init import uniform_

from safe_explorer.core.config import Config
from safe_explorer.core.net import Net, ConvNet


class ConstraintModel(ConvNet):
    def __init__(self, observation_space, action_dim):
        config = Config.get().safety_layer.constraint_model

        super(ConstraintModel, self)\
            .__init__(observation_space,
                      action_dim,
                      config.layers,
                      config.init_bound,
                      uniform_,
                      None)
