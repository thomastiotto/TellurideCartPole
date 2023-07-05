import argparse
from types import SimpleNamespace
from SI_Toolkit.computation_library import TensorType, NumpyLibrary
from differentiable_plasticity_adaptation.models.Hebb_MLP import HebbMLP
from differentiable_plasticity_adaptation.models.MLP import MLP
import torch
import torch.nn as nn
from torchinfo import summary

import numpy as np

from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function

from differentiable_plasticity_adaptation.train_cartpole_controller import model_def

try:
    from SI_Toolkit_ASF.predictors_customization import STATE_INDICES
except ModuleNotFoundError:
    print("SI_Toolkit_ASF not yet created")

from SI_Toolkit.Functions.General.Initialization import (get_net,
                                                         get_norm_info_for_net)
from SI_Toolkit.Functions.TF.Compile import CompileAdaptive


class controller_neural_imitator_heb(template_controller):
    _computation_library = NumpyLibrary

    def configure(self):
        self.input_at_input = self.config_controller["input_at_input"]

        self.model = model_def.define_rec_model_struct(50, 7, 2)
        # self.model = model_def.define_ff_model_struct(100, 7, 1)

        summary(self.model, input_size=(32, 7))

        self.model.load_state_dict(torch.load(
            'differentiable_plasticity_adaptation/exported/MLP_parameters-CPS-17-02-2023-UpDown-Imitation-noise-after'))
        self.model.eval()

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        import time as timer
        from CartPole.state_utilities import ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, \
            ANGLE_SIN_IDX

        net_input = [s[ANGLED_IDX], s[ANGLE_COS_IDX], s[ANGLE_SIN_IDX], s[POSITION_IDX], s[POSITIOND_IDX],
                     updated_attributes['target_position'], updated_attributes['target_equilibrium'],
                     # updated_attributes['L']
                     ]
        min, max = self.find_min_max(net_input)
        net_input = self.normalize_inputs(net_input, min, max)
        print('State', net_input)

        with torch.no_grad():
            Q = self.model(torch.Tensor(net_input)).item()
            print('Q', Q)

        return Q

    def find_min_max(self, data):
        return np.min(data), np.max(data)

    def normalize_inputs(self, data, min, max):
        return 2 * (data - min) / (max - min) - 1

    def controller_reset(self):
        self.configure()

    # def _step_compilable(self, net_input):
    #
    #     self.lib.assign(self.net_input_normed, self.normalize_inputs(net_input))
    #
    #     net_input = self.lib.reshape(self.net_input_normed, (-1, 1, len(self.net_info.inputs)))
    #
    #     net_output = self.net(net_input)
    #
    #     net_output = self.denormalize_outputs(net_output)
    #
    #     return net_output
