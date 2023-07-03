import argparse
from types import SimpleNamespace
from SI_Toolkit.computation_library import TensorType, NumpyLibrary
from differentiable_plasticity_adaptation.models.Hebb_MLP import HebbMLP
from differentiable_plasticity_adaptation.models.MLP import MLP

import numpy as np

from Control_Toolkit.Controllers import template_controller
from SI_Toolkit.Functions.General.Normalising import get_normalization_function, get_denormalization_function

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

        args = SimpleNamespace(**{'input_size': 8,
                                  'hidden_size': 50,
                                  'output_size': 1,
                                  'depth': 1,
                                  'lr': 2e-3,
                                  'epochs': 100,
                                  'show_loss': 1})
        self.model = MLP(args)
        self.model.load_parameters(
            '/Users/thomas/Desktop/MLP_parameters-cartpole-trained-3.pickle')

    def step(self, s: np.ndarray, time=None, updated_attributes: "dict[str, TensorType]" = {}):
        import time as timer
        from CartPole.state_utilities import ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, \
            ANGLE_SIN_IDX

        net_input = [s[ANGLED_IDX], s[ANGLE_COS_IDX], s[ANGLE_SIN_IDX], s[POSITION_IDX], s[POSITIOND_IDX],
                     updated_attributes['target_position'], updated_attributes['target_equilibrium'],
                     updated_attributes['L']]
        print('State', net_input)
        # TODO outputs are tiny
        start_t = timer.time()
        Q = self.model.infer(net_input)
        print('Time for NN step: ', timer.time() - start_t)

        print('Q', Q)

        return Q

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
