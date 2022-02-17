import numpy as np
import tensorflow as tf

from CartPole.state_utilities import STATE_INDICES, STATE_VARIABLES, CONTROL_INPUTS, create_cartpole_state
from CartPole.state_utilities import ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX

from CartPole.cartpole_model import Q2u, cartpole_fine_integration, L


STATE_INDICES_TF = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(STATE_INDICES.keys())), values=tf.constant(list(STATE_INDICES.values()))),
    default_value=-100, name=None
)

class next_state_predictor_ODE():
    
    def __init__(self, dt, intermediate_steps):
        self.s = create_cartpole_state()
        self.s_next = create_cartpole_state()

        self.intermediate_steps = intermediate_steps
        self.t_step = dt / float(self.intermediate_steps)
        
    def step(self, Q, params):

        if params is None:
            pole_half_length = L
        else:
            pole_half_length = params

        u = Q2u(Q)
    
        (
            self.s_next[..., ANGLE_IDX], self.s_next[..., ANGLED_IDX], self.s_next[..., POSITION_IDX], self.s_next[..., POSITIOND_IDX], self.s_next[..., ANGLE_COS_IDX], self.s_next[..., ANGLE_SIN_IDX]
        ) = cartpole_fine_integration(
            angle=self.s[..., ANGLE_IDX],
            angleD=self.s[..., ANGLED_IDX],
            angle_cos=self.s[..., ANGLE_COS_IDX],
            angle_sin=self.s[..., ANGLE_SIN_IDX],
            position=self.s[..., POSITION_IDX],
            positionD=self.s[..., POSITIOND_IDX],
            u=u,
            t_step=self.t_step,
            intermediate_steps=self.intermediate_steps,
            L=pole_half_length,
        )

        return self.s_next



def augment_predictor_output(output_array, net_info):

    if 'angle' not in net_info.outputs:
        output_array[..., STATE_INDICES['angle']] = \
            np.arctan2(
                output_array[..., STATE_INDICES['angle_sin']],
                output_array[..., STATE_INDICES['angle_cos']])
    if 'angle_sin' not in net_info.outputs:
        output_array[..., STATE_INDICES['angle_sin']] = \
            np.sin(output_array[..., STATE_INDICES['angle']])
    if 'angle_cos' not in net_info.outputs:
        output_array[..., STATE_INDICES['angle_cos']] = \
            np.sin(output_array[..., STATE_INDICES['angle']])

    return output_array
