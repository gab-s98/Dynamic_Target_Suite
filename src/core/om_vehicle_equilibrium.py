import openmdao.api as om
import numpy as np

class VehicleEquilibriumComponent(om.ImplicitComponent):
    """
    A generic OpenMDAO ImplicitComponent that wraps ANY CasADi-based 
    vehicle dynamics model (3-DOF, 7-DOF, 14-DOF). It dynamically sets up inputs, states, 
    and residuals based on the provided model's metadata.
    """
    def initialize(self):
        self.options.declare('vehicle_model', types=object, 
                             desc='An instance of a CasADi vehicle model exposing get_casadi_functions() or get_trim_casadi_functions()')

    def setup(self):
        model = self.options['vehicle_model']
        
        # Support both old and new interface
        if hasattr(model, 'get_trim_casadi_functions'):
            funcs = model.get_trim_casadi_functions()
        else:
            funcs = model.get_casadi_functions()
            
        self.meta            = funcs['meta']
        self.res_func        = funcs['res_func']
        self.jac_states_func = funcs['jac_states_func']
        self.jac_inputs_func = funcs['jac_inputs_func']
        self.jac_params_func = funcs['jac_params_func']
        
        # Unit map: key -> units
        _units = {
            'v_x':             'm/s',
            'v_y':             'm/s',
            'delta_steer':     'rad',
            'wd_f':            'unitless',
            'lltd':            'unitless',
            'aero_bal':        'unitless',
            'lmuy_mult':       'unitless',
            'beta':            'rad',
            'r':               'rad/s',
            'w_fl':            'rad/s',
            'w_fr':            'rad/s',
            'w_rl':            'rad/s',
            'w_rr':            'rad/s',
            'phi':             'rad',
            'z_fl':            'm',
            'z_fr':            'm',
            'z_rl':            'm',
            'z_rr':            'm',
            'camber_f_static': 'deg',
            'camber_r_static': 'deg',
            'toe_f_static':    'deg',
            'toe_r_static':    'deg',
            'K_phi_f':         'N*m/rad',
            'K_phi_r':         'N*m/rad',
            'K_z_f':           'N/m',
            'K_z_r':           'N/m',
            'L':               'm',
            't_f':             'm',
            't_r':             'm',
        }
        
        def _get_unit(var):
            return _units.get(var, None)
        
        # Add OpenMDAO inputs (kinematics/controls)
        for var in self.meta['inputs']:
            self.add_input(var, val=0.0, units=_get_unit(var))
            
        # Add OpenMDAO inputs (parameters/design vars)
        for var in self.meta['params']:
            self.add_input(var, val=0.0, units=_get_unit(var))
            
        # Add OpenMDAO outputs (states)
        for var in self.meta['states']:
            self.add_output(var, val=0.0, units=_get_unit(var))
            
        # Declare analytical partial derivatives for states wrt states/inputs/params
        states = self.meta['states']
        inputs = self.meta['inputs']
        params = self.meta['params']
        
        for state in states:
            self.declare_partials(of=state, wrt=states)
            self.declare_partials(of=state, wrt=inputs)
            self.declare_partials(of=state, wrt=params)

    def _pack_vectors(self, inputs, outputs):
        """Helper to pack OpenMDAO dictionaries into lists for CasADi.
        np.real() is used so this works correctly under both normal and
        complex-step (force_alloc_complex=True) evaluation modes.
        """
        u = [float(np.real(inputs[k].flat[0])) for k in self.meta['inputs']]
        x = [float(np.real(outputs[k].flat[0])) for k in self.meta['states']]
        p = [float(np.real(inputs[k].flat[0])) for k in self.meta['params']]
        return u, x, p

    def apply_nonlinear(self, inputs, outputs, residuals):
        """Evaluate the residual equations using CasADi."""
        u, x, p = self._pack_vectors(inputs, outputs)
        
        res_val = np.array(self.res_func(u, x, p)).flatten()
        
        for i, var in enumerate(self.meta['states']):
            residuals[var] = res_val[i]

    def linearize(self, inputs, outputs, partials):
        """Evaluate the analytic Jacobians using CasADi."""
        u, x, p = self._pack_vectors(inputs, outputs)
        
        J_x = np.array(self.jac_states_func(u, x, p))
        J_u = np.array(self.jac_inputs_func(u, x, p))
        J_p = np.array(self.jac_params_func(u, x, p))
        
        states = self.meta['states']
        inputs_meta = self.meta['inputs']
        params_meta = self.meta['params']
        
        for i, row_var in enumerate(states):
            for j, col_var in enumerate(states):
                partials[row_var, col_var] = J_x[i, j]
                
        for i, row_var in enumerate(states):
            for j, col_var in enumerate(inputs_meta):
                partials[row_var, col_var] = J_u[i, j]
                
        for i, row_var in enumerate(states):
            for j, col_var in enumerate(params_meta):
                partials[row_var, col_var] = J_p[i, j]
