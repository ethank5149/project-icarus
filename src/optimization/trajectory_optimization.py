import openmdao.api as om
import joblib
import numpy as np

class AeroSurrogateComponent(om.ExplicitComponent):
    """
    Wraps the GPR surrogate for OpenMDAO.
    """
    def initialize(self):
        self.options.declare('model_path', default='aero_surrogate.pkl')

    def setup(self):
        self.add_input('mach', val=2.0)
        self.add_input('alpha', val=0.0)
        self.add_output('drag_force', val=0.0)
        self.gpr = joblib.load(self.options['model_path'])

    def compute(self, inputs, outputs):
        X_new = np.array([[inputs['mach'][0], inputs['alpha'][0]]])
        cd_coeff = self.gpr.predict(X_new)
        
        # Physics calculation
        # Force = 0.5 * rho * v^2 * Area * Cd
        rho = 1.225
        outputs['drag_force'] = 0.5 * rho * 500**2 * 0.1 * cd_coeff

class InterceptorProb(om.Problem):
    def setup(self):
        model = self.model
        group = model.add_subgroup('flight')
        
        # Add the surrogate component
        group.add_subsystem('aero', AeroSurrogateComponent())
        
        # Add a design variable
        group.add_design_var('aero.alpha', lower=0, upper=15)
        group.add_objective('aero.drag_force')

# Initialize and run
prob = InterceptorProb()
prob.setup()
prob.run_model()

print(f"Optimal Drag Found: {prob.get_val('flight.aero.drag_force')[0]:.2f} N")