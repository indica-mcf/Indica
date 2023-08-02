from indica.workflows.bayes_workflow_example import ExampleBayesWorkflow


"""
IDEAS:

including abstract class tests in here...

if save phantom profiles is mutable

does it initialise

setup models:
    does it run
    give wrong diagnostic name
    give duplicate names
    give correct names

_phantom_data:
    does it run
check dictionary:
    all have time dim
    all exist
    
    with noise
    all have error?
    shape
    
_exp_data:
    does it run
    test if no data
    test shape of data
        all have dim t

setup_optimiser
    does it run
    start points
        contains data
        dims of start points
        does sample from high density work
    
call
    does it run
    are results all np.array

"""

class TestBayesWorkflowExample:
    def setup_class(self):
        return

    def setup_method(self):
        return

    def teardown_method(self):
        return

    def test_plasma_initializes(self):
        assert hasattr(self, "plasma")


