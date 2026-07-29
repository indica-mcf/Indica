import indica.examples.example_diagnostic_models as example_diagnostic_models
import indica.examples.example_equilibrium as example_equilibrium
import indica.examples.example_operators as example_operators
import indica.examples.example_plasma as example_plasma
import indica.examples.example_readers as example_readers
import indica.examples.example_sawtooth_crash as example_sawtooth_crash
import indica.examples.example_transforms as example_transforms


def run_all():
    run_functions(example_equilibrium)
    run_functions(example_diagnostic_models)
    run_functions(example_operators)
    run_functions(example_plasma)
    run_functions(example_readers)
    run_functions(example_sawtooth_crash)
    run_functions(example_transforms)


def run_functions(imported):
    d = dir(imported)
    for function_name in d:
        function = getattr(imported, function_name)
        if callable(function) and function_name.startswith("example_"):
            print(f"\n*** Runnning > {imported.__name__}.{function_name} ***\n")
            _ = function()
    return


if __name__ == "__main__":
    run_all()
