import sys
from frankwolfepy import load_julia_packages
frankwolfe, _  = load_julia_packages("FrankWolfe", "PythonCall")
sys.modules[__name__] = frankwolfe