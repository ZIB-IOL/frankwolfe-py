import juliacall
jl = juliacall.newmodule("FW")
jl.seval("using PythonCall")

def wrap_objective_function(f):
    return jl.seval("f -> (x -> pyconvert(Float64, f(x)))")(f)

