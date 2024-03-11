import juliacall
jl = juliacall.newmodule("FK")
jl.seval("using PythonCall")

def wrap_obj_func(f):
    return jl.seval("f -> x -> pyconvert(Float64, f(x))")(f)
