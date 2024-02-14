import shutil
from jill.install import install_julia

def _find_julia():
    return shutil.which("julia")

def _ensure_julia_installed():
    if not _find_julia():
        print("No Julia version found. Installing Julia.")
        install_julia()
        if not _find_julia():
            raise RuntimeError(
                "Julia installed with jill but `julia` binary cannot be found in the path"
            )

def load_julia_packages(*names):
    """
    Load Julia packages and return references to them, automatically installing julia and
    the packages as necessary.
    """
    _ensure_julia_installed()

    script = """import Pkg
    Pkg.activate(\"frankwolfepy\", shared=true)
    try
        import {0}
    catch e
        e isa ArgumentError || throw(e)
        Pkg.add([{1}])
        import {0}
    end
    {0}""".format(", ".join(names), ", ".join(f'"{name}"' for name in names))

    script = script.replace("\n", ";")

    from juliacall import Main
    return Main.seval(script)

