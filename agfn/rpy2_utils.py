import rpy2 
import os 
import sys 
import rpy2.robjects as robjects # type: ignore  
from rpy2.robjects.packages import importr 

from typing import Callable 
from contextlib import contextmanager 

rutils = importr('utils') 

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def install_uninstalled(package, install_repo: Callable = lambda pack: rutils.install_packages(pack), translations=dict()):
    try: 
        return importr(package, robject_translations=translations) 
    except rpy2.rinterface_lib.embedded.RRuntimeError: 
        install_repo(package) 
        return importr(package, robject_translations=translations) 

def install_packages(): 
    BIOC_PACKAGES = ['ggm', 'graph', 'RBGL', 'Rgraphviz'] 
    CRAN_PACKAGES = ['dagitty', 'igraph', 'SEMgraph', 'lavaan']
    
    # Install BiocManager 
    biocmanager = install_uninstalled('BiocManager') 
    
    for package in BIOC_PACKAGES: 
        install_uninstalled(package, install_repo=biocmanager.install) 
    
    # Install required packages 
    for package in CRAN_PACKAGES: 
        install_uninstalled(package, translations={'.env': '__env'} if package == 'igraph' else dict()) 

def tensor3d_to_rmatrix(tensor3d): 
    return [
        t.cpu().numpy() for t in tensor3d  
    ] 

if __name__ == '__main__': 
    install_packages() 