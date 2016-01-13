import numpy as np

def project_to_psd(inp_mat):
    sym_mat = (inp_mat + inp_mat.T)*0.5   
    e_vals,e_vecs = np.linalg.eig(sym_mat)
    nz_evals = np.maximum(e_vals, 0)
    out_mat = np.matmul(np.matmul(e_vecs,np.matrix(np.diag(nz_evals))),e_vecs.T)
    return out_mat
