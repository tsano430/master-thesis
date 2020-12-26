# coding: utf-8

# file name: hals.py
# Author: Takehiro Sano
# License: GNU General Public License v3.0


import sys
import numpy as np
import numpy.linalg as LA
import warnings
warnings.simplefilter('error', category=RuntimeWarning)


def objective(X, U, V, E, l1term):
    """Compute the objective function value for NMF
    
    Parameters
    ----------
    X: given matrix
    U: factor matrix (basis)
    V: factor matrix (coefficient)
    E: X - U * V^T
    l1term: parameters for l1-regularization terms
    
    Return
    -------
    obj_val: objective function value
    """
    if E is None:
        E = X - np.dot(U, V.T)
    l1_val = l1term[0] * np.sum(np.abs(U)) + l1term[1] * np.sum(np.abs(V))
    obj_val = 0.5 * LA.norm(E, 'fro') + l1_val 
    return obj_val


def initializeUV(X, n_components, random_state=None):
    """Random initialization
    
    Parameters
    ----------
    X: given matrix
    n_components: given dimension size
    random_state: seed value
    
    Returns
    -------
    U: factor matrix (basis)
    V: factor matrix (coefficient)
    """
    row, col = X.shape
    np.random.seed(random_state)
    U = np.random.rand(row * n_components).reshape(row, n_components)
    V = np.random.rand(col * n_components).reshape(col, n_components)
    return U, V


def initializeUV_svd(X, n_components):
    """SVD-based initialization (Note that this function is in scikit-learn)
    
    Parameters
    ----------
    X: given matrix
    n_components: given dimension size
    random_state: seed value
    
    Returns
    -------
    U: factor matrix (basis)
    V: factor matrix (coefficient)
    """
    # Singular Value Decomposition
    U, S, V = LA.svd(X, n_components)
    #print(S[:n_components])
    
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = LA.norm(x_p), LA.norm(y_p)
        x_n_nrm, y_n_nrm = LA.norm(x_n), LA.norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v
    
    W = W[:, :n_components]
    H = H[:n_components, :]

    return W, H.T


# def normalizeUV(U, V):
#     """Delete ambiguous expressions for U and V by using normalization
    
#     Parameters
#     ----------
#     U: factor matrix (basis)
#     V: factor matrix (coefficient)
    
#     Returns
#     -------
#     U: unique factor matrix (basis)
#     V: unique factor matrix (coefficient)
#     """
#     Uuni = np.empty_like(U)
#     Vuni = np.empty_like(V)

#     Unrms = LA.norm(U, axis=0)

#     for i, Unrm in enumerate(Unrms):
#         Vuni[:, i] = V[:, i] * Unrm
#         if Unrm > 0.0:
#             Uuni[:, i] = U[:, i] / Unrm
#         else:
#             Uuni[:, i] = np.random.rand(U.shape[0])

#     return Uuni, Vuni


def calc_grad_U(X, U, V, E, l1term):
    """Calculate the gradient of U
    
    Parameters
    ----------
    X: given matrix
    U: factor matrix (basis)
    V: factor matrix (coefficient)
    E: X - U * V^T
    l1term: parameters for l1-regularization terms
    
    Return
    -------
    grad: the gradient of U
    """
    if E is None:
        E = X - np.dot(U, V.T)
    grad = - np.dot(E, V) + l1term[0] 
    return grad


def calc_grad_V(X, U, V, E, l1term):
    """Calculate the gradient of V
    
    Parameters
    ----------
    X: given matrix
    U: factor matrix (basis)
    V: factor matrix (coefficient)
    E: X - U * V^T
    l1term: parameters for l1-regularization terms
    
    Return
    -------
    grad: the gradient of V
    """
    if E is None:
        E = X - np.dot(U, V.T)
    grad = - np.dot(E.T, U) + l1term[1] 
    return grad


def calc_pgrad_U(X, U, V, E, l1term, tau=0.0):
    """Calculate the projected gradient of U
    
    Parameters
    ----------
    X: given matrix
    U: factor matrix (basis)
    V: factor matrix (coefficient)
    E: X - U * V^T
    l1term: parameters for l1-regularization terms
    
    Return
    -------
    pgrad: the projected gradient of U
    """
    if E is None:
        E = X - np.dot(U, V.T)
    grad = calc_grad_U(X, U, V, E, l1term)
    pgrad = np.where(U > tau, grad, np.fmin(0.0, grad))
    return pgrad


def calc_pgrad_V(X, U, V, E, l1term, tau=0.0):
    """Calculate the projected gradient of V
    
    Parameters
    ----------
    X: given matrix
    U: factor matrix (basis)
    V: factor matrix (coefficient)
    E: X - U * V^T
    l1term: parameters for l1-regularization terms
    
    Return
    -------
    pgrad: the projected gradient of V
    """
    if E is None:
        E = X - np.dot(U, V.T)
    grad = calc_grad_V(X, U, V, E, l1term)
    pgrad = np.where(V > tau, grad, np.fmin(0.0, grad))
    return pgrad


def calc_pgrad_norm(X, U, V, E, l1term, tau=0.0):
    """Calculate the projected gradient norm
    
    Parameters
    ----------
    X: given matrix
    U: factor matrix (basis)
    V: factor matrix (coefficient)
    E: X - U * V^T
    l1term: parameters for l1-regularization terms
    
    Return
    -------
    pgrad_val: the projected gradient norm
    """
    if E is None:
        E = X - np.dot(U, V.T)
    pgradU = calc_pgrad_U(X, U, V, E, l1term, tau=tau)
    pgradV = calc_pgrad_V(X, U, V, E, l1term, tau=tau)
    pgradUV = np.hstack([pgradU.T, pgradV.T])
    pgrad_val = LA.norm(pgradUV, 'fro')**2
    return pgrad_val


def stopkkt(X, U, V, E, l1term, init, tol):
    """KKT-based stopping condition
    
    Parameters
    ----------
    X: given matrix
    U: factor matrix (basis)
    V: factor matrix (coefficient)
    E: X - U * V^T
    l1term: parameters for l1-regularization terms
    init: the initial value of the projected gradient norm
    tol: tolerance error
    
    Return
    -------
    If 'True', stop the algorithm, 
    otherwise, continue the algorithm. 
    """
    if E is None:
        E = X - np.dot(U, V.T)
    pgrad_val = calc_pgrad_norm(X, U, V, E, l1term)
    if (pgrad_val / init) > tol:
        return False
    else:
        return True


def hals_algorithm(X, _U, _V, _E, l1term, normal_flag, delta=1e-8):
    """Modified HALS algorithm
    
    Parameters
    ----------
    X: given matrix
    _U: factor matrix (basis)
    _V: factor matrix (coefficient)
    _E: X - U * V^T
    l1term: parameters for l1-regularization terms
    reg_flag: regularization flag
    delta: small positive constant (usually 1e-16)
    
    Returns
    -------
    U: updated factor matrix (basis)
    V: updated factor matrix (coefficient)
    E: updated residual
    """
    U = np.copy(_U)
    V = np.copy(_V)
    E = np.copy(_E)

    dm = U.shape[1]
    uvt = np.empty_like(X)
    
    for i in range(dm):
        np.outer(U[:, i], V[:, i], uvt)
        R = E + uvt
        
        U[:, i] = np.fmax(0.0, np.dot(R, V[:, i]) + delta * U[:, i] - l1term[0]) / (np.dot(V[:, i], V[:, i]) + delta)

        if normal_flag:
            Unrm = LA.norm(U[:, i])
            V[:, i] *= Unrm
            if Unrm > 0.0:
                U[:, i] /= Unrm
            else:
                U[:, i] = np.random.rand(U.shape[0])

        if normal_flag:
            V[:, i] = np.fmax(0.0, np.dot(R.T, U[:, i]) + delta * V[:, i] - l1term[1])
        else:
            V[:, i] = np.fmax(0.0, np.dot(R.T, U[:, i]) + delta * V[:, i] - l1term[1]) / (np.dot(U[:, i], U[:, i]) + delta)
        
        np.outer(U[:, i], V[:, i], uvt)
        E = R - uvt

    return U, V, E


class NMF:
    """
    Conduct Nonnegative Matrix Factorization (NMF)
    """
    def __init__(self, n_components, max_iter=200, tol=1e-4, random_state=None, stopkkt_flag=True,
    calc_obj=False, calc_pgrad=False, eps=1e-8, l1term=(0.0, 0.0), normal_flag=None):
        """Create a new instance"""
        self.n_components = n_components
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state
        self.normal_flag = normal_flag

        # stopkkt
        self.stopkkt_flag = stopkkt_flag
        self.tol = tol

        # calculation flag
        self.calc_obj = calc_obj
        self.calc_pgrad = calc_pgrad

        # L1 regularization
        self.l1term = l1term


    def fit(self, X):
        """fit method"""
        fea, smp = X.shape

        # regularization or not
        if self.l1term == (0.0, 0.0):
            if self.normal_flag is None:
                self.normal_flag = True
        else:
            self.normal_flag = False

        # init
        U, V = initializeUV_svd(X, self.n_components)
        E = X - np.dot(U, V.T)

        if self.calc_obj:
            init_obj = objective(X, U, V, E, self.l1term)
            obj = [1.0]
        
        if self.stopkkt_flag or self.calc_pgrad:
            init_pgrad = calc_pgrad_norm(X, U, V, E, self.l1term)
            pgrad = [1.0]

        # 
        for it in range(1, self.max_iter+1):
            try:
                U, V, E = hals_algorithm(X, U, V, E, self.l1term, self.normal_flag, self.eps)
            except Exception as e:
                print('Iteration: {}'.format(it))
                raise(e)

            if self.calc_obj:
                obj_val = objective(X, U, V, E, self.l1term)
                obj.append(obj_val / init_obj)

            if self.calc_pgrad:
                pgrad_val = calc_pgrad_norm(X, U, V, E, self.l1term)
                pgrad.append(pgrad_val / init_pgrad)

            if self.stopkkt_flag and stopkkt(X, U, V, E, self.l1term, init_pgrad, self.tol):
                break
        
        self.U, self.V = U, V

        if self.calc_obj:
            self.obj = obj

        if self.calc_pgrad:
            self.pgrad = pgrad
        
        return self
    

    def get_basis(self):
        """getter for the basis matrix"""
        return self.U
    

    def get_coef(self):
        """getter for the coefficient matrix"""
        return self.V


    def get_obj(self):
        """getter for the objective function value per iteration"""
        if not self.calc_obj:
            print('Objective function value: not calculated')
            return 
        return self.obj

    
    def get_pgrad_norm(self):
        """getter for the projected gradient norm per iteration"""
        if not self.calc_pgrad:
            print('Projected gradient norm: not calculated')
            return 
        return self.pgrad

    
def pmf_hals_algorithm(X, _U, _V, _E, l1term, normal_flag, eps=1e-8):
    """Original HALS algorithm
    
    Parameters
    ----------
    X: given matrix
    _U: factor matrix (basis)
    _V: factor matrix (coefficient)
    _E: X - U * V^T
    l1term: parameters for l1-regularization terms
    reg_flag: regularization flag
    eps: small positive constant (usually 1e-16)
    
    Returns
    -------
    U: updated factor matrix (basis)
    V: updated factor matrix (coefficient)
    E: updated residual
    """
    U = np.copy(_U)
    V = np.copy(_V)
    E = np.copy(_E)
    dm = U.shape[1]
    uvt = np.empty_like(X)
    
    for i in range(dm):
        np.outer(U[:, i], V[:, i], uvt)
        R = E + uvt
        
        U[:, i] = np.fmax(eps, (np.dot(R, V[:, i]) - l1term[0]) / np.dot(V[:, i], V[:, i]))
        
        if normal_flag:
            Unrm = LA.norm(U[:, i])
            V[:, i] *= Unrm
            if Unrm > 0.0:
                U[:, i] /= Unrm
            else:
                U[:, i] = np.random.rand(U.shape[0])
            U[:, i] = np.fmax(eps, U[:, i])
            V[:, i] = np.fmax(eps, V[:, i])

        V[:, i] = np.fmax(eps, (np.dot(R.T, U[:, i]) - l1term[1]) / np.dot(U[:, i], U[:, i]))
        
        np.outer(U[:, i], V[:, i], uvt)
        E = R - uvt

    return U, V, E


class PMF(NMF):
    """
    Conduct Positive Matrix Factorization (PMF)
    """
    def __init__(self, n_components, max_iter=200, tol=1e-4, random_state=None, stopkkt_flag=True, 
    calc_obj=False, calc_pgrad=False, eps=1e-8, l1term=(0.0, 0.0), normal_flag=None):
        super().__init__(n_components, max_iter=max_iter, tol=tol, random_state=random_state, stopkkt_flag=stopkkt_flag, 
        calc_obj=calc_obj, calc_pgrad=calc_pgrad, eps=eps, l1term=l1term, normal_flag=normal_flag)


    def fit(self, X):
        """fit method"""
        fea, smp = X.shape

        # regularization or not
        if self.l1term == (0.0, 0.0):
            if self.normal_flag is None:
                self.normal_flag = True
        else:
            self.normal_flag = False

        # init
        U, V = initializeUV_svd(X, self.n_components)
        U = np.fmax(self.eps, U)
        V = np.fmax(self.eps, V)
        E = X - np.dot(U, V.T)

        if self.calc_obj:
            init_obj = objective(X, U, V, E, self.l1term)
            obj = [1.0]
        
        if self.stopkkt_flag or self.calc_pgrad:
            init_pgrad = calc_pgrad_norm(X, U, V, E, self.l1term)
            pgrad = [1.0]

        for it in range(1, self.max_iter+1):
            try:
                U, V, E = pmf_hals_algorithm(X, U, V, E, self.l1term, self.normal_flag, self.eps)
            except Exception as e:
                print('Iteration: {}'.format(it))
                raise(e)

            if self.calc_obj:
                obj_val = objective(X, U, V, E, self.l1term)
                obj.append(obj_val / init_obj)

            if self.calc_pgrad:
                pgrad_val = calc_pgrad_norm(X, U, V, E, self.l1term)
                pgrad.append(pgrad_val / init_pgrad)

            if self.stopkkt_flag and stopkkt(X, U, V, E, self.l1term, init_pgrad, self.tol):
                break
            
        self.U, self.V = U, V

        if self.calc_obj:
            self.obj = obj

        if self.calc_pgrad:
            self.pgrad = pgrad
        
        return self
