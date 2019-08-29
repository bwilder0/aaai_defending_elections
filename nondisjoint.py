import numpy as np
from numba import jit
from utils import greedy, project_uniform_matroid_boundary
from functools import partial 

@jit
def gradient_budget(x, P, w):
    n = len(w)
    m = len(x)
    grad = np.zeros(m, dtype=np.float32)
    for i in range(n):
        p_fail = 1 - x*P[:,i]
        p_all_fail = np.prod(p_fail)
        for j in range(m):
            grad[j] += w[i] * P[j, i] * p_all_fail/p_fail[j]
    return grad

@jit
def hessian_budget(x, P, w):
    n = len(w)
    m = len(x)
    hessian = np.zeros((m,m), dtype=np.float32)
    for i in range(n):
        p_fail = 1 - x*P[:,i]
        p_all_fail = np.prod(p_fail)
        for j in range(m):
            for k in range(m):
                hessian[j, k] = -w[i] * P[j, i] * p_all_fail/(p_fail[j] * p_fail[k])
    return hessian

@jit
def objective_budget(x, P, w):
    n = len(w)
    total = 0
    for i in range(n):
        p_fail = 1 - x*P[:,i]
        p_all_fail = np.prod(p_fail)
        total += w[i] * (1 - p_all_fail)
    return total

def objective_budget_set(S, P, w):
    return objective_budget(indicator(S, P.shape[0]), P, w)

@jit
def get_p_reached(x, P):
    m = P.shape[1]
    p_reach = np.zeros((m))
    for i in range(m):
        p_fail = 1 - x*P[:,i]
        p_reach[i] = 1 - np.prod(p_fail)
    return p_reach

def indicator(S, n):
    x = np.zeros((n))
    x[list(S)] = 1
    return x

    return x

def oga(P, preferences, k_d, k_a, num_iter, step_size = 0.01, verbose=False):
    '''
    Runs online gradient ascent from the perspective of the adversary, with 
    greedy best responses for the defender. Returns the list of best responses
    played by the defender (the uniform distribution over which is their historical
    mixed strategy)
    
    P_ij should be the probability channel i reaches voter j
    
    preferences is a 0-1 vector where 1 indicates that a voter preferes c_d
    
    k_d/k_a are the defender and attacker budgets
    '''
    #random initialization
    n = P.shape[0]
    x_a = project_uniform_matroid_boundary(np.random.rand((n)), k_a)
    #historical mixed strategy for defender
    sigma_d = []
    values = []
    for t in range(num_iter):
        #get current probability the attacker reaches each voter
        p_attacker_reach = get_p_reached(x_a, P)
        #solve best response for the defender
        obj_defender = partial(objective_budget_set, P = P, w = preferences*p_attacker_reach)
        S_d, _ = greedy(range(n), k_d, obj_defender)
        sigma_d.append(S_d)
        x_d = indicator(S_d, n)
        #gradient step + projection on the attacker's mixed strategy
        p_defender_reach = get_p_reached(x_d, P)
        grad_attacker = gradient_budget(x_a, P, preferences*(1 - p_defender_reach))
        objective_value = objective_budget(x_a, P, preferences*(1 - p_defender_reach))
        values.append(objective_value)
        if verbose:
            print(t, objective_value)
        x_a = project_uniform_matroid_boundary(x_a + step_size*grad_attacker, k_a)
    return sigma_d, x_a, values

def exponentiated_gradient(P, preferences, k_d, k_a, num_iter, step_size = 0.01, verbose=False):
    '''
    Runs online gradient ascent from the perspective of the adversary, with 
    greedy best responses for the defender. Returns the list of best responses
    played by the defender (the uniform distribution over which is their historical
    mixed strategy)
    
    P_ij should be the probability channel i reaches voter j
    
    preferences is a 0-1 vector where 1 indicates that a voter preferes c_d
    
    k_d/k_a are the defender and attacker budgets
    '''
    #random initialization
    n = P.shape[0]
    x_a = project_uniform_matroid_boundary(np.random.rand((n)), k_a)
    #historical mixed strategy for defender
    sigma_d = []
    values = []
    for t in range(num_iter):
        #get current probability the attacker reaches each voter
        p_attacker_reach = get_p_reached(x_a, P)
        #solve best response for the defender
        obj_defender = partial(objective_budget_set, P = P, w = preferences*p_attacker_reach)
        S_d, _ = greedy(range(n), k_d, obj_defender)
        sigma_d.append(S_d)
        x_d = indicator(S_d, n)
        #gradient step + projection on the attacker's mixed strategy
        p_defender_reach = get_p_reached(x_d, P)
        grad_attacker = gradient_budget(x_a, P, preferences*(1 - p_defender_reach))
        objective_value = objective_budget(x_a, P, preferences*(1 - p_defender_reach))
        values.append(objective_value)
        if verbose:
            print(t, objective_value)
#        x_a = project_uniform_matroid_boundary(x_a + step_size*grad_attacker, k_a)
        x_a = x_a * np.exp(step_size*grad_attacker)
        x_a[x_a > 1] = 1
        x_a = k_a*x_a/x_a.sum()
    return sigma_d, x_a, values

def exponentiated_gradient_asymmetric(P, preferences, k_d, k_a, num_iter, n_samples=100, step_size = 0.01, verbose=False):
    '''
    Runs online gradient ascent from the perspective of the adversary, with 
    greedy best responses for the defender. Returns the list of best responses
    played by the defender (the uniform distribution over which is their historical
    mixed strategy)
    
    P_ij should be the probability channel i reaches voter j
    
    preferences is a 0-1 vector where 1 indicates that a voter preferes c_d
    
    k_d/k_a are the defender and attacker budgets
    '''
    #random initialization
    n = P.shape[0]
    x_a = np.random.rand((n_samples, n))
    for i in range(n_samples):
        x_a[i] = project_uniform_matroid_boundary(x_a[i], k_a)
    #historical mixed strategy for defender
    sigma_d = []
    for t in range(num_iter):
        #get current probability the attacker reaches each voter
        w = np.zeros((P.shape[1]))
        for i in range(n_samples):
            w += preferences[i]*get_p_reached(x_a[i], P)
        w /= n_samples
        #solve best response for the defender
        obj_defender = partial(objective_budget_set, P = P, w = w)
        S_d, _ = greedy(range(n), k_d, obj_defender)
        sigma_d.append(S_d)
        x_d = indicator(S_d, n)
        #gradient step + projection on the attacker's mixed strategy
        p_defender_reach = get_p_reached(x_d, P)
        for i in range(n_samples):
            grad_attacker = gradient_budget(x_a[i], P, preferences[i]*(1 - p_defender_reach))
#            objective_value = objective_budget(x_a[i], P, preferences[i]*(1 - p_defender_reach))
#            values.append(objective_value)
#            if verbose:
#                print(t, objective_value)
    #        x_a = project_uniform_matroid_boundary(x_a + step_size*grad_attacker, k_a)
            x_a[i] = x_a[i] * np.exp(step_size*grad_attacker)
            x_a[i][x_a[i] > 1] = 1
            x_a[i] = k_a*x_a[i]/x_a[i].sum()
    return sigma_d, x_a

def attacker_br_value(sigma_d, P, preferences, k_a):
    '''
    Find a greedy best response to a given defender mixed strategy, given as
    the uniform distribution over the list of sets sigma_d
    '''
    n = P.shape[0]
    p_defender_reach = np.zeros((P.shape[1]))
    #average over sigma_d to get the probability each voter is reached by the 
    #defender
    for S_d in sigma_d:
        x = indicator(S_d, n)
        p_defender_reach += get_p_reached(x, P)
    p_defender_reach /= len(sigma_d)
    #call greedy for best response
    obj_attacker = partial(objective_budget_set, P = P, w = preferences*(1 - p_defender_reach))
    S_a, value = greedy(range(n), k_a, obj_attacker)
    return value


def attacker_br_mip_value(sigma_d, P, preferences, k_a):
    '''
    Find a greedy best response to a given defender mixed strategy, given as
    the uniform distribution over the list of sets sigma_d
    '''
    n = P.shape[0]
    p_defender_reach = np.zeros((P.shape[1]))
    #average over sigma_d to get the probability each voter is reached by the 
    #defender
    for S_d in sigma_d:
        x = indicator(S_d, n)
        p_defender_reach += get_p_reached(x, P)
    p_defender_reach /= len(sigma_d)
    #call greedy for best response
    obj_attacker = partial(objective_budget_set, P = P, w = preferences*(1 - p_defender_reach))
    S_a = mip_br(P, k_a, preferences*(1 - p_defender_reach))
    return obj_attacker(S_a)

def attacker_br_mip_value_asymmetric(sigma_d, P, preferences, k_a):
    '''
    Find a greedy best response to a given defender mixed strategy, given as
    the uniform distribution over the list of sets sigma_d
    '''
    n = P.shape[0]
    p_defender_reach = np.zeros((P.shape[1]))
    #average over sigma_d to get the probability each voter is reached by the 
    #defender
    for S_d in sigma_d:
        x = indicator(S_d, n)
        p_defender_reach += get_p_reached(x, P)
    p_defender_reach /= len(sigma_d)
    value = 0
    for i in range(preferences.shape[0]):
        obj_attacker = partial(objective_budget_set, P = P, w = preferences[i]*(1 - p_defender_reach))
        S_a = mip_br(P, k_a, preferences[i]*(1 - p_defender_reach))
        value += obj_attacker(S_a)
    value /= preferences.shape[0]
    return value


def defender_br_mip_value(x_a, P, preferences, k_d):
    '''
    Find a greedy best response to a given defender mixed strategy, given as
    the uniform distribution over the list of sets sigma_d
    '''
    n = P.shape[0]
    p_attacker_reach = get_p_reached(x_a, P)    
    S_d = mip_br(P, k_d, preferences*p_attacker_reach)
    p_defender_reach = get_p_reached(indicator(S_d, n), P) 
    value = objective_budget(x_a, P, preferences*(1 - p_defender_reach))
    return value

def defender_br_mip_value_asymmetric(x_a, P, preferences, k_d):
    '''
    Find a greedy best response to a given defender mixed strategy, given as
    the uniform distribution over the list of sets sigma_d
    '''
    n = P.shape[0]
    w = np.zeros((P.shape[1]))
    for i in range(x_a.shape[0]):
        w += preferences[i]*get_p_reached(x_a[i], P) 
    w /= x_a.shape[0]
    S_d = mip_br(P, k_d, w)
    p_defender_reach = get_p_reached(indicator(S_d, n), P) 
    value = 0
    for i in range(x_a.shape[0]):
        value += objective_budget(x_a[i], P, preferences[i]*(1 - p_defender_reach))
    value /= x_a.shape[0]
    return value


def mip_br(P, budget, w, n_samples = 200):
    import gurobipy as gp
    m = gp.Model()
    m.params.OutputFlag = 0
    x = []
    n = P.shape[0]
    for i in range(n):
        x.append(m.addVar(vtype = gp.GRB.BINARY))
    reached_vars = []
    objective = 0
    for i in range(n_samples):
        reached_vars.append([])
        for j in range(P.shape[1]):
            reached_vars[i].append(m.addVar(vtype=gp.GRB.BINARY))
            objective += 1./n_samples * w[j] * reached_vars[i][j]
    m.update()
    for i in range(n_samples):
        P_sampled = np.random.binomial(1, P)
        for j in range(P.shape[1]):
            m.addConstr(reached_vars[i][j] <= gp.quicksum(x[k] for k in np.where(P_sampled[:, j] != 0)[0]))
    m.addConstr(gp.quicksum(x) <= budget)
    m.setObjective(objective, gp.GRB.MAXIMIZE)
    m.optimize()
    return set([i for i in range(n) if x[i].x > 0.5])
    
        
