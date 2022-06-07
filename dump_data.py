import json
import pickle as pkl
import numpy as np
import ast
def dumpEdgeFeatures(filename,edge_features,original_indice):
	l = len(original_indice)
	dict_indice = np.zeros((l))
	for i in range(l):
		dict_indice[int(original_indice[i])] = i

	for i in range(len(edge_features.indices[1])):
		edge_features.indices[1][i] = dict_indice[edge_features.indices[1][i]]
		
	features = {"values":edge_features.values.tolist(),
			    "indices":edge_features.indices.tolist()
			   }
	data = json.dumps(features)
	file = open(filename,'w')
	file.write(data)
	file.close() 
 
def dumpRowFeatures(filename,row_features):
    features = {
        "names":["bias","objective_cosine_similarity","is_tight","dual_solution_value","scaled_age"],
        "values":row_features.tolist()
    }
    data = json.dumps(features)
    file = open(filename,'w')
    file.write(data)
    file.close()

def dumpVariableFeatures(filename,variable_features):
    original_indice = variable_features[:,-1]
    l = len(original_indice)
    dict_indice = np.zeros((l))
    for i in range(l):
        dict_indice[int(original_indice[i])] = i
    for i in range(l):
        variable_features[i][-1] = dict_indice[i]
    value = variable_features[variable_features[:,-1].argsort()]
    features = {
        "names":["objective","is_type_binary","is_type_integer","is_type_implicit_integer",\
                 "is_type_continuous","has_lower_bound","has_upper_bound","normed_reduced_cost",\
                 "solution_value","solution_frac","is_solution_at_lower_bound","is_solution_at_upper_bound",\
                 "scaled_age","incumbent_value","average_incumbent_value","is_basis_lower","is_basis_basic",\
                 "is_basis_upper","is_basis_zero"],
        "values":value[:,:-1].tolist()
    }
    data = json.dumps(features)
    file = open(filename,'w')
    file.write(data)
    file.close()

def dumpSolution_Ecole(filename,pyscip):
    bestsol = pyscip.getBestSol()
    solutionPool = []
    for sol in pyscip.getSols():
        solutionPool.append(np.around(list(ast.literal_eval(sol.__repr__()).values()),0).astype(int).tolist())
    solutions = {
        "Best_Solution":np.around(list(ast.literal_eval(bestsol.__repr__()).values()),0).astype(int).tolist(),
        "Solution_Pool":solutionPool
    }
    data = json.dumps(solutions)
    file = open(filename,'w')
    file.write(data)
    file.close()

def dumpSolution_Gurobi(filename,solver,nbSolMax = 10,poolGap=0.1):
    # Limit how many solutions to collect
    model.setParam(GRB.Param.PoolSolutions, nbSolMax)

    # Limit the search space by setting a gap for the worst possible solution
    # that will be accepted
    model.setParam(GRB.Param.PoolGap, poolGap)

    # do a systematic search for the k- best solutions
    model.setParam(GRB.Param.PoolSearchMode, 2)

    model.optimize()
    status = model.Status
    if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        print('The model cannot be solved because it is infeasible or unbounded')
        sys.exit(1)
    if status != GRB.OPTIMAL:
        print('Optimization was stopped with status ' + str(status))
        sys.exit(1)
#the first solution is the best  
#     #get the beat solution
#     bestSol = {}
#     for var in model.getVars():
#         bestSol[var.varName] = var.X
        

    #get the solution pool
    poolSol = {}
    nSolution = model.SolCount
    for e in range (nSolution):
        sol = {}
        model.setParam(GRB.Param.SolutionNumber, e)
        for var in model.getVars():
            sol[var.varName] = var.X
        poolSol[model.PoolObjVal]=sol
    
    solutions = {
        "Nombre_Solutions":nSolution,
#         "Best_Solution":bestSol,
        "Solution_Pool":poolSol
    }

    data = json.dumps(solutions)
    file = open(filename,'w')
    file.write(data)
    file.close()
