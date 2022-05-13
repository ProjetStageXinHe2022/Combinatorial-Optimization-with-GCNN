import json
def dumpRowFeatures(filename,row_features):
    features = {}
    num = 0
    for node in row_features:
        node_features = {"bias" : node[0],
                    "objective_cosine_similarity" : node[1],
                   "is_tight" : node[2],
                   "dual_solution_value" : node[3],
                   "scaled_age" : node[4]
                    }
        features[num] = node_features
        num+=1
    data = json.dumps(features)
    file = open(filename,'w')
    file.write(data)
    file.close()

def dumpVariableFeatures(filename,variable_features):
    features = {}
    num = 0
    for node in variable_features:
        node_features = {"objective" : node[0],
                "is_type_binary" : node[1],
               "is_type_integer" : node[2],
               "is_type_implicit_integer" : node[3],
               "is_type_continuous" : node[4],
                "has_lower_bound" : node[5],
                "has_upper_bound" : node[6],
               "normed_reduced_cost" : node[7],
               "solution_value" : node[8],
               "solution_frac" : node[9],
                "is_solution_at_lower_bound" : node[10],
                "is_solution_at_upper_bound" : node[11],
               "scaled_age" : node[12],
               "incumbent_value" : node[13],
               "average_incumbent_value" : node[14],
                "is_basis_lower" : node[15],
                "is_basis_basic" : node[16],
               "is_basis_upper" : node[17],
               "is_basis_zero" : node[18]
                }
        features[num] = node_features
        num+=1
    data = json.dumps(features)
    file = open(filename,'w')
    file.write(data)
    file.close()

def dumpSolution_Ecole(filename,pyscip):
    bestsol = pyscip.getBestSol()
    solutions = {
        "Best_Solution":bestsol.__str__(),
        "Solution_Pool":pyscip.getSols().__str__()
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
    

# dumpRowFeatures("features/row.json",obs.row_features)
# dumpVariableFeatures("features/v.json",obs.variable_features)
# model = gb.read(problme_name+".lp")

# model.optimize()
# dumpSolution_Gurobi("test_dump_gurobi.json",model)
