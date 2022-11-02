#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
# ______________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and
# Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
# rights in this software.
# This software is distributed under the 3-clause BSD License
# ______________________________________________________________________________

#import pprint
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

import pyomo.environ as pyo

# organize data using the data file and user input product and component headers
def _get_data_from_csv(csv_filepath, process_variant_columns, shared_component_columns,
                       feasibility_column, annualized_cost_column, num_shared_component_designs):
    """
    Organize raw data from a csv file into the sets for PFD optimization.
    This is based on specified product headers & component headers passed by the user.

    Args:
       csv_filepath : str
          location of csv file containing the data
       process_variant_columns : list of str
          list of column names corresponding to the variables that define the boundary conditiosn for the 
          process variant
       TODO: finish documentation
            - 'unit headers':           list[(string)]      list of names corresponding to unit headers in csv file
            - 'success header':         string              name corresponding to success header in csv file
            - 'cost header':            string              name corresponding to cost header in csv file
            - 'units manufactured':     list[int]           number of each unit type user wants to allow for manufacturing
       info : dict
          Dictionary containing information to read the input csv
            - 'data location':          string              location of csv file
            - 'installation headers':   list[(string)]      list of names corresponding to installation headers in csv file 
            - 'unit headers':           list[(string)]      list of names corresponding to unit headers in csv file
            - 'success header':         string              name corresponding to success header in csv file
            - 'cost header':            string              name corresponding to cost header in csv file
            - 'units manufactured':     list[int]           number of each unit type user wants to allow for manufacturing

    Returns: 
        dict 
           Returns a dictionary containing organized sets, with following entries
            - 'data':       pandas df           raw data
            - 'K':          list[string]        list of names corresponding to all unit types to be designed commonly
            - 'S_k':        dict{k:[]}          dictionary, indexed by unit type k, which corresponds to a list of designs for that unit
            - 'I':          set(tuple)          set of tuples corresponding to all installation sites
            - 'A_i':        dict{i:[]}          dictionary, indexed by installation i, which corresponds to a list of alternatives for that installation
            - 'Q_a':        dict{a:[]}          dictionary, indexed by alternative a, which corresponds to a list of tuples corresponding to (k,s) in the alt.
            - 'alpha_ia':   dict{(i,a):c}       dictionary, indexed by installation i, alternative a, corresponding to the cost of that alternative at the installation.
            - 'N_k':        dict{k:int}         dictionary, indexed by unit k, corresponding to the number of designs we want for that unit k
    """

    # SPLIT DATA 

    # read data from csv file into panda dictionary
    data = pd.read_csv(csv_filepath)

    # grab data we need, organize into arrays
    installation_details = np.vstack( [ data[p] for p in process_variant_columns ] ).T
    unit_details = np.vstack( [ data[c] for c in shared_component_columns ] ).T
    cost = np.array( data[annualized_cost_column] )
    success = np.array( data[feasibility_column] )
    n_rows = len(success)
    assert n_rows == installation_details.shape[0]
    assert n_rows == unit_details.shape[0]
    assert n_rows == len(cost)

    # get the size options for each shared component
    S_k = { nm:sorted( set( unit_details[:,k] ) ) for k,nm in enumerate(shared_component_columns) }

    # get the set of all possible process variants from the data
    I = map(tuple,installation_details.tolist()) # list of tuples of all combinations in the csv file
    I = sorted(set(I)) # sorted set of unique tuples

    # set of feasible alternatives for each process variant
    A_i = {i:[] for i in I}
    alpha_ia = {}

    # loop through data to store alternatives & costs
    for row in range(n_rows):
        # grab & store alternative data and cost data
        installation_specs = tuple(installation_details[row])
        unit_specs = tuple(unit_details[row])
        ia = installation_specs + unit_specs # concatenate process variant and shared component tuples
        alpha_ia[ ia ] = cost[row]

        # alternative is only stored if success = True
        if success[row] == True:
            A_i[installation_specs].append(unit_specs)

    # Q_a = list of tuples of (shared_component_name, size) for all shared components that are utilized within a particular alternative
    # Each entry in Q_a should be the same length as the number of shared components
    Q_a = dict()
    for r in range(n_rows):
        Q_a[tuple(unit_details[r])] = [ (nm,unit_details[r][k]) for k,nm in enumerate(shared_component_columns) ]

    # specify number of each component type to manufacture
    assert sorted(shared_component_columns) == sorted(num_shared_component_designs.keys())
    N_k = dict(num_shared_component_designs)

    # create dict of sets as return
    sets = {
        'K': list(shared_component_columns),
        'S_k': S_k,
        'I': I,
        'A_i': A_i,
        'Q_a': Q_a,
        'alpha_ia': alpha_ia,
        'N_k': N_k
    }  

    # check that data makes sense
    # TODO: uncomment once check data is cleaned up
    # check_data(data,info,sets)

    return sets

# TODO: clean up check_data
def check_data(data, info, sets):
    """
    Check data & constructed sets.
    Args:
       info : dict
          Dictionary containing information to read the input csv (see :meth:`organize_data_for_product_family_design`)
       sets : dict
          Dictionary containing the sets and parameters to create the model
    Returns:
        -- None.
    """

    # grab data we need, organize into arrays
    installation_details = np.vstack( [ data[p] for p in info['installation headers'] ] ).T
    unit_details = np.vstack( [  data[c] for c in info['unit headers'] ] ).T
    cost = np.array( data[info['cost header']] )
    success = np.array(  data[info['success header']] )


    # (1) check if all costs are different- if not, give warning that p-median may fail
    set_of_costs=set(list(data[info['cost header']]))
    if len(set_of_costs)<len(cost):
        print('\nWarning: not all costs are unique, which means the P-median formulation will potentially fail.\n')
    

    # (2) check that dataset is the right size
    necessary_dataset_size=1
    
    # all possible unit combos
    for k in sets['K']:
        necessary_dataset_size=necessary_dataset_size*len(sets['S_k'][k])
    # mult all possible unit combos by all possible installation combos
    necessary_dataset_size=necessary_dataset_size*len(sets['I'])

    actual_dataset_size=data.shape[0]            
    if actual_dataset_size!=necessary_dataset_size:
        print('\nData Error: the required number of data points, for this process family, should be', necessary_dataset_size)
        print('\tThe actual dataset size is', actual_dataset_size)
        print('The dataset should contain all possible combinations of units and installations.')
        quit()


    # (3) Check that all products have at least ONE feasible alternative
    cannot_design=[]
    for i in sets['I']:
        if len(sets['A_i'][i])==0:
            print('\nWarning: installation', i, 'does not have any feasible alternatives.')
            print('Removing this installation specification from the set, cannot design for this installation.\n')
            cannot_design.append(i)
    
    # for those that we could not design, remove from set i in I
    if len(cannot_design)!=0:
        for i in cannot_design:
            sets['I'].remove(i)
            sets['A_i'].pop(i)


def _build_discretized_formulation_pyomo_model(sets):
    """
    Builds pyomo model for the discretized product family design problem.
    Inputs: 
        -- sets:    dict            dictionary containing organized sets, with following entries
    Returns:
        -- m:       Pyomo model     Pyomo model 
    """

    model = pyo.ConcreteModel()
    model.sets = sets

    # indices for x_ia
    x_ia_indices = [ tuple( (i,a) ) for i in sets['I'] for a in sets['A_i'][i] ]
    model.x_ia = pyo.Var(x_ia_indices, bounds = pyo.Binary)

    # indices for z_ks
    z_ks_indices = [ tuple( (k,s) ) for k in sets['K'] for s in sets['S_k'][k] ]
    model.z_ks = pyo.Var(z_ks_indices, within = pyo.Binary)

    # obj. min. total weighted cost of all installations, i
    model.obj = pyo.Objective( expr = sum( model.x_ia[(i, a)] * sets['alpha_ia'][i + a] \
        for i in sets['I'] for a in sets['A_i'][i]) )

    # only manufacture a certain number of each unit type
    @model.Constraint(sets['K'])
    def max_number_of_units_manufacture(model, k):
        return sum( model.z_ks[k, s] for s in sets['S_k'][k] ) <= sets['N_k'][k]
    
    # only 1 alternative can be selected for each installation
    @model.Constraint(sets['I'])
    def only_one_alternative_selected(model, *args):
        return sum( model.x_ia[args, a] for a in sets['A_i'][args] ) == 1
    
    # need to create new list to iterate through for final constraint
    alt_select_data = []
    for i in sets['I']:
        for a in sets['A_i'][i]:
            for q in sets['Q_a'][a]:
                alt_select_data.append([i, a, q])
    
    # only select alt.'s if all of their individual units are selected for manufacture
    @model.Constraint(alt_select_data)
    def alternative_selectability(model, *args):
        return ( model.x_ia[ tuple((args[0:-2])) ] <= model.z_ks[(args[-2], args[-1])] )

    return model


def solve_product_family_design_pyomo_model(model, show='True', save_results_to=None, solver='glpk'):
    """
    Solves pyomo model for the discretized product family design problem.
    Inputs: 
        -- model:               Pyomo model     Pyomo model
        -- show:                bool            indicator for printing solution to terminal or not
        -- save_results_to      string          location of csv file to save results to
        -- solver:              string          name of solver. 
    Returns:
        -- sol:     dict            dictionary of alternative selected for each installation i
    """
    # solve model
    opt=pyo.SolverFactory(solver)
    results=opt.solve(model, tee=True)

    # if saving results, open file.
    if save_results_to!=None:

        results_file=open(save_results_to, 'w')

        current_time = datetime.datetime.now()
        results_file.write( str('Date: ' + str(current_time.month) + '/' + str(current_time.day) + '/' + str(current_time.year) + '\n') )
        results_file.write( str('Time: ' + str(current_time.hour) + ':' + str(current_time.minute) + ':' + str(current_time.second) + '\n\n'))
        results_file.write( str('Solve Time: ' + str(results.solver.time) + '\n'))
        results_file.write( str('Solver Status: ' + str(results.solver.status) + '\n'))
        results_file.write( str('Termination Condition: ' + str(results.solver.termination_condition) +'\n\n'))
        results_file.write( str('Total Annualized Cost = ' + str(pyo.value(model.obj)) + '\n\n'))
        results_file.write('The assignments of alternatives to installation are:\n')


    # create dict to hold selected alt for each installation i
    sol={i:[] for i in model.sets['I']}

    # loop through each installation & alternative to grab which was selected
    for i in model.sets['I']:
        for a in model.sets['A_i'][i]:

            if pyo.value(model.x_ia[i,a] >= 0.9):
                sol[i] = a

                # display value if indicated
                if show:
                    print('Installation = ', i, ' | Alt. = ', a)
                
                # save results if indicated
                if save_results_to!=None:
                    results_file.write( str('Installation =' + str(i) + ' | Alt. =' + str(a) + '\n') )
    
    return [model,sol]

def build_discretized_mip(csv_filepath, process_variant_columns, shared_component_columns,
                       feasibility_column, annualized_cost_column, num_shared_component_designs):

    """
    TODO: document this from _get_data_from_csv and reference this one for that function
    """
    sets = _get_data_from_csv(csv_filepath, process_variant_columns, shared_component_columns,
                       feasibility_column, annualized_cost_column, num_shared_component_designs)
    model_instance = _build_discretized_formulation_pyomo_model(sets)
    return model_instance

if __name__ == '__main__':

    # elect which test to run
    test_1=True
    test_2=False
    test_3=False
    test_4=False
    test_5=False
    test_6=False

    # DIFFERENT HVAC DATASET SPLITS
    
    # test 1: small refrigeration data (42 data points, 1 unit, 1 product)
    if test_1:
        print('Test 1: Small Example: 42 data points from Refrigeration Dataset')
        print('\t1 shared evaporator with 6 designs.')
        print('\t2 installation details, with 1 outside air temperature and 7 capacities')
        csv_filepath = "tests/data/rfr_data_42pts.csv"
        process_variant_columns = ['Capacity']
        shared_component_columns = ['Evaporator Area']
        feasibility_column = ['Success']
        annualized_cost_column = ['Total Cost']
        num_shared_component_designs = {'Evaporator Area': 2}
        test_results_file='tests/test_1_results.txt'

    # test 2: medium refrigeration data (336 data points, 1 unit, 2 products (8 OAT x 7 CAP))
    if test_2:
        print('Test 2: Medium Example: 336 data points from Refrigeration Dataset')
        csv_filepath = "data\\\hvac_datasets\\hvac_data_336pts.csv"
        process_variant_columns = ['Capacity', 'Outside Air Temperature']
        shared_component_columns = ['Evaporator Area']
        feasibility_column = ['Success']
        annualized_cost_column = ['Total Cost']
        num_shared_component_designs = {'Evaporator Area': 2}
        test_results_file='test_results\\test_2_results.txt'

    
    # test 3: full refrigeration data (78,400 data points, 3 units, 2 products (8 OAT x 7 CAP))
    if test_3:
        print('Test 2: Full (78,400pts) Refrigeration Dataset')
        csv_filepath = "data\\hvac_datasets\\hvac_data.csv"
        process_variant_columns = ['Capacity', 'Outside Air Temperature']
        shared_component_columns = ['Evaporator Area', 'Condenser Area', 'Compressor Design Flow']
        feasibility_column = ['Success']
        annualized_cost_column = ['Total Cost']
        num_shared_component_designs = {'Evaporator Area': 2, 'Condenser Area': 2, 'Compressor Design Flow': 2}
        test_results_file='test_results\\test_3_results.txt'

    # MEA FACILITY DATASET
    
    # test 4: MEA carbon capture facility data (no heat exchanger)
    if test_4:
        print('Test 4: MEA Carbon Capture Facility Dataset')
        csv_filepath = "data\\MEA_carbon_capture_data.csv"
        process_variant_columns = ['Inlet Flow (kg/hr) ', 'Inlet CO2 MassFrac']
        shared_component_columns = ['Absorber Packing Diameter (m)', 'Stripper Packing Diameter (m)']
        feasibility_column = ['All Bounds Met']
        annualized_cost_column = ['Total Cost of Plant']
        num_shared_component_designs = {'Absorber Packing Diameter (m)':3, 'Stripper Packing Diameter (m)': 2}
        test_results_file='test_results\\test_4_results.txt'

    # TESTING SOME WONKY CASES

    # test 5: some installations have no alternatives.
    if test_5:
        print('Test 5: An installation does not have any feasible combinations.')
        print('\tUsing: 42 data points from Refrigeration Dataset, where CAP=200, OAT=35 has no alternatives.')
        print('\t1 shared evaporator with 6 designs.')
        print('\t2 installation details, with 1 outside air temperature and 7 capacities.\n')
        csv_filepath = "data\\wonky_datasets\\hvac_data_42pts_no_alternatives.csv"
        process_variant_columns = ['Capacity', 'Outside Air Temperature']
        shared_component_columns = ['Evaporator Area']
        feasibility_column = ['Success']
        annualized_cost_column = ['Total Cost']
        num_shared_component_designs = {'Evaporator Area': 2}
        test_results_file='test_results\\test_5_results.txt'

    # test 6: incorrect dataset
    if test_6:
        print('Test 6: Dataset is incorrect in size.')
        print('\tUsing: 42 data points from Refrigeration Dataset, with a couple of data points removed.')
        csv_filepath = "data\\wonky_datasets\\hvac_data_42pts_incorrect_dataset.csv"
        process_variant_columns = ['Capacity', 'Outside Air Temperature']
        shared_component_columns = ['Evaporator Area']
        feasibility_column = ['Success']
        annualized_cost_column = ['Total Cost']
        num_shared_component_designs = {'Evaporator Area': 2}
        test_results_file='test_results\\test_6_results.txt'
    
    model_instance = build_discretized_mip(csv_filepath=csv_filepath,
                                           process_variant_columns=process_variant_columns,
                                           shared_component_columns=shared_component_columns,
                                           feasibility_column=feasibility_column,
                                           annualized_cost_column=annualized_cost_column,
                                           num_shared_component_designs=num_shared_component_designs)
    
    solved_model, sol = solve_product_family_design_pyomo_model(model_instance, 
                                                                save_results_to=test_results_file, 
                                                                solver='gurobi')
