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
            list of column names corresponding to the variables that define the boundary conditions for the 
            process variants
        shared_component_columns : list of str
            list of column names coresponding to the components shared across process variants
        feasibility_column: str
            column name corresponding to True/False feasibility data
        annualized_cost_column : str
            column name corresponding to the total annualized cost of each boundary condition & unit
        num_shared_component_designs : dict
            dictionary, keys corresponding to each component in shared_component_columns)
            each entry corresponds to an integer value for how of that unit should be designed.
    Returns: 
        sets : dict 
            A dictionary containing organized sets from csv data, with the following entries
                K : list        
                    list of (string) names corresponding to all unit types to be designed commonly
                S_k : dict          
                    dictionary, indexed shared component, corresponding to a list of designs for that unit
                I : set
                    set of tuples corresponding to all possible combination of boundary conditions
                A_i : dict       
                    dictionary, indexed by set I, corresponding to a list of alternatives (tuples of unit designs) 
                    for that installation
                Q_a : dict          
                    dictionary, indexed by alternative, corresponding to a list of tuples corresponding to (k,s) 
                    in each alternative
                alpha_ia : dict       
                    dictionary, indexed by installation i & alternative a, corresponding to the cost
                N_k : dict         
                    dictionary, indexed by common unit k, corresponding to the number of designs to be selected for that unit
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

    # make sure dimensions of data are correct
    if installation_details.shape[0]!=n_rows or unit_details.shape[0]!=n_rows or len(cost)!=n_rows:
        print('Error: Not all columns contain the same number of data points.')
        print('Check dataset.')
        quit()

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
    check_data(data,shared_component_columns,sets)

    return sets

# TODO: clean up check_data
def check_data(data, cost, sets):
    """
    Check data & constructed sets to ensure P-Median viability & alternatives availability.

    Args:
        data : pandas df
            Data from csv file
        cost : numpy array
            Array containing costing data for each data point
        sets : dict
            Dictionary containing the sets and parameters to create the model 
            (see :meth:`organize_data_for_product_family_design`)
    Returns:
    """

    # (1) check if all costs are different- if not, give warning that P-median may fail
    # find set of all cost data (i.e. eliminate duplicates)
    set_of_costs=set(list(data[sets['K']])) 

    # if the length(set of cost) is not the same as the length(list of costs) there are duplicates
    if len(set_of_costs)<len(cost): 
        print('\nWarning: not all costs are unique, which means the P-median formulation will potentially fail.\n')
    

    # (2) Check that all products have at least ONE feasible alternative
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

    Args:
        sets : dict
            Dictionary containing the sets and parameters to create the model 
            (see :meth:`organize_data_for_product_family_design`)
    Returns:
        model : Pyomo model
            Instance of a Pyomo model formulated with the discretize process family design formulation
    """

    model = pyo.ConcreteModel()
    model.sets = sets

    # indices for x_ia: all possible combinations of installation i boundary conditions & alternative
    x_ia_indices = [ tuple( (i,a) ) for i in sets['I'] for a in sets['A_i'][i] ]
    model.x_ia = pyo.Var(x_ia_indices, bounds = pyo.Binary) # 0 <= x_ia <= 1

    # indices for z_ks: all possible possible designs for shared units
    z_ks_indices = [ tuple( (k,s) ) for k in sets['K'] for s in sets['S_k'][k] ]
    model.z_ks = pyo.Var(z_ks_indices, within = pyo.Binary) # z_ks = {0,1}

    # obj. = min. total weighted cost of all installations, i
    model.obj = pyo.Objective( expr = sum( model.x_ia[(i, a)] * sets['alpha_ia'][i + a] \
        for i in sets['I'] for a in sets['A_i'][i]) )

    # only manufacture a certain number of each unit type
    @model.Constraint(sets['K'])
    def max_number_of_units_to_manufacture(model, k):
        return sum( model.z_ks[k, s] for s in sets['S_k'][k] ) <= sets['N_k'][k]
    
    # only 1 alternative can be selected for each installation
    @model.Constraint(sets['I'])
    def only_select_one_alternative(model, *args):
        i=args # arguments represent tuple entries for installation i
        return sum( model.x_ia[i, a] for a in sets['A_i'][i] ) == 1
    
    # create new list of all installation i, alternative a, Q_a[a] for final constraint
    altnernative_selectability_data = []
    for i in sets['I']:
        for a in sets['A_i'][i]:
            for q in sets['Q_a'][a]:
                altnernative_selectability_data.append([i, a, q])
    
    # only select alt.'s if all of their individual units are selected for manufacture
    @model.Constraint(altnernative_selectability_data)
    def alternative_selectability(model, *args):
        ia=tuple((args[0:-2])) # all elements, except last two, hold (i,a) data
        k=args[-2] # second to last element holds shared unit name
        s=args[-1] # last element holds shared unit design
        return ( model.x_ia[ ia ] <= model.z_ks[( k,s )] )

    return model


def build_discretized_mip(csv_filepath, process_variant_columns, shared_component_columns,
                       feasibility_column, annualized_cost_column, num_shared_component_designs):
    """
    Builds pyomo model for the discretized product family design problem.

    Args:
        csv_filepath : str
            location of csv file containing the data
        process_variant_columns : list of str
            list of column names corresponding to the variables that define the boundary conditiosn for the 
            process variants
        shared_component_columns : list of str
            list of column names coresponding to the components shared across process variants
        feasibility_column: str
            column name corresponding to True/False feasibility data
        annualized_cost_column : str
            column name corresponding to the total annualized cost of each boundary condition & unit
        num_shared_component_designs : dict
            dictionary, keys corresponding to each component in shared_component_columns)
            each entry corresponds to an integer value for how of that unit should be designed.
    Returns:
        model_instance : Pyomo model
            Instance of a Pyomo model with the discretize process family design formulation
    """
    # get sets and check data
    sets = _get_data_from_csv(csv_filepath, process_variant_columns, shared_component_columns,
                       feasibility_column, annualized_cost_column, num_shared_component_designs)

    # build pyomo model
    model_instance = _build_discretized_formulation_pyomo_model(sets)

    return model_instance


def create_results_summary(model, process_variant_columns, show=True, csv_pathstring=None):
    """
    Prints all installations with their assigned alternatives 
    Optionally creates a csv file of results with printed information.

    Args:
        model : Pyomo model
            Solved instance of a process family design Pyomo model
        process_variant_columns : list
            list of column names corresponding to the variables that define the boundary conditiosn for the 
            process variants
        show : boolean, optional
            Show is True to output organized results in terminal, otherwise False
            by default True
        create_csv : str, optional
            Path location (string) to where to write the csv file 
            by default None
    Returns:
        sol : dict
            Dictionary, indexed by installation i, corresponding to the assigned alternative a
    """

    # if saving results, open file.
    if csv_pathstring!=None:
        results_file=open(csv_pathstring, 'w')
        current_time = datetime.datetime.now()
        results_file.write( str('Date: ' + str(current_time.month) + '/' + str(current_time.day) + '/' + str(current_time.year) + '\n') )
        results_file.write( str('Time: ' + str(current_time.hour) + ':' + str(current_time.minute) + ':' + str(current_time.second) + '\n\n'))
        results_file.write( str('Total Annualized Cost = ' + str(pyo.value(model.obj)) + '\n\n'))
        results_file.write('The assignments of alternatives to installation are:\n')

     # create dict to hold selected alt for each installation i
    sol={i:[] for i in model.sets['I']}

    # loop through each installation & alternative to grab which was selected
    for i in model.sets['I']:
        for a in model.sets['A_i'][i]:

            # if binary indicated=1, that alternative was selected for this installation
            if pyo.value(model.x_ia[i,a] >= 0.9):

                # add to dictionary
                sol[i] = a

                # display value if indicated
                if show:
                    print('\nInstallation:')
                    for i_index, i_name in enumerate(process_variant_columns):
                        print('\t', i_name, '=', i[i_index])
                    print('Alternative Selected:')
                    for a_index, a_name in enumerate(model.sets['K']):
                        print('\t', a_name, '=', a[a_index])
                
                # save results if indicated
                if csv_pathstring!=None:
                    results_file.write( '\nInstallation :\n' )
                    for i_index, i_name in enumerate(process_variant_columns):
                        results_file.write( str('\t' + str(i_name) + '=' + str(i[i_index]) + '\n') )
                    results_file.write( 'Alternative Selected:\n' )
                    for a_index, a_name in enumerate(model.sets['K']):
                        results_file.write( str('\t' + str(a_name) + '=' + str(a[a_index]) + '\n') )
