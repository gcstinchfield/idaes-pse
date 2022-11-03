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
import sys
import os
import pprint
sys.path.append(os.path.abspath("."))  # current folder is ~/tests
pprint.pprint(sys.path)
import pytest
import pyomo.environ as pyo

glpk_available = pyo.SolverFactory('glpk').available()

from idaes.apps.pfamos.family_design import build_discretized_mip, create_results_summary

@pytest.mark.skipif(not glpk_available, reason="The 'glpk' solver is not available")
@pytest.mark.unit
def test_small_42():
    """
    Tests building & solving discretized formulation for a small refrigeration case
        42 data points, 1 product (7 capacities) and 1 unit type (6 evap sizes)
    """
    csv_filepath = "./data/rfr_data_42pts.csv"
    process_variant_columns = ['Capacity']
    shared_component_columns = ['Evaporator Area']
    feasibility_column = ['Success']
    annualized_cost_column = ['Total Cost']
    num_shared_component_designs = {'Evaporator Area': 2}

    # build pyomo model
    model_instance = build_discretized_mip(csv_filepath=csv_filepath,
                                           process_variant_columns=process_variant_columns,
                                           shared_component_columns=shared_component_columns,
                                           feasibility_column=feasibility_column,
                                           annualized_cost_column=annualized_cost_column,
                                           num_shared_component_designs=num_shared_component_designs)

    # solve model instance
    opt=pyo.SolverFactory('glpk')
    opt.solve(model_instance, tee=True)

    # output organized results
    sol=create_results_summary(model_instance, process_variant_columns)

    assert model_instance is not None
    assert pyo.value(model_instance.obj)==pytest.approx(549630)

    assert pyo.value(model_instance.x_ia[((80),(50))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((100),(50))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((120),(50))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((140),(50))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((160),(80))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((180),(80))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((200),(80))])==pytest.approx(1)

def test_medium_336():
    """
    Tests building & solving discretized formulation for a medium refrigeration case
        336 data points, 2 product2 (7 capacities x 8 outside aire temperares) 
        and 1 unit type (6 evap sizes)
    """
    csv_filepath = "./data/rfr_data_336pts.csv"
    process_variant_columns = ['Capacity', 'Outside Air Temperature']
    shared_component_columns = ['Evaporator Area', 'Condenser Area', 'Compressor Design Flow']
    feasibility_column = ['Success']
    annualized_cost_column = ['Total Cost']
    num_shared_component_designs = {'Evaporator Area': 2, 'Condenser Area': 2, 'Compressor Design Flow': 2}

    # build pyomo model
    model_instance = build_discretized_mip(csv_filepath=csv_filepath,
                                           process_variant_columns=process_variant_columns,
                                           shared_component_columns=shared_component_columns,
                                           feasibility_column=feasibility_column,
                                           annualized_cost_column=annualized_cost_column,
                                           num_shared_component_designs=num_shared_component_designs)
    
    # solve model instance
    opt=pyo.SolverFactory('glpk')
    opt.solve(model_instance, tee=True)

    # output organized results
    sol=create_results_summary(model_instance, process_variant_columns)

    assert model_instance is not None
    assert pyo.value(model_instance.obj)==pytest.approx(4397045)

    assert pyo.value(model_instance.x_ia[((80,28),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((80,29),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((80,30),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((80,31),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((80,32),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((80,33),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((80,34),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((80,35),(50,25,105))])==pytest.approx(1)

    assert pyo.value(model_instance.x_ia[((100,28),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((100,29),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((100,30),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((100,31),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((100,32),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((100,33),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((100,34),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((100,35),(50,25,105))])==pytest.approx(1)

    assert pyo.value(model_instance.x_ia[((120,28),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((120,29),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((120,30),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((120,31),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((120,32),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((120,33),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((120,34),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((120,35),(50,25,105))])==pytest.approx(1)

    assert pyo.value(model_instance.x_ia[((140,28),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((140,29),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((140,30),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((140,31),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((140,32),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((140,33),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((140,34),(50,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((140,35),(50,25,105))])==pytest.approx(1)

    assert pyo.value(model_instance.x_ia[((160,28),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((160,29),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((160,30),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((160,31),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((160,32),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((160,33),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((160,34),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((160,35),(80,25,105))])==pytest.approx(1)

    assert pyo.value(model_instance.x_ia[((180,28),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((180,29),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((180,30),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((180,31),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((180,32),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((180,33),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((180,34),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((180,35),(80,25,105))])==pytest.approx(1)

    assert pyo.value(model_instance.x_ia[((200,28),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((200,29),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((200,30),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((200,31),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((200,32),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((200,33),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((200,34),(80,25,105))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((200,35),(80,25,105))])==pytest.approx(1)


def test_missing_alternatives_dataset():
    """
    Tests building & solving discretized formulation when there are no feasible 
    combinations for one of the boundary condition combinations
        42 data points, 1 product (7 capacities) and 1 unit type (6 evap sizes)
        Removed all alternatives for (CAP=200) 
    """
    csv_filepath = "./data/rfr_data_no_alternatives.csv"
    process_variant_columns = ['Capacity', 'Outside Air Temperature']
    shared_component_columns = ['Evaporator Area']
    feasibility_column = ['Success']
    annualized_cost_column = ['Total Cost']
    num_shared_component_designs = {'Evaporator Area': 2}

    # build pyomo model
    model_instance = build_discretized_mip(csv_filepath=csv_filepath,
                                           process_variant_columns=process_variant_columns,
                                           shared_component_columns=shared_component_columns,
                                           feasibility_column=feasibility_column,
                                           annualized_cost_column=annualized_cost_column,
                                           num_shared_component_designs=num_shared_component_designs)

    # solve model instance
    opt=pyo.SolverFactory('glpk')
    opt.solve(model_instance, tee=True)

    # output organized results
    sol=create_results_summary(model_instance, process_variant_columns)

    assert model_instance is not None
    assert pyo.value(model_instance.obj)==pytest.approx(445634)

    assert pyo.value(model_instance.x_ia[((80),(50))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((100),(50))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((120),(50))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((140),(50))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((160),(80))])==pytest.approx(1)
    assert pyo.value(model_instance.x_ia[((180),(80))])==pytest.approx(1)
    
    # check that the installation CAP=200 is not included
    for i in model_instance.sets['I']:
        assert i!=200