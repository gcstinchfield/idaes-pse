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
sys.path.append(os.path.abspath(".."))  # current folder is ~/tests

import pytest
from pyomo.environ import SolverFactory

glpk_available = SolverFactory('glpk').available()

from idaes.apps.pfamos.family_design import build_discretized_mip, solve_product_family_design_pyomo_model

@pytest.mark.skipif(not glpk_available, reason="The 'glpk' solver is not available")
@pytest.mark.unit
def test_small_42():
    # Test 1: Small Example: 42 data points from Refrigeration Dataset
    # 1 shared evaporator with 6 designs.
    # 2 installation details, with 1 outside air temperature and 7 capacities
    csv_filepath = "./data/rfr_data_42pts.csv"
    process_variant_columns = ['Capacity']
    shared_component_columns = ['Evaporator Area']
    feasibility_column = ['Success']
    annualized_cost_column = ['Total Cost']
    num_shared_component_designs = {'Evaporator Area': 2}

    model_instance = build_discretized_mip(csv_filepath=csv_filepath,
                                           process_variant_columns=process_variant_columns,
                                           shared_component_columns=shared_component_columns,
                                           feasibility_column=feasibility_column,
                                           annualized_cost_column=annualized_cost_column,
                                           num_shared_component_designs=num_shared_component_designs)

    solved_model, sol = solve_product_family_design_pyomo_model(model_instance, 
                                                                solver='glpk')

    assert solved_model is not None
    assert solved_model is None
