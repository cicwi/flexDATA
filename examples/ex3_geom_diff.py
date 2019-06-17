#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example shows how to compare two geometries.
"""
from flexdata import data
from copy import deepcopy

path = '/export/scratch2/adriaan/eierwekker/'

# ------------------------------------------------------------------------------
# Compare just the values in two geometries
geom1 = data.read_flexraylog(path)
geom2 = deepcopy(geom1)
geom2['src2obj'] = 10

corrections = data.geom_diff(geom1, geom2)
print("Corrections: " + str(corrections))
# outputs "Corrections: {'src2obj': -489...}"

# ------------------------------------------------------------------------------
geom1 = data.read_flexraylog(path)
geom2 = deepcopy(geom1).to_dictionary()
geom2['new_key'] = 'new_value'

all_changes = data.geom_diff(geom1, geom2, full_diff=True)
print("Additions: " + str(all_changes['added']))
# outputs "Additions: {'new_key'}"
