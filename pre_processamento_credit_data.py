# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 22:09:58 2019

@author: ronal
"""

import pandas as pd
base = pd.read_csv('credit_data.csv')
base.describe()