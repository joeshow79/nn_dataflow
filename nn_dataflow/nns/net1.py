""" $lic$                                                                                                                                                                                
Copyright (C) 2016-2019 by The Board of Trustees of Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, PoolingLayer

'''
net1
'''

NN = Network('net1')
# Linear.
NN.set_input_layer(InputLayer(10, 1)) 
NN.add('0', FCLayer(10, 20))
NN.add('1', FCLayer(20, 30))
NN.add('1p', PoolingLayer(30, 1, 1)) 
NN.add('2', FCLayer(30, 40))
NN.add('3', FCLayer(40, 50))
