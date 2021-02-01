name: KSR-LSDA
network = neural_xc.build_sliding_net(
      window_size=1,
      num_filters_list=[16, 16, 16],
      activation='swish')

Trained on: [(1, 1), (2, 2)] 
Validated on: [(4, 3)] 
