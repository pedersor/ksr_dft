name: KSR-GGA
network = neural_xc.build_sliding_net(
    window_size=2,
    num_filters_list=[16, 16, 16],
    activation='swish')

Trained on: [(1, 1), (2, 2), (3, 3), (4, 4)] 
Validated on: [(4, 1)] 

Trained on: [(1, 1), (2, 2), (3, 3), (4, 4)] 
Validated on: None 
