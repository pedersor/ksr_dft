name: KSR-LDA
network = neural_xc.build_sliding_net(
    window_size=1,
    num_filters_list=[16, 16, 16],
    activation='swish')

Trained on: [(2, 2), (3, 3)] 
Validated on: [(1, 1)] 

Trained on: [(2, 2), (3, 3)] 
Validated on: [(1, 1)] 
