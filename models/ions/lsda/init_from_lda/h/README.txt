name: KSR-LDA
network = neural_xc.build_sliding_net(
    window_size=1,
    num_filters_list=[16, 16, 16],
    activation='swish')

initial_params_file='lda_optimal_ckpt.pkl'

Trained on: [(1, 1)] 
Validated on: [(1, 1)] 
