name: KSR-LDA
network = neural_xc.build_sliding_net(
        window_size=1,
        num_filters_list=[16, 16, 16],
        activation='swish')

# additional notes
trained on [(2, 2), (3, 3)]
validated on [(1,1)]
