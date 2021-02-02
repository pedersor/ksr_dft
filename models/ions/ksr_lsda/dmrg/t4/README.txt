name: KSR-LSDA
network = neural_xc.build_sliding_net(
      window_size=1,
      num_filters_list=[16, 16, 16],
      activation='swish')

seed: 1 
Trained on: [(1, 1), (2, 2), (3, 3), (4, 1)] 
Validated on: [(4, 3)] 
optimal ckpt path: ../models/ions/ksr_lsda/dmrg/t4/ckpt-00280 
