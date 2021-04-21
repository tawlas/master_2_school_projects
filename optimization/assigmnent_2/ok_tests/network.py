test = {
  'name': 'network',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> np.random.seed(10);
          >>> 
          >>> n_input  = 5;
          >>> n_hidden = 4;
          >>> n_output = 3;
          >>> params = initialize_parameters(n_input, n_hidden, n_output);
          >>> 
          >>> X = np.random.rand(10, n_input);
          >>> 
          >>> twolayer_network(X, params)
          array([[0.29287282, 0.47466353, 0.23246365],
                 [0.43291093, 0.29935568, 0.26773339],
                 [0.28229147, 0.37932587, 0.33838266],
                 [0.230469  , 0.59355683, 0.17597417],
                 [0.34380574, 0.21934467, 0.43684959],
                 [0.24427434, 0.2042511 , 0.55147457],
                 [0.45563539, 0.29954769, 0.24481692],
                 [0.41075115, 0.24857582, 0.34067303],
                 [0.3617473 , 0.26180309, 0.37644961],
                 [0.30082965, 0.23956349, 0.45960686]])
          """,
          'hidden': False,
          'locked': False
        }
      ],
      'scored': True,
      'setup': '',
      'teardown': '',
      'type': 'doctest'
    }
  ]
}
