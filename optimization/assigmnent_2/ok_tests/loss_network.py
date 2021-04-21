test = {
  'name': 'loss_network',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> np.random.seed(42);
          >>> 
          >>> X, c = generate_data();
          >>> Y = to_categorical(c);
          >>> 
          >>> n_input  = X.shape[1];
          >>> n_output = Y.shape[1];
          >>> params = initialize_parameters(n_input, 10, n_output);
          >>> 
          >>> optimization_problem(params, X, Y)
          1.6300502421423746
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
