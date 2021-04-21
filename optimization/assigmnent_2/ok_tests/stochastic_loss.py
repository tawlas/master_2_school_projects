test = {
  'name': 'stochastic_loss',
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
          >>> cost_fun = StochasticOptimizationProblem(X, Y, batch_size=60);
          >>> 
          >>> np.round([cost_fun(params,i) for i in range(4)], decimals=4)
          array([1.7891, 1.5911, 1.3897, 1.5964])
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
