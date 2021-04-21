test = {
  'name': 'cross_entropy',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> np.random.seed(10);
          >>> 
          >>> n_samples = 10;
          >>> n_output  = 5;
          >>> 
          >>> labels = np.random.randint(n_output, size=n_samples);
          >>> Y_True = to_categorical(labels);
          >>> Y      = np.random.rand(n_samples, n_output);
          >>> 
          >>> cross_entropy(Y, Y_True)
          1.2871054749330935
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
