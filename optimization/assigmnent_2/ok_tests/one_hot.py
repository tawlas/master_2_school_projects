test = {
  'name': 'one_hot',
  'points': 2,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> labels = np.array([1, 1, 2, 1, 3, 6, 2, 4]);
          >>> 
          >>> to_categorical(labels)
          array([[0., 1., 0., 0., 0., 0., 0.],
                 [0., 1., 0., 0., 0., 0., 0.],
                 [0., 0., 1., 0., 0., 0., 0.],
                 [0., 1., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 1., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 1.],
                 [0., 0., 1., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 1., 0., 0.]])
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> labels = np.array([1, 2, 1, 2, 4]);
          >>> 
          >>> to_categorical(labels)
          array([[0., 1., 0., 0., 0.],
                 [0., 0., 1., 0., 0.],
                 [0., 1., 0., 0., 0.],
                 [0., 0., 1., 0., 0.],
                 [0., 0., 0., 0., 1.]])
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
