test = {
  'name': 'dense_layer',
  'points': 3,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> np.random.seed(1);
          >>> X = np.random.randn(5,2);
          >>> W = np.random.randn(2,3);
          >>> b = np.random.randn(1,3);
          >>> 
          >>> dense_layer(X, W, b)
          array([[ 2.43748776, -4.91782916,  0.19136239],
                 [-0.53259405, -1.00624933,  1.39265422],
                 [ 1.97680712, -5.27013415,  2.29463375],
                 [ 2.67101975, -5.33544929,  0.31690124],
                 [ 0.38981317, -1.81785236,  0.21363035]])
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> np.random.seed(1);
          >>> X = np.random.randn(5,2);
          >>> W = np.random.randn(2,3);
          >>> b = np.random.randn(1,3);
          >>> 
          >>> dense_layer(X, W, b, 'relu')
          array([[2.43748776, 0.        , 0.19136239],
                 [0.        , 0.        , 1.39265422],
                 [1.97680712, 0.        , 2.29463375],
                 [2.67101975, 0.        , 0.31690124],
                 [0.38981317, 0.        , 0.21363035]])
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> np.random.seed(1);
          >>> X = np.random.randn(5,2);
          >>> W = np.random.randn(2,3);
          >>> b = np.random.randn(1,3);
          >>> 
          >>> dense_layer(X, W, b, 'softmax')
          array([[9.03793377e-01, 5.77691008e-04, 9.56289317e-02],
                 [1.17930471e-01, 7.34378846e-02, 8.08631645e-01],
                 [4.21079161e-01, 2.99955719e-04, 5.78620883e-01],
                 [9.12983130e-01, 3.04296832e-04, 8.67125734e-02],
                 [5.13235911e-01, 5.64339022e-02, 4.30330187e-01]])
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
