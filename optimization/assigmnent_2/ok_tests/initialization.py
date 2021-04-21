test = {
  'name': 'initialization',
  'points': 4,
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
          >>> W1, b1, W2, b2 = initialize_parameters(n_input, n_hidden, n_output);
          >>> 
          >>> W1
          array([[ 0.84216925,  0.45238214, -0.97739696, -0.00530241],
                 [ 0.39296737, -0.4554221 ,  0.16792427,  0.06865212],
                 [ 0.00271414, -0.11042687,  0.27386981,  0.76086764],
                 [-0.61036112,  0.65033763,  0.14459839,  0.28152975],
                 [-0.71885036,  0.08546807,  0.93890364, -0.68292857]])
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> np.random.seed(10);
          >>> 
          >>> n_input  = 5;
          >>> n_hidden = 4;
          >>> n_output = 3;
          >>> W1, b1, W2, b2 = initialize_parameters(n_input, n_hidden, n_output);
          >>> 
          >>> b1
          array([[0., 0., 0., 0.]])
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> np.random.seed(10);
          >>> 
          >>> n_input  = 5;
          >>> n_hidden = 4;
          >>> n_output = 3;
          >>> W1, b1, W2, b2 = initialize_parameters(n_input, n_hidden, n_output);
          >>> 
          >>> W2
          array([[-1.39846508, -1.23275037,  0.18814002],
                 [ 1.68642657,  0.79456971,  1.18272251],
                 [ 0.07010908,  0.98853272, -0.19180129],
                 [ 0.43360084, -0.1890218 , -0.38842013]])
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> np.random.seed(10);
          >>> 
          >>> n_input  = 5;
          >>> n_hidden = 4;
          >>> n_output = 3;
          >>> W1, b1, W2, b2 = initialize_parameters(n_input, n_hidden, n_output);
          >>> 
          >>> b2
          array([[0., 0., 0.]])
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
