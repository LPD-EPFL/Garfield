def pairwise(data):
  """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
  Args:
    data Indexable (including ability to query length) containing the elements
  Returns:
    Generator over the pairs of the elements of 'data'
  """
  n = len(data)
  for i in range(n - 1):
    for j in range(i + 1, n):
      yield (data[i], data[j])