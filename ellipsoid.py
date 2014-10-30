import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def get_2d_ellipse_parameters(A):
  l, v = np.linalg.eig(A)
  width = 2.0 * math.sqrt(l[0])
  height = 2.0 * math.sqrt(l[1])
  angle = (math.atan2(v[1,0], v[0,0]) + math.pi) * 360.0 / (2.0 * math.pi)
  return width, height, angle

def plot_ellipse(E):
  fig = plt.figure()
  ax = fig.add_subplot(111, aspect='equal')
  for e in E:
    width, height, angle = get_2d_ellipse_parameters(e[1])
    e2 = Ellipse(e[0], width, height, angle, fill=False, color=np.random.rand(3))
    ax.add_artist(e2)
  ax.set_xlim(-10, 10)
  ax.set_ylim(-10, 10)
  return fig

def vol(A):
  dim = A.shape[0]
  s_vol = math.pow(math.pi, dim / 2.0) / math.gamma(dim / 2.0 + 1.0)
  return s_vol * math.sqrt(np.linalg.det(A))

def ellipsoid_method(E, oracle, L):
  hyperplanes = []
  d = E.shape[0]
  a = np.zeros((d, 1))
  ellipsoids = [(a, E)]

  while vol(E) > L:
    res, val = oracle(a)
    if res:
      return True, val, ellipsoids, hyperplanes
    else:
      b = np.dot(E, val) / math.sqrt(np.dot(val.transpose(), np.dot(E, val)))
      a = a - (1.0 / (d + 1)) * b
      E = (d * d / (d * d - 1.0)) * (E - (2.0 / (d + 1)) * np.dot(b, b.transpose()))
      ellipsoids.append((a, E))
      hyperplanes.append(val)
  
  return False, E, ellipsoids, hyperplanes

def rectangle_oracle(xmin, xmax):
  d = xmin.shape[0]
  def oracle(x):
    hyperplane = np.zeros((d, 1))
    for i in range(d):
      if x[i] < xmin[i]:
        hyperplane[i] = -1.0
        return False, hyperplane
    for i in range(d):
      if x[i] > xmax[i]:
        hyperplane[i] = 1.0
        return False, hyperplane
    return True, x
  return oracle

def poly_nonnegative(c, (a, b)):
  roots = np.real(np.roots(np.squeeze(c)))
  rootsab = []
  for r in roots:
    if r >= a and r <= b:
      rootsab.append(r)
  rootsab = sorted(rootsab)
  eval_points = [a, b]
  for i in range(len(rootsab) - 1):
    eval_points.append((rootsab[i] + rootsab[i + 1]) / 2.0)
  vals = np.polyval(c, eval_points)
  for i, v in enumerate(vals):
    if v < 0.0:
      return False, eval_points[i]
  return True, None

def poly_plot(cs, (a, b)):
  colors = ['blue', 'red', 'green']
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_xlim(a, b)
  for i, c in enumerate(cs):
    xs = np.linspace(a, b, 100)
    ys = np.polyval(c, xs)
    ax.plot(xs, ys, color=colors[i])
  return fig

def make_distribution(c, (a, b)):
  nonneg, _ = poly_nonnegative(c, (a, b))
  if not nonneg:
    raise ValueError('Polynomial is negative on the given interval.')
  C = np.polyint(c)
  vals = np.polyval(C, [a, b])
  mass = vals[1] - vals[0]
  return c / mass

def get_cdf(c, (a, b)):
  C = np.polyint(np.squeeze(c))
  d = c.size
  C[d] = C[d] - np.polyval(C, [a])[0]
  return C

def sample_from(c, (a, b)):
  C = get_cdf(c, (a, b))
  d = c.size
  randval = np.random.random_sample()
  Cp = C
  Cp[d] = C[d] - randval
  roots = np.real(np.roots(Cp))
  rootsab = []
  for r in roots:
    if r >= a and r <= b:
      rootsab.append(r)
  rootvals = np.polyval(Cp, rootsab)
  real_roots = []
  for i, v in enumerate(rootvals):
    if abs(v) <= 1e-10:
      already_seen = False
      for r in real_roots:
        if abs(r - rootsab[i]) < 1e-10:
          already_seen = True
      if not already_seen:
        real_roots.append(rootsab[i])
  if len(real_roots) != 1:
    raise ValueError('Wrong number of roots.')
  return real_roots[0]

def sample_from_multiple(c, (a, b), n):
  samples = []
  for i in range(n):
    samples.append(sample_from(c, (a, b)))
  return samples

def sampling_experiment(c, (a, b), n):
  fig = poly_plot(c, (a, b))
  ax = fig.get_axes()[0]
  samples = []
  for i in range(n):
    samples.append(sample_from(c, (a, b)))
  ys = [1.0 / n] * n;
  ax.stem(samples, ys)
  return fig

def plot_poly_with_samples(c, (a, b), E):
  fig = poly_plot([c], (a, b))
  ax = fig.get_axes()[0]
  stemval = ax.get_ylim()[1] / 10.0
  ys = [stemval] * len(E)
  markerline, stemline, baseline = ax.stem(E, ys)
  plt.setp(markerline, color='red', markeredgecolor='red')
  plt.setp(stemline, color='red')
  plt.setp(baseline, linewidth=0)
  return fig

def construct_discrete_problem(c, (a, b), E):
  C = get_cdf(c, (a, b))
  Es = sorted(E)
  points = [a] + Es + [b]
  vals = np.polyval(C, Es)
  total_val = np.polyval(C, [b])[0]
  last_val = 0
  last_x = a
  discrete_vals = []
  for i, v in enumerate(vals):
    discrete_vals.append((v - last_val, (last_x, Es[i])))
    discrete_vals.append((-1.0 / len(E), (Es[i], Es[i])))
    last_val = v
    last_x = Es[i]
  discrete_vals.append((total_val - last_val, (last_x, b)))
  return discrete_vals

def negate_discrete_problem(l):
  new_prob = []
  for le in l:
    new_prob.append((-le[0], le[1]))
  return new_prob

def solve_discrete_problem(linput, k):
  def num_positive(l):
    num = 0
    for li, _ in l:
      if li >= 0:
        num += 1
    return num
  l = linput[:]
#  print '-----------------------------------'
#  print 'Input'
#  for le in l:
#    print le
  while(num_positive(l) > k):
    if l[0][0] < 0:
      l = l[1 : ]
    if l[-1][0] < 0:
      l = l[ : -1]
    min_index = 0
    for i in range(1, len(l)):
      if abs(l[i][0]) < abs(l[min_index][0]):
        min_index = i
    new_sum = l[min_index][0]
    new_left = l[min_index][1][0]
    new_right = l[min_index][1][1]
    if min_index > 0:
      new_sum += l[min_index - 1][0]
      new_left = l[min_index - 1][1][0]
      l.pop(min_index - 1)
      min_index -= 1
    if min_index < len(l) - 1:
      new_sum += l[min_index + 1][0]
      new_right = l[min_index + 1][1][1]
      l.pop(min_index + 1)
    new_element = (new_sum, (new_left, new_right))
    l[min_index] = new_element
#    print '-----------------------------------'
#    for le in l:
#      print le
  result = []
  end_sum = 0
  for le in l:
    if le[0] >= 0:
      end_sum += le[0]
      result.append(le[1])
  return end_sum, result

def compute_ak(c, (a, b), E, k):
  discrete_problem1 = construct_discrete_problem(c, (a, b), E)
  discrete_problem2 = negate_discrete_problem(discrete_problem1)
  val1, sol1 = solve_discrete_problem(discrete_problem1, k)
  val2, sol2 = solve_discrete_problem(discrete_problem2, k)
  val2 = -val2
  if abs(val1) >= abs(val2):
    val = val1
    sol = sol1
  else:
    val = val2
    sol = sol2
  return val, sol

def construct_ak_hyperplane(c, val, sol):
  h = np.zeros((c.size, 1))
  d = c.size - 1
  for j in range(c.size):
    coeff = 0
    jp = d - j
    for s in sol:
      coeff += (math.pow(s[1], jp + 1) - math.pow(s[0], jp + 1)) / (jp + 1.0)
    h[j, 0] = coeff
  if val < 0.0:
    h = h * -1.0
  return h

def construct_poly_oracle(tau, (a, b), k, E):
  def oracle(c):
    d = c.size - 1
    nonneg, x = poly_nonnegative(c, (a, b))
    if not nonneg:
      hyperplane = np.zeros((d + 1, 1))
      for i in range(c.size):
        ip = d - i
        hyperplane[i, 0] = math.pow(x, ip)
      hyperplane = -hyperplane
#      print 'negative'
      return False, hyperplane
    ak_val, ak_sol = compute_ak(c, (a, b), E, k)
    if abs(ak_val) <= tau:
      return True, c
    else:
      hyperplane = construct_ak_hyperplane(c, ak_val, ak_sol)
#      print 'large ak: {}'.format(ak_val)
#      print ak_sol
      return False, hyperplane
  return oracle
