from __future__ import print_function
from __future__ import division

import collections
import math
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate
import scipy.integrate
import scipy.stats
import scipy.optimize
import cvxopt
import sys
import time
from matplotlib.patches import Ellipse

import ellipsoid_cpp


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
#  print(np.linalg.det(A))
  if np.linalg.det(A) < 0.0:
#    print('WARNING: early exit because of instability (volume)')
    return 0.0
  dim = A.shape[0]
  s_vol = math.pow(math.pi, dim / 2.0) / math.gamma(dim / 2.0 + 1.0)
  return s_vol * math.sqrt(np.linalg.det(A))


def ellipsoid_method(E, oracle, L):
#  hyperplanes = []
  d = E.shape[0]
  a = np.zeros((d, 1))
#  ellipsoids = [(a, E)]
  num_iter = 0
#  print('\nEllipsoid d = {}'.format(d))

  while vol(E) > L:
#    print((vol(E), L))
    num_iter += 1
    res, val = oracle(a)
    if res:
      return True, val, num_iter
#      return True, val, ellipsoids, hyperplanes
    else:
#      print(math.sqrt(np.dot(val.transpose(), np.dot(E, val))))
      tmp_prod = np.dot(val.transpose(), np.dot(E, val))
      if tmp_prod <= 0.0:
#        print('WARNING: early exit because of instability (sqrt)')
        return False, E, num_iter
      b = np.dot(E, val) / math.sqrt(tmp_prod)
      a = a - (1.0 / (d + 1)) * b
#      print(a)
#      print(b)
      E = (d * d / (d * d - 1.0)) * (E - (2.0 / (d + 1)) * np.dot(b, b.transpose()))
#      ellipsoids.append((a, E))
#      hyperplanes.append(val)
  
#  return False, E, ellipsoids, hyperplanes
#  plot_ellipse(ellipsoids[:-1])
  return False, E, num_iter


def find_analytic_center(A, b, x0, gap_tolerance=1e-7):
  m = b.size
  n = x0.size

  # phase 1
  c = np.zeros(n + 1)
  c[n] = -1.0
  Ap = np.hstack((A, np.ones((m, 1))))
  Apcvx = cvxopt.matrix(Ap)
  ccvx = cvxopt.matrix(c)
#  ccvx = cvxopt.matrix(c.reshape((n + 1, 1)))
#  btmp = b.reshape((m, 1))
#  print(btmp)
#  bcvx = cvxopt.matrix(btmp, size=(m,1))
  bcvx = cvxopt.matrix(b)
#  print(Apcvx)
#  print(ccvx)
#  print(bcvx)
  sol = cvxopt.solvers.lp(ccvx, Apcvx, bcvx, solver='glpk')
  if sol['status'] != 'optimal':
    print('ERROR: cvxopt phase 1 solver failed: {}'.format(sol['status']))
    return 'error', Ap, b
  lp_sol = np.squeeze(np.array(sol['x']))
  gap = lp_sol[-1]
  x0 = lp_sol[:-1]

  if gap < gap_tolerance:
#    print('WARNING: no analytic center found')
    return 'empty', Ap, b, gap
  
  if not np.all(np.dot(A, x0) <= b):
    print('ERROR: phase 1 failed to produce a feasible point (LP gap: {}).'.format(np.squeeze(np.array(sol['x']))[-1]))
    return 'error', Ap, b, gap

#  print(c)
#  print(Ap)
#  print(b)
#  opts = {}
#  opts['maxiter'] = 1000
#  opts['disp'] = True
#  res = scipy.optimize.linprog(c, A_ub=Ap, b_ub=b, options=opts)
#  if not res.success:
#    print('ERROR: phase 1 solver failed')
#    return False, Ap, b
#  if res.status == 2:
#    print('ERROR: phase 1 problem infeasible')
#  if res.status == 3:
#    print('ERROR: phase 1 problem unbounded')
#  if res.status == 1:
#    print('ERROR: phase 1 problem iteration limit reached')
#  print('Feasibility margin: {}'.format(res.x[-1]))
#  x0 = res.x[:-1]

  max_iters = 100
  alpha = .1
  beta = .5
  tol = 1e-8


#  e = 0.01
  y = b - np.dot(A, x0)
#  y[y <= e] = 1.0

#  print('y = {}'.format(str(y)))

  x = x0
  w = np.zeros(m)
  num_iter = 0

  for ii in range(max_iters):
    num_iter += 1

    g = -1.0 / y
    H = np.diag(1.0 / (y * y))
    rd = g + w
    rp = np.dot(A, x) + y - b
#    print('{}'.format(str(np.dot(A.transpose(), w))))
#    print('rd = {}'.format(str(rd)))
#    print('rp = {}'.format(str(rp)))
    res = np.concatenate((np.dot(A.transpose(), w), rd, rp))

    if np.linalg.norm(res) < math.sqrt(tol):
#      print('Find center early exit')
      break

    Hsq = np.diag(1.0 / y)
    Hmsq = np.diag(y)
    dx, _, _, _ = np.linalg.lstsq(np.dot(Hsq, A), - np.dot(Hsq, rp) + np.dot(Hmsq, g))
    dy = -np.dot(A, dx) - rp
    dw = -np.dot(H, dy) - rd

#    print('dx = {}'.format(str(dx)))
#    print('dy = {}'.format(str(dy)))
#    print('dw = {}'.format(str(dw)))

    t = 1.0
    while np.min(y + t * dy) <= 0.0:
      t *= beta

#    print(np.dot(A.transpose(), w + t * dw))
#    print(w + t * dw - 1.0 / (y + t * dy))
#    print(np.dot(A, x + t * dx) + y + t * dy - b)
    newres = np.concatenate((np.dot(A.transpose(), w + t * dw), w + t * dw - 1.0 / (y + t * dy), np.dot(A, x + t * dx) + y + t * dy - b))

    num_linesearch_iter = 0
    while np.linalg.norm(newres) > (1.0 - alpha * t) * np.linalg.norm(res):
      t = beta * t
      newres = np.concatenate((np.dot(A.transpose(), w + t * dw), w + t * dw - 1.0 / (y + t * dy), np.dot(A, x + t * dx) + y + t * dy - b))
      num_linesearch_iter += 1
      if num_linesearch_iter > 100:
        print('Ran more than 100 iterations of line search, this should probably not happen.')
        break

    x = x + t * dx
    y = y + t * dy
    w = w + t * dw
#    print('x = {}'.format(str(x)))
#    print('y = {}'.format(str(y)))
#    print('w = {}'.format(str(w)))
  
  tmp = b - np.dot(A, x)
  H = np.dot(A.transpose(), np.dot(np.diag(1.0 / (tmp * tmp)), A))
  return 'success', x, H, num_iter


def accmp_basic(dim, bound, oracle, num_iter=10, verbose=False, gap_tolerance=1e-7):
#  sqbound = math.sqrt(bound)
  C = np.vstack((np.eye(dim), -np.eye(dim)))
  d = bound * np.ones(2 * dim)
  x = np.zeros(dim)

  x_best = x
  val_best = 1e100
  total_num_newton_iter = 0
  num_oracle_calls = 0

  for ii in range(num_iter):
#    print('x_best = {}'.format(str(x_best)))
#    print('val_best = {}'.format(val_best))
#    print('C = {}'.format(str(C)))
#    print('d = {}'.format(str(d)))
#    status, query, H, n_newton = find_analytic_center(C, d, x_best)
    res = find_analytic_center(C, d, x_best, gap_tolerance=gap_tolerance)
    if res[0] == 'empty':
      return x_best, val_best, num_oracle_calls, total_num_newton_iter
    if res[0] == 'error':
      return None, num_oracle_calls, total_num_newton_iter, res[1], res[2]
    if res[0] != 'success':
      print('ERROR: unknown status code from find_analytic_center: {}'.format(res[0]))
      print(res)
      return None
    query = res[1]
    H = res[2]
    n_newton = res[3]
    total_num_newton_iter += n_newton
#    print('query = {}  num Newton iterations = {}'.format(str(query), n_newton))
    feasible, subgradient, value = oracle(query)
    num_oracle_calls += 1
#    print('feasible = {}  value = {}'.format(feasible, value))
#    print('subgradient = {}'.format(str(subgradient)))
    if not feasible:
      C = np.vstack((C, subgradient))
      d = np.concatenate((d, [value]))
    else:
      if value < val_best:
        val_best = value
        x_best = query
      C = np.vstack((C, subgradient))
      d = np.concatenate((d, [np.dot(subgradient, query) + val_best - value]))
#      d = np.concatenate((d, [np.dot(subgradient, query)]))
  return x_best, val_best, num_oracle_calls, total_num_newton_iter


def make_accmp_test_oracle(xmin, xmax):
  d = xmin.shape[0]
  def oracle(x):
    hyperplane = np.zeros(d)
    for i in range(d):
      if x[i] < xmin[i]:
        hyperplane[i] = -1.0
        return False, hyperplane, -xmin[i]
    for i in range(d):
      if x[i] > xmax[i]:
        hyperplane[i] = 1.0
        return False, hyperplane, xmax[i]
    return True, x, np.linalg.norm(x) * np.linalg.norm(x)
  return oracle


def construct_poly_accmp_oracle((a, b), k, samples, sample_weight):
  def oracle(c):
    d = c.size - 1
    nonneg, x = poly_nonnegative(c, (a, b))
    if not nonneg:
      hyperplane = np.zeros(d + 1)
#      print('Negative at point {}'.format(x))
      for i in range(c.size):
        ip = d - i
        hyperplane[i] = math.pow(x, ip)
      hyperplane = -hyperplane
#      print 'negative'
      return False, hyperplane, 0.0
    ak_val, ak_sol = compute_ak_cpp2(c, (a, b), samples, sample_weight, k)
    hyperplane = np.squeeze(construct_ak_hyperplane(c, ak_val, ak_sol))
#    print('Ak distance (with sign) {}'.format(ak_val))
    return True, hyperplane, abs(ak_val)
  return oracle


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


def get_kde_pdf(kde):
  def pdf(xs):
    return np.exp(kde.score_samples(xs.reshape(-1, 1)))
  return pdf


def plot_kde(kdes, (a, b), fig=None, colors=['blue', 'red', 'green'], num_points=300):
  if fig is None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(a, b)
  else:
    ax = fig.get_axes()[0]
  xs = np.linspace(a, b, num_points)

  for ii, kde in enumerate(kdes):
    ys = np.exp(kde.score_samples(xs.reshape(-1, 1)))
    ax.plot(xs, ys, color=colors[ii])

  return fig


def plot_poly(polys, (a, b), fig=None, colors=['blue', 'red', 'green'], num_points=300):
  if fig is None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(a, b)
  else:
    ax = fig.get_axes()[0]

  for i, poly in enumerate(polys):
    if isinstance(poly[0], tuple):
      xs = []
      ys = []
      for piece in poly:
        cur_xs = np.linspace(piece.left, piece.right, num_points * (piece.right - piece.left) / (b - a) + 2)
        cur_ys = np.polyval(piece.hypothesis, cur_xs)
        xs.extend(cur_xs)
        ys.extend(cur_ys)
    else:
      xs = np.linspace(a, b, num_points)
      ys = np.polyval(poly, xs)
    ax.plot(xs, ys, color=colors[i])

  return fig


def spline_plot(splines, (a, b), fig=None, colors=['green', 'red', 'blue']):
  num_points = 300
  if fig is None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(a, b)
  else:
    ax = fig.get_axes()[0]

  ax.set_ylim(ymin=0)
  xs = np.linspace(a, b, num_points)
  for ii, spl in enumerate(splines):
    ys = spl(xs)
    ax.plot(xs, ys, color=colors[ii])

  return fig


class gmm_distribution:
  def __init__(self, params):
    self.params = params

  def get_pdf(self):
    return get_mixture_pdf(self.params)

  def draw_samples(self, n):
    return sample_from_mixture(self.params, n)


class normal_distribution:
  def __init__(self, mean, variance):
    self.mean = mean
    self.variance = variance

  def get_pdf(self):
    def pdf(x):
      return scipy.stats.norm.pdf(x, loc=self.mean, scale=self.variance)
    return pdf

  def draw_samples(self, n):
    return np.random.normal(self.mean, self.variance, n)


class beta_distribution:
  def __init__(self, a, b):
    self.a = a
    self.b = b

  def get_pdf(self):
    def pdf(x):
      return scipy.stats.beta.pdf(x, a=self.a, b=self.b)
    return pdf

  def draw_samples(self, n):
    return numpy.random.beta(self.a, self.b, n)


class gamma_distribution:
  def __init__(self, shape, scale):
    self.shape = shape
    self.scale = scale

  def get_pdf(self):
    def pdf(x):
      return scipy.stats.gamma.pdf(x, self.shape, scale=self.scale)
    return pdf

  def draw_samples(self, n):
    return numpy.random.gamma(self.shape, self.scale, n)


class mixture_distribution:
  def __init__(self, components, weights):
    self.components = components
    self.weights = weights

  def get_pdf(self):
    def pdf(x):
      component_pdfs = [comp.get_pdf() for comp in self.components]
      y = self.weights[0] * component_pdfs[0](x)
      for ii in range(1, len(self.weights)):
        y += self.weights[ii] * component_pdfs[ii](x)
      return y
    return pdf

  def draw_samples(self, n):
    samples = np.array([])
    comp_indices = np.random.multinomial(n, self.weights)
    for ii, num in enumerate(comp_indices):
      samples = np.append(samples, self.components[ii].draw_samples(num))
    np.random.shuffle(samples)
    return samples


def get_mixture_pdf(mixture):
  def pdf(x):
    m0 = mixture[0]
    y = m0[0] * scipy.stats.norm.pdf(x, loc=m0[1], scale=m0[2])
    for ii in range(1, len(mixture)):
      m = mixture[ii]
      y += m[0] * scipy.stats.norm.pdf(x, loc=m[1], scale=m[2])
    return y
  return pdf


def get_ppoly_pdf(ppoly):
  def pdf(x):
    y = np.zeros(x.shape)
    for part in ppoly:
      x_ind = np.logical_and(x >= part.left, x <= part.right)
      y[x_ind] = np.polyval(part.hypothesis, x[x_ind])
    return y
  return pdf



def compute_l1_mc(pdf1, pdf2, (a, b), num_points):
  samples = np.random.uniform(a, b, num_points)
  val1 = pdf1(samples)
  val2 = pdf2(samples)
  diff = np.abs(val1 - val2)
  return (b - a) / num_points * np.sum(diff)


def compute_l1_quad(pdf1, pdf2, (a, b)):
  func = lambda x : np.abs(pdf1(x) - pdf2(x))
#  integral, err = scipy.integrate.quadrature(func, a, b, vec_func=True, maxiter=70, tol=1e-10)
#  integral, err = scipy.integrate.quadrature(func, a, b, vec_func=True, maxiter=70, tol=1e-10)
#  print(err)
#  integral, _ = scipy.integrate.fixed_quad(func, a, b, n=200)
  integral = scipy.integrate.romberg(func, a, b, vec_func=True, tol=1e-5, divmax=30)
#  integral, _ = scipy.integrate.quad(func, a, b)
  return integral


def fit_spline(ppoly, num_points, degree):
  xs = []
  ys = []
  knots = []
  a, b = ppoly[0].left, ppoly[-1].right
  for piece in ppoly:
    cur_xs = np.linspace(piece.left, piece.right, num_points * (piece.right - piece.left) / (b - a))
    cur_ys = np.polyval(piece.hypothesis, cur_xs)
    xs.extend(cur_xs)
    ys.extend(cur_ys)
    knots.append(piece.right)
  knots = knots[:-1]
  spline = scipy.interpolate.LSQUnivariateSpline(xs, ys, knots, bbox=(a,b), k=degree)
  return spline, xs, ys


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


def sample_multiple_from(c, (a, b), n):
  samples = []
  for i in range(n):
    samples.append(sample_from(c, (a, b)))
  return sorted(samples)


def plot_distribution(distribution, (l, r), fig=None, color='blue', num_points=300):
  xs = np.linspace(l, r, num_points)
  pdf = distribution.get_pdf()
  ys = pdf(xs)
  if fig is None: 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(l, r)
  else:
    ax = fig.get_axes()[0]
  ax.plot(xs, ys, color=color)
  return fig


def plot_distribution_with_samples(distribution, samples, (a, b)):
  fig = plot_distribution(distribution, (a, b))
  ax = fig.get_axes()[0]
  stemval = ax.get_ylim()[1] / 10.0
  ys = [stemval] * len(samples)
  markerline, stemline, baseline = ax.stem(samples, ys)
  plt.setp(markerline, color='red', markeredgecolor='red')
  plt.setp(stemline, color='red')
  plt.setp(baseline, linewidth=0)
  return fig


def sample_from_mixture(mixture, n):
  samples = []
  weights = [m[0] for m in mixture]
  component_indices = np.random.multinomial(n, weights)
  for ii, num in enumerate(component_indices):
    samples.extend(list(np.random.normal(mixture[ii][1], mixture[ii][2], num)))
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



def compute_ak_cpp2(c, (a, b), E, sample_weight, k):
  C = get_cdf(c, (a, b))
  val, sol = ellipsoid_cpp.compute_ak_cpp(C, (a, b), E, sample_weight, k)
  return val, list(sol)


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


def construct_poly_oracle(tau, (a, b), k, E, sample_weight):
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
    ak_val, ak_sol = compute_ak_cpp2(c, (a, b), E, sample_weight, k)
    if abs(ak_val) <= tau:
#      print('Point feasible, error {} <= {}: {}'.format(abs(ak_val), tau, str(c)))
      return True, c
    else:
      hyperplane = construct_ak_hyperplane(c, ak_val, ak_sol)
#      print 'large ak: {}'.format(ak_val)
#      print ak_sol
      return False, hyperplane
  return oracle


def compute_tv_norm(c, (a, b)):
  C = np.polyint(c)
  roots = np.real(np.roots(c))
  rootsab = [a, b]
  for r in roots:
    if r >= a and r <= b:
      rootsab.append(r)
  rootsab = sorted(rootsab)
  res = 0.0
  vals = np.polyval(C, rootsab)
  for ii in range(len(vals) - 1):
    res += abs(vals[ii + 1] - vals[ii])
  return res


def convert_histogram_to_pp_hypothesis(hist):
  res = []
  for ii in hist:
    res.append(HypothesisPiece(ii.left, ii.right, ii.left_sample_index, ii.right_sample_index, [ii.value]))
  return res

def convert_piecewise_linear_to_pp_hypothesis(hist):
  res = []
  for ii in hist:
    res.append(HypothesisPiece(ii.left, ii.right, ii.left_sample_index, ii.right_sample_index, [ii.slope, ii.offset]))
  return res

def compute_derivative_coefficients(x, d, num_derivs):
  moment_curve = [math.pow(x, y) for y in range(d, -1, -1)]
  res = np.zeros((d + 1))
  for ii in range(d + 1):
    if (d - ii) - num_derivs >= 0:
      res[ii] = math.pow(x, d - ii - num_derivs)
      res[ii] *= math.factorial(d - ii) / math.factorial(d - ii - num_derivs)
    else:
      res[ii] = 0.0
  return res


def pp_patch(ppoly, eps, samples, remove_small_pieces=False, force_boundaries_to_zero=False):
  eps = float(eps)
  n = len(samples)
  d = len(ppoly[0].hypothesis) - 1
  n_samples_eps2 = int(eps * n / 2.0)
  n_samples_eps = int(eps * n)

  # step 1: removing small intervals
  if remove_small_pieces:
    tmp_ppoly1 = [ppart for ppart in ppoly if ppart.right_sample_index - ppart.left_sample_index > n_samples_eps or (not force_boundaries_to_zero and (ppart.left_sample_index == 0 or ppart.right_sample_index == len(samples)))]
  else:
    tmp_ppoly1 = [ppart for ppart in ppoly]
  
  tmp_ppoly2 = []
  for ii in range(len(tmp_ppoly1)):
    cur = tmp_ppoly1[ii]
    if cur.left_sample_index != 0 or force_boundaries_to_zero:
      new_left_sample_index = min(cur.left_sample_index + n_samples_eps2, len(samples) - 1)
      if cur.left_sample_index == 0:
        new_left = ppoly[0].left
      else:
        new_left = (samples[new_left_sample_index] + samples[new_left_sample_index - 1]) / 2.0
    else:
      new_left_sample_index = 0
      new_left = ppoly[0].left
    if cur.right_sample_index != len(samples) or force_boundaries_to_zero:
      new_right_sample_index = max(cur.right_sample_index - n_samples_eps2, new_left_sample_index + 1)
      if cur.right_sample_index == len(samples):
        new_right = ppoly[-1].right
      else:
        new_right = (samples[new_right_sample_index] + samples[new_right_sample_index - 1]) / 2.0
    else:
      new_right_sample_index = len(samples)
      new_right = ppoly[-1].right
    new_piece = HypothesisPiece(new_left, new_right, new_left_sample_index, new_right_sample_index, cur.hypothesis)
    tmp_ppoly2.append(new_piece)

#  print(tmp_ppoly2)

  res = []

  # step 2: insert connection pieces
  if tmp_ppoly2[0].left_sample_index != 0:
    if not force_boundaries_to_zero:
      print('ERROR: no left boundary - this should not happen.')
      return None
    left = ppoly[0].left
    left_sample_index = 0
    right_sample_index = tmp_ppoly[0].left_sample_index
    right = tmp_ppoly[0].left
    print('TODO: fit polynomial')
    res.append(new_piece)
  else:
    res.append(tmp_ppoly2[0])

  for ii in range(len(tmp_ppoly2) - 1):
    pleft = res[-1]
    pright = tmp_ppoly2[ii + 1]
   
    mid_left = pleft.right
    mid_right = pright.left
    num_constraints = d + 1 if d % 2 == 1 else d
#    num_constraints = min(num_constraints, 4)
    constraints_matrix = np.zeros((num_constraints, d + 1))
#    constraints_matrix = np.zeros((num_constraints, num_constraints))
    constraints_values = np.zeros((num_constraints))

    moment_curve_left = [math.pow(mid_left, x) for x in range(d, -1, -1)]
    moment_curve_right = [math.pow(mid_right, x) for x in range(d, -1, -1)]

    for jj in range(num_constraints // 2):
      tmp_hyp_l = np.polyder(pleft.hypothesis, jj)
      tmp_hyp_r = np.polyder(pright.hypothesis, jj)

      constraints_values[2 * jj] = np.polyval(tmp_hyp_l, [mid_left])[0]
      constraints_values[2 * jj + 1] = np.polyval(tmp_hyp_r, [mid_right])[0]
#      print('{}-th derivatives to match: {} and {}'.format(jj, constraints_values[2 * jj], constraints_values[2 * jj + 1]))
      constraints_matrix[2 * jj, :] = compute_derivative_coefficients(mid_left, d, jj)
      constraints_matrix[2 * jj + 1, :] = compute_derivative_coefficients(mid_right, d, jj)

    if d % 2 == 1:
      mid_hyp, _, _, _ = np.linalg.lstsq(constraints_matrix, constraints_values)
    else:
      print('NOT YET IMPLEMENTED')

    res.append(HypothesisPiece(mid_left, mid_right, pleft.right_sample_index, pright.left_sample_index, mid_hyp))
    res.append(pright)

  if tmp_ppoly2[-1].right_sample_index != len(samples):
    if not force_boundaries_to_zero:
      print('ERROR: no right boundary - this should not happen.')
    print('TODO: add right boundary piece')

#    print(res[0])
#    print(res[1])
#    print(res[2])

  return res


HypothesisPiece = collections.namedtuple('HypothesisPiece', ['left', 'right', 'left_sample_index', 'right_sample_index', 'hypothesis'])

def pp_learning(target_num_pieces, d, initial_num_pieces, (a, b), samples, ak_delta=0.001, verbose=0, akproj_num_iter=25, akproj_upper_bound=-1, akproj_gap_tolerance=1e-7):
  # Initial partitioning
  n = len(samples)
  samples_per_piece = n // initial_num_pieces
  num_extra_samples = n % initial_num_pieces
  sample_weight = 1.0 / n

  intervals = [[]]
  cur_intervals = intervals[0]
  
  if verbose >= 1:
    print('Computing initial pieces')
    sys.stdout.flush()

  last_boundary = a
  last_sample_index = 0
  for ii in range(initial_num_pieces):
    left = last_boundary
    left_sample_index = last_sample_index
    right_sample_index = left_sample_index + samples_per_piece
    if ii < num_extra_samples:
      right_sample_index += 1
    if ii == initial_num_pieces - 1:
      right = b
    else:
      right = (samples[right_sample_index - 1] + samples[right_sample_index]) / 2.0
#    print((left, right, left_sample_index, right_sample_index, ak_delta))
#    hypothesis, _, _ = project_Ak(d, d, (left, right), samples[left_sample_index : right_sample_index], sample_weight=sample_weight, delta=ak_delta, verbose=(verbose >= 3))
    hypothesis, _, num_oracle, num_newton = project_Ak_accmp(d, d, (left, right), samples[left_sample_index : right_sample_index], sample_weight=sample_weight, verbose=(verbose >= 3), num_iter=akproj_num_iter, upper_bound=akproj_upper_bound, gap_tolerance=akproj_gap_tolerance)
    if verbose >= 2:
      print('num oracle = {}  num_newton = {}\n'.format(num_oracle, num_newton))
    cur = HypothesisPiece(left, right, left_sample_index, right_sample_index, hypothesis)
    cur_intervals.append(cur)
    last_boundary = right
    last_sample_index = right_sample_index

#  for ii in cur_intervals:
#    print(ii)
  
  while len(cur_intervals) > target_num_pieces and not (len(cur_intervals) == target_num_pieces + 1 and len(cur_intervals) % 2 == 1):
    candidates = []
    cur_intervals = intervals[-1]
    
    if len(intervals) >= 2 and len(cur_intervals) == len(intervals[-2]):
      print('ERROR: last merging iteration made no progress. Exiting.\n')

    if verbose >= 1:
      print('Current number of intervals: {}'.format(len(cur_intervals)))
      sys.stdout.flush()

    for ii in range(len(cur_intervals) // 2):
      left = cur_intervals[2 * ii].left
      right = cur_intervals[2 * ii + 1].right
      left_sample_index = cur_intervals[2 * ii].left_sample_index
      right_sample_index = cur_intervals[2 * ii + 1].right_sample_index
      #print('Input to Akproj: {}, {}, ({}, {}), samples from {} to {}, {}, {}'.format(d, d, left, right, left_sample_index, right_sample_index, sample_weight, ak_delta))
#      hypothesis, _, _ = project_Ak(d, d, (left, right), samples[left_sample_index : right_sample_index], sample_weight=sample_weight, delta=ak_delta, verbose=(verbose >= 2))
      hypothesis, _, num_oracle, num_newton = project_Ak_accmp(d, d, (left, right), samples[left_sample_index : right_sample_index], sample_weight=sample_weight, verbose=(verbose >= 2), num_iter=akproj_num_iter, upper_bound=akproj_upper_bound, gap_tolerance=akproj_gap_tolerance)
      if verbose >= 2:
        print('num oracle = {}  num_newton = {}\n'.format(num_oracle, num_newton))
      err, _ = compute_ak_cpp2(hypothesis, (left, right), samples[left_sample_index : right_sample_index], sample_weight, d)
      err = abs(err)
      #print('Error: {}'.format(err))
      candidates.append((ii, hypothesis, err))
    
    errors = sorted([x[2] for x in candidates])
    threshold = errors[-(target_num_pieces // 2)]

    next_intervals = []
    for ii, hyp, err in candidates:
      if err < threshold:
        left = cur_intervals[2 * ii].left
        right = cur_intervals[2 * ii + 1].right
        left_sample_index = cur_intervals[2 * ii].left_sample_index
        right_sample_index = cur_intervals[2 * ii + 1].right_sample_index
        cur = HypothesisPiece(left, right, left_sample_index, right_sample_index, hyp)
        next_intervals.append(cur)
      else:
        next_intervals.append(cur_intervals[2 * ii])
        next_intervals.append(cur_intervals[2 * ii + 1])

    if len(cur_intervals) % 2 == 1:
      next_intervals.append(cur_intervals[-1])
    intervals.append(next_intervals)
    #print('\n------------------------------------------------------------\n')
    #for ii in next_intervals:
    #  print(ii)

  return intervals[-1]


def get_num_samples_in_range_inclusive(samples, a, b):
  return len([x for x in samples if (x >= a and x <= b)])

def get_num_samples_in_range_exclusive(samples, a, b):
  return len([x for x in samples if (x > a and x < b)])


def project_A1_linear((a, b), samples, sample_weight=-1, verbose=False, gap=0.0001, num_hierarchy_levels=2, max_num_iter=20):
  if sample_weight == -1:
    sample_weight = 1.0 / len(samples)
  total_weight = sample_weight * len(samples)

  lpA = np.array([[-a, -1.0, 0.0], [-b, -1.0, 0.0], [(b*b - a*a) / 2.0, b-a, -1.0], [(a*a - b*b) / 2.0, a-b, -1.0]])
  lpb = np.array([0.0, 0.0, total_weight, -total_weight])
  lpc = np.array([0.0, 0.0, 1.0])

  num_hierarchy_parts = 1
  for ii in range(num_hierarchy_levels - 1):
    num_hierarchy_parts *= 2
    interval_width = (b - a) / float(num_hierarchy_parts)
    for jj in range(num_hierarchy_parts):
      c = a + jj * interval_width
      d = c + interval_width
      wpos = sample_weight * get_num_samples_in_range_exclusive(samples, c, d)
      wneg = sample_weight * get_num_samples_in_range_inclusive(samples, c, d)
      print('left = {}  right = {}  weight = {}'.format(c, d, wpos))
      new_rows = np.array([[(d*d - c*c) / 2.0, d-c, -1.0], [(c*c - d*d) / 2.0, c-d, -1.0]])
      lpA = np.vstack((lpA, new_rows))
      lpb = np.concatenate((lpb, np.array([wpos, -wneg])))
    for jj in range(num_hierarchy_parts - 1):
      c = a + jj * interval_width + interval_width / 2.0
      d = c + interval_width
      wpos = sample_weight * get_num_samples_in_range_exclusive(samples, c, d)
      wneg = sample_weight * get_num_samples_in_range_inclusive(samples, c, d)
      print('left = {}  right = {}  weight = {}'.format(c, d, wpos))
      new_rows = np.array([[(d*d - c*c) / 2.0, d-c, -1.0], [(c*c - d*d) / 2.0, c-d, -1.0]])
      lpA = np.vstack((lpA, new_rows))
      lpb = np.concatenate((lpb, np.array([wpos, -wneg])))
 
  lower_bound = 0.0
  best = 2 * total_weight
  best_coeffs = np.array([0.0, 0.0])
  num_iter = 1
  while best - lower_bound > gap and num_iter <= max_num_iter:
    print('num_iter = {}  best = {}  lower_bound = {}  gap = {}'.format(num_iter, best, lower_bound, best - lower_bound))
    num_iter += 1

    Acvx = cvxopt.matrix(lpA)
    bcvx = cvxopt.matrix(lpb)
    ccvx = cvxopt.matrix(lpc)
    sol = cvxopt.solvers.lp(ccvx, Acvx, bcvx, solver='glpk')
    if sol['status'] != 'optimal':
      print('ERROR: cvxopt solver failed: {}'.format(sol['status']))
      return 'error', lpA, lpb
    lp_sol = np.squeeze(np.array(sol['x']))
    lower_bound = lp_sol[-1]
    coeffs = lp_sol[:-1]

    a1_dst, tmp_intervals = compute_ak_cpp2(coeffs, (a, b), samples, sample_weight, 1)
    print('  a1_dst = {}'.format(abs(a1_dst)))

    if abs(a1_dst) < best:
      print('  found new best')
      best = abs(a1_dst)
      best_coeffs = coeffs

    c, d = tmp_intervals[0]
    if a1_dst >= 0.0:
      w = sample_weight * get_num_samples_in_range_exclusive(samples, c, d)
      print('  adding constraint for positive interval ({}, {}), weight = {}'.format(c, d, w))
      new_rows = np.array([[(d*d - c*c) / 2.0, d-c, -1.0]])
      lpA = np.vstack((lpA, new_rows))
      lpb = np.concatenate((lpb, np.array([w])))
    else:
      w = sample_weight * get_num_samples_in_range_inclusive(samples, c, d)
      new_rows = np.array([[(c*c - d*d) / 2.0, c-d, -1.0]])
      print('  adding constraint for negative interval ({}, {}), weight = {}'.format(c, d, w))
      lpA = np.vstack((lpA, new_rows))
      lpb = np.concatenate((lpb, np.array([-w])))
  
  print('final gap = {}  num_a1_calls = {}'.format(best - lower_bound, num_iter - 1))
  return best_coeffs, best, lpA, lpb, lpc



def project_Ak_accmp(d, k, (a, b), samples, sample_weight=-1, num_iter=25, verbose=False, upper_bound=-1, gap_tolerance=1e-7):
  if sample_weight == -1:
    sample_weight = 1.0 / len(samples)
  if upper_bound < 0:
    upper_bound = math.pow(math.sqrt(d + 1) * math.pow(1.0 + math.sqrt(2.0), d), d + 1)
  if verbose:
    print('upper_bound = {}\n'.format(upper_bound))
  oracle = construct_poly_accmp_oracle((a, b), k, samples, sample_weight)
  res = accmp_basic(d + 1, upper_bound, oracle, num_iter, gap_tolerance=gap_tolerance)
  return res


def project_Ak(d, k, (a, b), E, sample_weight=-1, delta=0.01, verbose=False):
  if sample_weight == -1:
    sample_weight = 1.0 / len(E)
  U = math.pow(math.sqrt(d + 1) * math.pow(1.0 + math.sqrt(2.0), d), d + 1)
  E0 = U * np.identity(d + 1)
  L = math.pow(delta / (d * math.pow(b - a + 1, d + 1)), d + 1)
  if verbose:
    print('U = {}  L = {}'.format(U, L))
#    print(E)
#    print((a, b))
  tau_low = 0.0
  tau_high = 1.0
  best_point = np.zeros((d + 1, 1))
  num_inner_iter = 0
  num_iter = 0
  while tau_high - tau_low > delta:
    num_iter += 1
    tau_m = (tau_high + tau_low) / 2.0
    if verbose:
      print('tau_low: {}   tau_high: {}   tau_m: {}'.format(tau_low, tau_high, tau_m), end='')
    oracle = construct_poly_oracle(tau_m, (a, b), d, E, sample_weight)
    ret, Ep, tmp_num_iter = ellipsoid_method(E0, oracle, L)
    num_inner_iter += tmp_num_iter
    if verbose:
      print('    num inner ellipsoid iterations: {}'.format(tmp_num_iter))
    if ret:
      tau_high = tau_m
      best_point = np.squeeze(Ep)
    else:
      tau_low = tau_m
  return best_point, num_iter, num_inner_iter


def run_experiment(c, (a, b), n_vals, num_trials):
  d = c.size - 1
  print('d = {}, a = {}, b = {}, c = {}'.format(d, a, b, c))
  eps_mean_vals = []
  eps_stddev_vals = []
  for n in n_vals:
    print('n = {}  '.format(n), end='')
    time_akproj = 0.0
    num_ellipsoids = 0
    num_inner_ellipsoid_iterations = 0
    tstart = time.clock()
    trial_vals = []
    for ii in range(num_trials):
      samples = sample_multiple_from(c, (a, b), n)
      tstart_akproj = time.clock()
      chat, num_iter, num_inner_iter = project_Ak(d, d, (a, b), samples, 0.5 * math.sqrt(float(d) / n))
      tend_akproj = time.clock()
      time_akproj += tend_akproj- tstart_akproj
      num_ellipsoids += num_iter
      num_inner_ellipsoid_iterations += num_inner_iter
      trial_vals.append(compute_tv_norm(c - chat, (a, b)))
      print('.', end='')
    tend = time.clock()
    print('  {} seconds, {} Ellipsoid runs, and {} inner Ellipsoid iterations per Ak-projection on average'.format(time_akproj / num_trials, float(num_ellipsoids) / num_trials, float(num_inner_ellipsoid_iterations) / num_trials))
    eps_mean_vals.append(np.mean(trial_vals))
    eps_stddev_vals.append(np.std(trial_vals))
  return eps_mean_vals, eps_stddev_vals











#######################################################
# Old code

def construct_discrete_cpp_problem(c, (a, b), E):
  C = get_cdf(c, (a, b))
  if E[0] != a:
    if E[-1] != b:
      points = [a] + E + [b]
    else:
      points = [a] + E
  else:
    if E[-1] != b:
      points = E + [b]
    else:
      points = E
  vals = np.polyval(C, E)
  total_val = np.polyval(C, [b])[0]
  last_val = 0
  last_x = a
#  problem = []
  problem = ellipsoid_cpp.AkIntervalVector()
  for i, v in enumerate(vals):
    int1 = ellipsoid_cpp.AkInterval()
    int1.left = last_x
    int1.right = E[i]
    int1.value = v - last_val
    problem.push_back(int1)
#    problem.append((v - last_val, (last_x, E[i])))
    int2 = ellipsoid_cpp.AkInterval()
    int2.left = E[i]
    int2.right = E[i]
    int2.value = -1.0 / len(E)
    problem.push_back(int2)
#    problem.append((-1.0 / len(E), (E[i], E[i])))
    last_val = v
    last_x = E[i]
  int3 = ellipsoid_cpp.AkInterval()
  int3.left = last_x
  int3.right = b
  int3.value = total_val - last_val
  problem.push_back(int3)
#  problem.append((total_val - last_val, (last_x, b)))
  return problem


def construct_discrete_problem(c, (a, b), E):
  C = get_cdf(c, (a, b))
  points = [a] + E + [b]
  vals = np.polyval(C, E)
  total_val = np.polyval(C, [b])[0]
  last_val = 0
  last_x = a
  discrete_vals = []
  for i, v in enumerate(vals):
    discrete_vals.append((v - last_val, (last_x, E[i])))
    discrete_vals.append((-1.0 / len(E), (E[i], E[i])))
    last_val = v
    last_x = E[i]
  discrete_vals.append((total_val - last_val, (last_x, b)))
  return discrete_vals


def negate_discrete_problem(l):
  new_prob = []
  for le in l:
    new_prob.append((-le[0], le[1]))
  return new_prob


def negate_discrete_cpp_problem(l):
  for ii in range(len(l)):
    l[ii].value = -l[ii].value
  return l


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


def compute_ak_cpp(c, (a, b), E, k):
  discrete_problem1 = construct_discrete_cpp_problem(c, (a, b), E)
  sol1 = ellipsoid_cpp.DoublePairVector()
  val1 = ellipsoid_cpp.solve_discrete_problem_cpp(discrete_problem1, k, sol1)
#  val1 = ellipsoid_cpp.solve_discrete_problem_cpp(discrete_problem1, k, sol1)
  sol2 = ellipsoid_cpp.DoublePairVector()
  discrete_problem2 = negate_discrete_cpp_problem(discrete_problem1)
  val2 = ellipsoid_cpp.solve_discrete_problem_cpp(discrete_problem2, k, sol2)
  val2 = -val2
  if abs(val1) >= abs(val2):
    return val1, list(sol1)
  else:
    return val2, list(sol2)
