import numba, time, tqdm, cv2

import numpy as np
import matplotlib.pyplot as plt

@numba.jit(cache=True)
def normalize(v: numba.float64[:]):
  return v / np.linalg.norm(v)

@numba.experimental.jitclass([
    ('pos', numba.float64[:]),
    ('dir', numba.float64[:]),
])
class Ray:
  def __init__(self, pos, dir):
    self.pos = pos
    self.dir = dir

@numba.experimental.jitclass([
    ('t', numba.float64),
    ('pos', numba.float64[:]),
    ('normal', numba.float64[:]),
])
class Intersection:
  def __init__(self, t, pos, normal):
    self.t = t
    self.pos = pos
    self.normal = normal

@numba.experimental.jitclass([
    ('center', numba.float64[:]),
    ('velocity', numba.float64[:]),
    ('radius', numba.float64),
])
class Sphere:
  """
  Sphere shape
  """
  def __init__(self, center, radius):
    self.center = center
    self.radius = radius
    self.velocity = np.zeros_like(center)

  def intersect(self, ray: Ray):
    pos_center = ray.pos - self.center
    a = 1
    b = 2 * np.dot(ray.dir, pos_center)
    c = np.dot(pos_center, pos_center) - self.radius * self.radius

    empty_vec = np.empty_like(ray.pos)
    intersect = None
    delta: np.float64 = b**2 - 4*a*c
    if delta > 1e-7:
      x0: np.float64 = (-b - np.sqrt(delta)) / (2 * a)
      x1: np.float64 = (-b + np.sqrt(delta)) / (2 * a)
      if x0 >= 0:
        # x0 is closest.
        x: np.float64 = x0
        intersect = Intersection(x, ray.pos + x * ray.dir, empty_vec)
      elif x1 >= 0:
        # x0 is negative, x1 is closest
        x: np.float64 = x1
        intersect = Intersection(x, ray.pos + x * ray.dir, empty_vec)
    elif abs(delta) < 1e-7:
      x: np.float64 = -b / (2*a)
      if x >= 0:
        intersect = Intersection(x, ray.pos + x * ray.dir, empty_vec)
    if intersect is None:
      return False, Intersection(np.inf, empty_vec, empty_vec)
    else:
      return True, intersect

@numba.experimental.jitclass([
  ('center', numba.float64[:]),
  ('size', numba.float64[:]),
])
class Cube:
  def __init__(self, center, size):
    self.center = center
    self.size = size

  def intersect(self, r: Ray):
    """Ray-box intersection test
    @OPTIONAL: Implement ray-box intersection test
    See https://gamedev.stackexchange.com/a/18459
    You don't need to fill in Intersection.normal yet; leave it None

    Useful numpy methods
     - np.maximum / np.minimum
     - np.max / np.min
    """

    df_x = 1.0 / r.dir[0]
    df_y = 1.0 / r.dir[1]
    df_z = 1.0 / r.dir[2]

    lb = self.center - self.size * 0.5
    rt = self.center + self.size * 0.5

    t1 = (lb[0] - r.pos[0]) * df_x
    t2 = (rt[0] - r.pos[0]) * df_x
    t3 = (lb[1] - r.pos[1]) * df_y
    t4 = (rt[1] - r.pos[1]) * df_y
    t5 = (lb[2] - r.pos[2]) * df_z
    t6 = (rt[2] - r.pos[2]) * df_z

    tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
    tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))

    empty_vec = np.empty_like(r.pos)
    if tmax < 0: return False, Intersection(np.inf, empty_vec, empty_vec)
    if tmin > tmax: return False, Intersection(np.inf, empty_vec, empty_vec)

    t = tmin
    return True, Intersection(t, r.pos + r.dir * t, empty_vec)

@numba.experimental.jitclass([
  ('sphere_shapes', numba.types.ListType(Sphere.class_type.instance_type)),
  ('cube_shapes', numba.types.ListType(Cube.class_type.instance_type))
])
class CompositeShape:
  def __init__(self, spheres, cubes):
    self.sphere_shapes = spheres
    self.cube_shapes = cubes
  
  def intersect(self, ray: Ray):
    """Ray-composite shape intersection test
    @TODO: Implement ray-composite shape intersection test
    For every shape in self.shapes, return Intersection with the smallest t
    """
    min_t = np.inf
    empty_vec = np.empty_like(ray.pos)
    min_intersect = (False, Intersection(np.inf, empty_vec, empty_vec),)
    for shape in self.sphere_shapes:
      is_hit, intersect = shape.intersect(ray)
      if is_hit and intersect.t <= min_t:
        min_t = intersect.t
        min_intersect = (True, intersect)
    for shape in self.cube_shapes:
      is_hit, intersect = shape.intersect(ray)
      if is_hit and intersect.t <= min_t:
        min_t = intersect.t
        min_intersect = (True, intersect)
    return min_intersect

@numba.jit(parallel=False, fastmath=True)
def main(img:numba.float64[:,:], simul_steps:int):
  #img = np.zeros((W, H))
  W, H = img.shape
  camera_center = np.array([0,0,1], dtype=np.float64)
  spheres = [
    Sphere(np.array([1.5,0,-2], dtype=np.float64), 0.2), 
    Sphere(np.array([1,1,-3], dtype=np.float64), 0.3),
    Sphere(np.array([0.8,0.1,-3], dtype=np.float64), 0.2),
    Sphere(np.array([2,0.5,-4], dtype=np.float64), 0.1),
    Sphere(np.array([0.1,-1.5,-3.5], dtype=np.float64), 0.15),
    Sphere(np.array([1.5,-0.9,-2.5], dtype=np.float64), 0.12),
    Sphere(np.array([1.25,-0.1,-2.7], dtype=np.float64), 0.11),
    Sphere(np.array([0,1.25,-3.7], dtype=np.float64), 0.18),
    Sphere(np.array([1.1,0.3,-2.8], dtype=np.float64), 0.11),
  ]
  cubes = [Cube(np.array([-1.2,-1.2,-2], dtype=np.float64), np.array([0.5, 0.5, 0.5], dtype=np.float64))]

  gravity_accel = np.array([-1,0,0])
  for _ in range(simul_steps):
    for sp in spheres:
      sp.velocity += gravity_accel * (1/60)
      sp.center = sp.center + sp.velocity
      if sp.center[0] < -2:
        sp.velocity[0] *= -1
        sp.center[0] = -1.9999

  #env = CompositeShape([s1, s2, c1])
  env = CompositeShape(numba.typed.List(spheres), numba.typed.List(cubes))

  PCORES: int = 4 # this is for parallel but not works well LOL
  for thread_id in numba.prange(PCORES):
    for w in range(thread_id, W, PCORES):
      #if not ((w % PCORES) == thread_id): continue

      for h in range(H):
        # Compute pixel position from pixel index (w, h)
        pixel_pos = np.array([(w + 0.5) / W - 0.5, (h + 0.5) / H - 0.5, 0], dtype=np.float64)

        # Generate a ray originated form camera center to pixel position
        ray = Ray(camera_center, normalize(pixel_pos - camera_center))
        
        # Perform intersection test with environment
        intersect, info = env.intersect(ray)
        img[w,h] += 1 / info.t
  #return img

if __name__ == '__main__':
  print('compile...')
  main(np.zeros((4, 4), dtype=np.float64), 0)

  W = H = 1080
  img = np.zeros((W, H), dtype=np.float64)

  print('running...')
  t = time.time()
  main(img, 0)
  print('took', time.time() - t, 'sec')

  plt.figure(figsize=(10,10))
  plt.imshow(img, origin='lower')
  plt.colorbar()
  plt.savefig('./preview.png', dpi=320)

  print('running video...')
  t = time.time()

  w = round(W)
  h = round(H)
  fps = 30
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')

  out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

  def render(device, tqdm_position, i):
    img = np.zeros((W, H), dtype=np.float64)
    main(img, i)
    frame = (img - np.min(img))
    frame = frame / np.max(frame)
    frame = np.clip(frame, 0, 1)
    frame = frame * 255
    frame = frame.astype(np.uint8)
    frame = cv2.flip(frame, 0)
    res = cv2.applyColorMap(frame, cv2.COLORMAP_COOL)
    return (i, res)

  if not out.isOpened():
    print('File open failed!')
  else:
    for i in tqdm.tqdm(range(150)):
      _, res = render(0, 0, i)
      out.write(res)

  out.release()
  print('took', time.time() - t, 'sec')