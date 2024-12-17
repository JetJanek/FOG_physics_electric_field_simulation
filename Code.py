import matplotlib.pyplot as plt
import numpy as np
import time
import random
from matplotlib.animation import FuncAnimation

class Vector(list):
    def __init__(self, *el):
        for e in el:
            self.append(e)

    def __add__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] + other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self + other

    def __sub__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] - other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self - other

    def __mul__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] * other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self * other

    def __truediv__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] / other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self / other

    def __pow__(self, other):
        if type(other) is Vector:
            assert len(self) == len(other), "Error 0"
            r = Vector()
            for i in range(len(self)):
                r.append(self[i] ** other[i])
            return r
        else:
            other = Vector.emptyvec(lens=len(self), n=other)
            return self ** other

    def __mod__(self, other):
        return sum((self - other) ** 2) ** 0.5

    def mod(self):
        return self % Vector.emptyvec(len(self))

    def dim(self):
        return len(self)
    
    def __str__(self):
        if len(self) == 0:
            return "Empty"
        r = [str(i) for i in self]
        return "< " + " ".join(r) + " >"
    
    def _ipython_display_(self):
        print(str(self))

    @staticmethod
    def emptyvec(lens=2, n=0):
        return Vector(*[n for i in range(lens)])

    @staticmethod
    def randvec(dim):
        return Vector(*[random.random() for i in range(dim)])


class Point:
    def __init__(self, coords, mass=1.0, q=1.0, speed=None, **properties):
        self.coords = coords
        if speed is None:
            self.speed = Vector(*[0 for i in range(len(coords))])
        else:
            self.speed = speed
        self.acc = Vector(*[0 for i in range(len(coords))])
        self.mass = mass
        self.__params__ = ["coords", "speed", "acc", "q"] + list(properties.keys())
        self.q = q
        for prop in properties:
            setattr(self, prop, properties[prop])

    def move(self, dt):
        self.coords = self.coords + self.speed * dt

    def accelerate(self, dt):
        self.speed = self.speed + self.acc * dt

    def accinc(self, force):
        self.acc = self.acc + force / self.mass
    
    def clean_acc(self):
        self.acc = self.acc * 0
    
    def __str__(self):
        r = ["Point {"]
        for p in self.__params__:
            r.append("  " + p + " = " + str(getattr(self, p)))
        r += ["}"]
        return "\n".join(r)

    def _ipython_display_(self):
        print(str(self))


class InteractionField:
    def __init__(self, F):
        self.points = []
        self.F = F

    def move_all(self, dt):
        for p in self.points:
            p.move(dt)

    def intensity(self, coord):
        proj = Vector(*[0 for i in range(coord.dim())])
        single_point = Point(Vector(), mass=1.0, q=1.0)
        for p in self.points:
            if coord % p.coords < 10 ** (-10):
                continue
            d = p.coords % coord
            fmod = self.F(single_point, p, d) * (-1)
            proj = proj + (coord - p.coords) / d * fmod
        return proj

    def step(self, dt):
        self.clean_acc()
        for p in self.points:
            p.accinc(self.intensity(p.coords) * p.q)
            p.accelerate(dt)
            p.move(dt)

    def clean_acc(self):
        for p in self.points:
            p.clean_acc()
    
    def append(self, *args, **kwargs):
        self.points.append(Point(*args, **kwargs))
    
    def gather_coords(self):
        return [p.coords for p in self.points] 




epsilon = 0.5
F_max = 10000
u = InteractionField(lambda p1, p2, r: min(300000 * -p1.q * p2.q / max(r ** 2, epsilon), F_max))

for i in range(10):
    u.append(Vector.randvec(2) * 10, q=random.random() - 0.5, speed=Vector.randvec(2) * 0)

# Visualisation setup
STEP = 0.25
fig, ax = plt.subplots(figsize=(6, 6))

# Creating a grid for vector field
x_vals = np.arange(0, 10, STEP)
y_vals = np.arange(0, 10, STEP)
res = []

quiver = None
particles, = plt.plot([], [], 'ro', markersize=5)

def sigm(x):
    return 1 / (1 + np.exp(-x))

def calculate_potential(coord):
    potential = 0
    for p in u.points:
        d = max(coord % p.coords, epsilon)
        potential += p.q / d
    return potential

# Creating a grid for potential
X, Y = np.meshgrid(x_vals, y_vals)
potential_grid = np.zeros(X.shape)

# Creating a color map
heatmap = ax.imshow(potential_grid, extent=[0, 10, 0, 10], origin='lower', cmap='viridis', alpha=0.7)

def update(frame):
    global u, quiver, heatmap

    # Particles coord update
    u.step(0.0005)

    # Generating a vector field
    u_vectors_x = []
    u_vectors_y = []
    u_intensity = []

    for x in x_vals:
        for y in y_vals:
            inten = u.intensity(Vector(x, y))
            inten /= max(inten.mod(), 1e-5)
            u_vectors_x.append(inten[0])
            u_vectors_y.append(inten[1])
            u_intensity.append(inten.mod())

    # Deleting an old vector field
    if quiver:
        quiver.remove()

    # Generating a new vector field
    quiver = ax.quiver(
        np.repeat(x_vals, len(y_vals)),
        np.tile(y_vals, len(x_vals)),
        u_vectors_x, u_vectors_y,
        color=[(sigm(i), 0.5, 0.8 * (1 - sigm(i))) for i in u_intensity],
        angles='xy', scale_units='xy', scale=1.5
    )

    # Updating color map of potential
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            potential_grid[i, j] = calculate_potential(Vector(X[i, j], Y[i, j]))

    # Normlization of potential
    normalized_potential = (potential_grid - potential_grid.min()) / (potential_grid.max() - potential_grid.min())

    heatmap.set_data(normalized_potential)  
    heatmap.set_clim(0, 1)  

    # Updating coords of particles
    coords = np.array([p.coords for p in u.points])
    particles.set_data(coords[:, 0], coords[:, 1])

    return quiver, particles, heatmap


# Abimation setup
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)

plt.show()
