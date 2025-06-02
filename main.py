import numpy as np
import random
import time

import display

G = 10

class Particle:
    def __init__(self, x, vel):
        self.x = x
        self.vel = vel
        self.f_net = np.array([0.0,0.0,0.0])
    
class Simulation:
    def __init__(self, num_particles, length):
        self.particles = [Particle(np.array([random.normalvariate(0, 160), random.normalvariate(0, 160), 0]), np.array([0.0,0.0,0.0])) for _ in range(num_particles)]

        display.init()
        
        for tick_id in range(length):
            self.tick()
            time.sleep(0.01)
            if (tick_id % 5) == 0:
                self.display()

    def tick(self):
        self.compute_forces()
        self.adjust_vel()
        self.step()

    def compute_forces(self):
        for a in self.particles:
            a.f_net = np.array([0.0,0.0,0.0])
            for b in self.particles:
                if a != b:
                    dir = b.x - a.x
                    r = np.linalg.norm(dir)
                    if r >= 100:
                        a.f_net += dir*(G/pow(r,2))

    def adjust_vel(self):
        # integral of a over delta_t is approximately v*delta_t
        for particle in self.particles:
            particle.vel += particle.f_net

    def step(self):
        for particle in self.particles:
            particle.x += particle.vel
    
    def display(self):
        coords = []
        for particle in self.particles:
            #coords.append(particle.x.tolist())
            coords.append([particle.x[0], particle.x[1]])
        print("coords:", coords, "\n")
        display.update(coords)

sim = Simulation(50, 4000)