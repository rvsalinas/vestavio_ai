import logging

# -------------------------------------------------------------------
# CAUTION: Rename this file to avoid overshadowing the real library!
# -------------------------------------------------------------------

_initialized = False

# Mock CPU backend identifier
cpu = "cpu"
gpu = "gpu"

def init(backend):
    global _initialized
    if backend == cpu:
        logging.debug("Mock Genesis: Initializing with CPU backend.")
    elif backend == gpu:
        logging.debug("Mock Genesis: Initializing with GPU backend.")
    else:
        logging.debug(f"Mock Genesis: Initializing with '{backend}' backend.")

    _initialized = True


# ------------------------ Options & Configs ------------------------
class options:
    class SimOptions:
        def __init__(self, dt=0.01, substeps=1, gravity=(0, 0, -9.8)):
            self.dt = dt
            self.substeps = substeps
            self.gravity = gravity

    class ViewerOptions:
        def __init__(self, camera_pos=(0, 0, 0), camera_lookat=(0, 0, 0),
                     camera_fov=40, max_FPS=60):
            self.camera_pos = camera_pos
            self.camera_lookat = camera_lookat
            self.camera_fov = camera_fov
            self.max_FPS = max_FPS


# ------------------------ Morphs (Mock Entities) ------------------------
class morphs:
    class URDF:
        def __init__(self, file, collision=True, scale=1.0, pos=(0, 0, 0), euler=(0, 0, 0)):
            self.file = file
            self.collision = collision
            self.scale = scale
            self.pos = pos
            self.euler = euler

        def __repr__(self):
            return f"MockURDF(file={self.file}, collision={self.collision}, scale={self.scale})"

    class MJCF:
        def __init__(self, file, collision=False, scale=1.0, pos=(0, 0, 0), euler=(0, 0, 0)):
            self.file = file
            self.collision = collision
            self.scale = scale
            self.pos = pos
            self.euler = euler

        def __repr__(self):
            return f"MockMJCF(file={self.file}, collision={self.collision}, scale={self.scale})"


# ------------------------ Scene (Mock Simulation) ------------------------
class Scene:
    def __init__(self, sim_options=None, viewer_options=None, show_viewer=False):
        self.sim_options = sim_options
        self.viewer_options = viewer_options
        self.show_viewer = show_viewer
        self.entities = []

        logging.debug("Mock Scene initialized with viewer & simulation options.")

    def add_entity(self, morph):
        self.entities.append(morph)
        logging.debug(f"Mock Scene: Entity added -> {morph}")
        return morph

    def build(self, compile_kernels=False):
        logging.debug("Mock Scene: Building scene...")
        # No actual build, just logs
        logging.debug("Mock Scene: Build complete.")

    def step(self):
        logging.debug("Mock Scene: Stepping the simulation.")
        # No actual physics, just logs

    def close(self):
        logging.debug("Mock Scene: Closing the scene.")
        # No actual teardown, just logs