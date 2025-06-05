# display.py using ModernGL and Pyglet with "Realistic" Inverse Proportional Distance Sizing

import pyglet
import moderngl
import numpy as np
from pyrr import Matrix44, Vector3, vector # Using pyrr for vector/matrix math

# --- Constants ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
POINT_COLOR_TUPLE = (1.0, 1.0, 1.0)  # White (r,g,b) floats 0-1
BG_COLOR_TUPLE = (0.0, 0.0, 0.0, 1.0)    # Black (r,g,b,a)

# Default point size settings for inverse proportional scaling
DEFAULT_MODEL_PIXEL_SIZE_AT_REF_DIST = 7.0 # Pixel size of a point at reference_distance
DEFAULT_REFERENCE_DISTANCE = 500.0         # World units distance for model_pixel_size
DEFAULT_MIN_CLAMP_POINT_SIZE = 1.0         # Minimum pixel size
DEFAULT_MAX_CLAMP_POINT_SIZE = 50.0        # Maximum pixel size (for very close points)

# World coordinate bounds (for initial camera setup)
WORLD_VIEW_RADIUS = 500.0

# Global renderer instance
_renderer_instance = None

class ModernGLRenderer(pyglet.window.Window):
    def __init__(self, width, height, title='ModernGL 3D N-Body - Realistic Size'):
        config = pyglet.gl.Config(major_version=3, minor_version=3, double_buffer=True, depth_size=24)
        try:
            super().__init__(width, height, title, resizable=True, config=config)
        except pyglet.window.NoSuchConfigException:
            print("Warning: Could not get OpenGL 3.3 Core context. Trying default.")
            super().__init__(width, height, title, resizable=True)

        try:
            self.ctx = moderngl.create_context()
        except Exception as e:
            print(f"Failed to create ModernGL context: {e}")
            raise

        self.point_color_vec3 = POINT_COLOR_TUPLE
        
        # Point sizing parameters for inverse proportional scaling
        self.model_pixel_size_at_ref_dist = DEFAULT_MODEL_PIXEL_SIZE_AT_REF_DIST
        self.reference_distance = DEFAULT_REFERENCE_DISTANCE
        self.min_clamp_point_size = DEFAULT_MIN_CLAMP_POINT_SIZE
        self.max_clamp_point_size = DEFAULT_MAX_CLAMP_POINT_SIZE
        
        # Shaders
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330 core
                uniform mat4 mvp;
                // Point sizing uniforms for inverse proportional scaling
                uniform float model_pixel_size_at_ref_dist_vs;
                uniform float reference_dist_vs;
                uniform float min_clamp_point_size_vs;
                uniform float max_clamp_point_size_vs;
                uniform vec3 camera_world_pos_vs;    // Camera's world position

                in vec3 in_position; // Assuming in_position is already in world space

                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);

                    float dist_to_camera = distance(in_position, camera_world_pos_vs);
                    
                    // Prevent division by zero or extremely small distances
                    // This effective_min_dist should ideally be related to the near clip plane
                    // or a small fraction of the reference distance to maintain behavior.
                    float effective_min_dist = 0.01 * reference_dist_vs; 
                    float effective_distance = max(dist_to_camera, effective_min_dist);

                    float calculated_size = (model_pixel_size_at_ref_dist_vs * reference_dist_vs) / effective_distance;
                    
                    gl_PointSize = clamp(calculated_size, min_clamp_point_size_vs, max_clamp_point_size_vs);
                }
            ''',
            fragment_shader='''
                #version 330 core
                uniform vec3 point_color_fs;
                out vec4 out_color;
                void main() {
                    out_color = vec4(point_color_fs, 1.0);
                }
            '''
        )

        # Camera parameters (Orbit Camera)
        self.camera_distance = WORLD_VIEW_RADIUS * 2.0
        self.camera_target = Vector3([0.0, 0.0, 0.0])
        self.camera_yaw = 45.0
        self.camera_pitch = -30.0
        self.camera_pos = Vector3([0.0, 0.0, 0.0]) 
        
        self.fov = 45.0
        self.near_plane = 0.1 
        self.far_plane = WORLD_VIEW_RADIUS * 20.0
        
        self.model_matrix = Matrix44.identity(dtype='f4')
        self.view_matrix = Matrix44.identity(dtype='f4')
        self.projection_matrix = Matrix44.identity(dtype='f4')
        self._update_camera_matrices()

        # Mouse interaction state
        self.mouse_dragging_rotate = False
        self.mouse_dragging_pan = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.pan_sensitivity = 0.002 
        self.rotate_sensitivity = 0.4
        self.zoom_sensitivity = 0.05

        # Data and VBO/VAO
        self.num_points = 0
        initial_point_capacity = 100 
        self.vbo_capacity_points = initial_point_capacity
        self.vbo = self.ctx.buffer(reserve=initial_point_capacity * 3 * 4)
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f', 'in_position')])

        pyglet.gl.glClearColor(*BG_COLOR_TUPLE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    def _calculate_camera_position_and_up(self):
        rad_yaw = np.radians(self.camera_yaw)
        rad_pitch = np.radians(self.camera_pitch)

        cam_x = self.camera_target.x + self.camera_distance * np.cos(rad_pitch) * np.sin(rad_yaw)
        cam_y = self.camera_target.y + self.camera_distance * np.sin(rad_pitch)
        cam_z = self.camera_target.z + self.camera_distance * np.cos(rad_pitch) * np.cos(rad_yaw)
        
        self.camera_pos = Vector3([cam_x, cam_y, cam_z])

        world_up = Vector3([0.0, 1.0, 0.0])
        forward = vector.normalize(self.camera_target - self.camera_pos)
        # Handle case where forward and world_up are collinear
        if abs(np.dot(forward, world_up)) > 0.9999: # looking straight up or down
            # Use camera's local "right" vector, derived from yaw, as the up direction for look_at
            # This avoids issues with np.cross(forward, world_up) becoming zero vector
            camera_up = Vector3([-np.sin(rad_yaw), 0.0, np.cos(rad_yaw)]) # Rotated right vector on XZ plane
        else:
            right = vector.normalize(np.cross(forward, world_up))
            camera_up = vector.normalize(np.cross(right, forward))
        
        return camera_up

    def _update_camera_matrices(self):
        camera_up = self._calculate_camera_position_and_up()

        self.view_matrix = Matrix44.look_at(
            eye=self.camera_pos,
            target=self.camera_target,
            up=camera_up,
            dtype='f4'
        )
        aspect_ratio = self.width / self.height if self.height > 0 else 1.0
        self.projection_matrix = Matrix44.perspective_projection(
            fovy=self.fov,
            aspect=aspect_ratio,
            near=self.near_plane,
            far=self.far_plane,
            dtype='f4'
        )

    def set_coords(self, coords):
        if coords is None:
            self.num_points = 0
            return

        processed_coords_np = np.array(coords, dtype='f4')
        
        if processed_coords_np.ndim == 2 and processed_coords_np.shape[1] == 3:
            self.num_points = processed_coords_np.shape[0]
            data_bytes = processed_coords_np.tobytes()
        elif processed_coords_np.ndim == 1 and processed_coords_np.shape[0] % 3 == 0:
            self.num_points = processed_coords_np.shape[0] // 3
            data_bytes = processed_coords_np.tobytes()
        else:
            if processed_coords_np.size == 0:
                 self.num_points = 0
            else:
                print(f"Display Error: Coords shape {processed_coords_np.shape} unexpected. Expected (N, 3) or flat N*3. No points displayed.")
                self.num_points = 0
            return 

        if self.num_points > 0:
            required_bytes = len(data_bytes)
            if self.num_points > self.vbo_capacity_points or required_bytes > self.vbo.size :
                self.vbo.release()
                self.vao.release() 
                self.vbo_capacity_points = self.num_points + (self.num_points // 2) + 10
                self.vbo = self.ctx.buffer(reserve=self.vbo_capacity_points * 3 * 4)
                self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f', 'in_position')])
            self.vbo.write(data_bytes)

    def on_draw(self):
        self.ctx.clear(color=BG_COLOR_TUPLE[:3], depth=1.0)

        if self.num_points == 0:
            return 

        self._update_camera_matrices() 

        mvp = self.projection_matrix * self.view_matrix * self.model_matrix
        try:
            self.prog['mvp'].write(mvp.tobytes())
            self.prog['point_color_fs'].value = self.point_color_vec3
            
            # Write new point sizing uniforms
            self.prog['model_pixel_size_at_ref_dist_vs'].value = float(self.model_pixel_size_at_ref_dist)
            self.prog['reference_dist_vs'].value = float(self.reference_distance)
            self.prog['min_clamp_point_size_vs'].value = float(self.min_clamp_point_size)
            self.prog['max_clamp_point_size_vs'].value = float(self.max_clamp_point_size)
            self.prog['camera_world_pos_vs'].write(self.camera_pos.astype('f4').tobytes())

        except moderngl.Error as e:
            print(f"ModernGL Uniform Error: {e}")
            return
        except KeyError as e: # This helps catch misspelled uniform names
            print(f"ModernGL Uniform KeyError: Uniform '{e}' not found in shader. Check names.")
            return
        
        self.vao.render(moderngl.POINTS, vertices=self.num_points)

    def on_mouse_press(self, x, y, button, modifiers):
        self.last_mouse_x, self.last_mouse_y = x, y
        if button == pyglet.window.mouse.LEFT:
            self.mouse_dragging_rotate = True
        elif button == pyglet.window.mouse.RIGHT or button == pyglet.window.mouse.MIDDLE:
            self.mouse_dragging_pan = True

    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            self.mouse_dragging_rotate = False
        elif button == pyglet.window.mouse.RIGHT or button == pyglet.window.mouse.MIDDLE:
            self.mouse_dragging_pan = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.mouse_dragging_rotate:
            self.camera_yaw -= dx * self.rotate_sensitivity
            self.camera_pitch -= dy * self.rotate_sensitivity 
            self.camera_pitch = max(-89.9, min(89.9, self.camera_pitch))

        elif self.mouse_dragging_pan:
            _ , current_cam_up = self._calculate_camera_position_and_up()
            forward_vec = vector.normalize(self.camera_target - self.camera_pos)
            right_vec = vector.normalize(np.cross(forward_vec, current_cam_up))
            
            pan_scale = self.camera_distance * self.pan_sensitivity
            pan_x_vec = -right_vec * dx * pan_scale
            pan_y_vec = current_cam_up * dy * pan_scale
            
            self.camera_target += pan_x_vec + pan_y_vec
        self.last_mouse_x, self.last_mouse_y = x, y

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        zoom_amount = self.camera_distance * self.zoom_sensitivity * scroll_y
        self.camera_distance -= zoom_amount
        self.camera_distance = max(self.near_plane * 2.0, self.camera_distance) # Prevent zoom too close

    def on_resize(self, width, height):
        if height == 0: height = 1
        # Use get_framebuffer_size for HiDPI/Retina displays
        fb_width, fb_height = self.get_framebuffer_size()
        self.ctx.viewport = (0, 0, fb_width, fb_height)

    def run_events_and_draw_frame(self):
        if self.has_exit:
            return False 

        self.dispatch_events()  
        
        if self.visible:
            self.dispatch_event('on_draw')
            self.flip()
        return True

def init():
    global _renderer_instance
    if _renderer_instance is None:
        try:
            _renderer_instance = ModernGLRenderer(WINDOW_WIDTH, WINDOW_HEIGHT)
            print("Display: ModernGL Renderer Initialized with Pyglet.")
        except Exception as e:
            print(f"Display Error: Failed to initialize ModernGLRenderer: {e}")
            import traceback
            traceback.print_exc()
            _renderer_instance = None 
    return _renderer_instance is not None

def update(coords):
    if _renderer_instance:
        if _renderer_instance.has_exit: 
            return False 
        _renderer_instance.set_coords(coords)
        return _renderer_instance.run_events_and_draw_frame()
    else:
        print("Display Error: ModernGL renderer not initialized. Call init() first.", file=sys.stderr)
        return False

def cleanup():
    global _renderer_instance
    if _renderer_instance:
        try:
            if hasattr(_renderer_instance.ctx, 'release') and _renderer_instance.ctx:
                 _renderer_instance.ctx.release()
            if not _renderer_instance.has_exit:
                _renderer_instance.close()
            print("Display: Renderer cleaned up.")
        except Exception as e:
            print(f"Display Error during cleanup: {e}")
        _renderer_instance = None

# --- Example Standalone Usage ---
if __name__ == '__main__':
    print("Running display.py standalone test...")
    if not init():
        print("Display: Initialization failed. Exiting standalone test.")
        exit()

    # Modify default sizing parameters for testing if desired from your main simulation
    # For example:
    # if _renderer_instance:
    #    _renderer_instance.model_pixel_size_at_ref_dist = 10.0
    #    _renderer_instance.reference_distance = WORLD_VIEW_RADIUS * 0.5 
    #    _renderer_instance.min_clamp_point_size = 2.0
    #    _renderer_instance.max_clamp_point_size = 100.0


    num_test_points = 200
    test_coords_list = []
    # Create points on several concentric shells to clearly see distance effect
    shell_radii = [WORLD_VIEW_RADIUS * r for r in [0.2, 0.5, 1.0, 1.5, 2.0]]
    points_per_shell = num_test_points // len(shell_radii)

    for r_shell in shell_radii:
        for _ in range(points_per_shell):
            phi = np.random.uniform(0, np.pi)       # inclination
            theta = np.random.uniform(0, 2 * np.pi) # azimuth
            x = r_shell * np.sin(phi) * np.cos(theta)
            y = r_shell * np.sin(phi) * np.sin(theta)
            z = r_shell * np.cos(phi)
            test_coords_list.append([x,y,z])
    
    test_coords = np.array(test_coords_list, dtype='f4')
    static_test_coords = test_coords.copy() # Keep a copy for static display if not animating

    animation_active = True # Set to False to see static points
    angular_velocity_deg_s = 15.0 

    def animate_for_standalone(dt):
        global test_coords
        if animation_active:
            angle_rad = np.radians(angular_velocity_deg_s * dt)
            rot_matrix = Matrix44.from_y_rotation(angle_rad, dtype='f4')
            
            original_shape = test_coords.shape
            if original_shape[0] > 0: # Ensure there are points
                homogeneous_coords = np.hstack((test_coords, np.ones((original_shape[0], 1), dtype='f4')))
                # pyrr matrix multiplication can be tricky with broadcasting.
                # (matrix @ homogeneous_coords.T).T is common for (N,D) arrays.
                rotated_homogeneous = np.dot(homogeneous_coords, rot_matrix.T) # M*v (if v is row) or (v*M_T).T
                                                                               # For pyrr: rot_matrix * vector
                                                                               # Let's apply to each vector
                new_coords = np.empty_like(test_coords)
                for i in range(test_coords.shape[0]):
                    new_coords[i] = matrix44.apply_to_vector(rot_matrix, Vector3(test_coords[i,:]))[:3]
                test_coords = new_coords

        if not update(test_coords if animation_active else static_test_coords):
            pyglet.app.exit()

    if not (hasattr(pyglet.app, 'event_loop') and pyglet.app.event_loop.is_running):
        print("Display: Using pyglet.app.run() for standalone test.")
        pyglet.clock.schedule_interval(animate_for_standalone, 1/60.0)
        try:
            pyglet.app.run()
        except KeyboardInterrupt:
            print("Display: Exiting standalone test via KeyboardInterrupt.")
        finally:
            cleanup()
    else:
        print("Display: Pyglet app seems to be already running. Standalone animation might not work as expected.")

    print("Display: Standalone test finished.")