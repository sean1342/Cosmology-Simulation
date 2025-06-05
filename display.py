# display.py using ModernGL and Pyglet

import pyglet
import moderngl
import numpy as np
from pyrr import Matrix44, Vector3, vector # Using pyrr for vector/matrix math

# --- Constants ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
POINT_COLOR_TUPLE = (1.0, 1.0, 1.0)  # White (r,g,b) floats 0-1
BG_COLOR_TUPLE = (0.0, 0.0, 0.0, 1.0)    # Black (r,g,b,a)
POINT_SIZE_RENDER = 5.0 # Corresponds to gl_PointSize in shader

# World coordinate bounds (for initial camera setup)
# Define how large the typical simulation space is for camera framing.
# E.g., if points are mostly within -500 to 500 on each axis.
WORLD_VIEW_RADIUS = 500.0

# Global renderer instance
_renderer_instance = None

class ModernGLRenderer(pyglet.window.Window):
    def __init__(self, width, height, title='ModernGL 3D N-Body'):
        # Request an OpenGL 3.3 core profile window
        config = pyglet.gl.Config(major_version=3, minor_version=3, double_buffer=True, depth_size=24)
        try:
            super().__init__(width, height, title, resizable=True, config=config)
        except pyglet.window.NoSuchConfigException:
            print("Warning: Could not get OpenGL 3.3 Core context. Trying default.")
            super().__init__(width, height, title, resizable=True) # Fallback

        try:
            self.ctx = moderngl.create_context()
        except Exception as e:
            print(f"Failed to create ModernGL context: {e}")
            raise

        self.point_color_vec3 = POINT_COLOR_TUPLE
        self.point_render_size = POINT_SIZE_RENDER
        
        # Shaders
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330 core
                uniform mat4 mvp;
                uniform float point_size_vs;
                in vec3 in_position;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                    gl_PointSize = point_size_vs;
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
        self.camera_yaw = 45.0  # Degrees around Y
        self.camera_pitch = -30.0 # Degrees around X (local)
        
        self.fov = 45.0
        self.near_plane = 0.1
        self.far_plane = WORLD_VIEW_RADIUS * 10.0
        
        self.model_matrix = Matrix44.identity(dtype='f4')
        self.view_matrix = Matrix44.identity(dtype='f4')
        self.projection_matrix = Matrix44.identity(dtype='f4')
        self._update_camera_matrices() # Initial calculation

        # Mouse interaction state
        self.mouse_dragging_rotate = False
        self.mouse_dragging_pan = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.pan_sensitivity = 0.002 # Adjust as needed
        self.rotate_sensitivity = 0.4
        self.zoom_sensitivity = 0.05 # Percentage of current distance

        # Data and VBO/VAO
        self.num_points = 0
        # Reserve space for a moderate number of points. This will grow if needed.
        initial_point_capacity = 100 
        self.vbo_capacity_points = initial_point_capacity
        self.vbo = self.ctx.buffer(reserve=initial_point_capacity * 3 * 4) # 3 floats/point, 4 bytes/float
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f', 'in_position')])

        pyglet.gl.glClearColor(*BG_COLOR_TUPLE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE) # Allow gl_PointSize in vertex shader

    def _calculate_camera_position_and_up(self):
        # Calculate camera position based on target, distance, yaw, and pitch
        rad_yaw = np.radians(self.camera_yaw)
        rad_pitch = np.radians(self.camera_pitch)

        cam_x = self.camera_target.x + self.camera_distance * np.cos(rad_pitch) * np.sin(rad_yaw)
        cam_y = self.camera_target.y + self.camera_distance * np.sin(rad_pitch)
        cam_z = self.camera_target.z + self.camera_distance * np.cos(rad_pitch) * np.cos(rad_yaw)
        
        camera_position = Vector3([cam_x, cam_y, cam_z])

        # Determine the 'up' vector based on camera orientation to avoid issues
        # This creates a 'rolled' up vector if yawing around
        # More robust 'up' vector calculation for orbit camera:
        # Start with world up
        world_up = Vector3([0.0, 1.0, 0.0])
        
        # Forward vector
        forward = vector.normalize(self.camera_target - camera_position)
        # Right vector
        right = vector.normalize(np.cross(forward, world_up))
        # Recalculate local up vector
        camera_up = vector.normalize(np.cross(right, forward))
        
        # Handle cases where forward is aligned with world_up (looking straight up/down)
        if abs(np.dot(forward, world_up)) > 0.999: # gimbal lock / looking straight up or down
            # Use a rotated right vector as 'up' effectively using camera's local X as 'up'
            # This can be based on yaw:
            camera_up = Vector3([np.cos(rad_yaw), 0.0, -np.sin(rad_yaw)])


        return camera_position, camera_up


    def _update_camera_matrices(self):
        camera_position, camera_up = self._calculate_camera_position_and_up()

        self.view_matrix = Matrix44.look_at(
            eye=camera_position,
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
            if processed_coords_np.size == 0: # Empty input is valid
                 self.num_points = 0
            else:
                print(f"Display Error: Coords have unexpected shape: {processed_coords_np.shape}. Expected (N, 3) or flat list multiple of 3. Not updating points.")
                self.num_points = 0
            return 

        if self.num_points > 0:
            required_bytes = len(data_bytes)
            if self.num_points > self.vbo_capacity_points or required_bytes > self.vbo.size :
                # print(f"Display Info: Resizing VBO for {self.num_points} points.")
                self.vbo.release()
                self.vao.release() 
                
                self.vbo_capacity_points = self.num_points + 100 # Add some buffer
                self.vbo = self.ctx.buffer(reserve=self.vbo_capacity_points * 3 * 4)
                self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '3f', 'in_position')])
            
            self.vbo.write(data_bytes)
        # If num_points is 0, on_draw will handle not rendering.

    def on_draw(self):
        # This method is called by pyglet when the window needs to be redrawn
        self.ctx.clear(color=BG_COLOR_TUPLE[:3], depth=1.0) # Clear color and depth

        if self.num_points == 0:
            return 

        self._update_camera_matrices() # Update matrices based on current camera state

        mvp = self.projection_matrix * self.view_matrix * self.model_matrix
        try:
            self.prog['mvp'].write(mvp.tobytes())
            self.prog['point_color_fs'].value = self.point_color_vec3
            self.prog['point_size_vs'].value = self.point_render_size
        except moderngl.Error as e:
            print(f"ModernGL Uniform Error: {e}") # Catch issues if uniforms are not found
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
            self.camera_pitch = max(-89.9, min(89.9, self.camera_pitch)) # Clamp pitch

        elif self.mouse_dragging_pan:
            # Calculate camera's local right and up vectors for panning
            camera_pos, current_cam_up = self._calculate_camera_position_and_up()
            forward_vec = vector.normalize(self.camera_target - camera_pos)
            right_vec = vector.normalize(np.cross(forward_vec, current_cam_up))
            # Pan amount scaled by distance to make it feel more natural
            pan_scale = self.camera_distance * self.pan_sensitivity
            pan_x_vec = -right_vec * dx * pan_scale
            pan_y_vec = current_cam_up * dy * pan_scale # dy is often inverted from screen coords
            
            self.camera_target += pan_x_vec + pan_y_vec
            # Camera position also moves with the target for panning
            # self.camera_pos += pan_x_vec + pan_y_vec # This would be if camera_pos is independent
                                                    # For orbit camera, target moves, pos recalculates.

        self.last_mouse_x, self.last_mouse_y = x, y
        # Matrices will be updated in on_draw

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        # Zoom by changing camera distance
        zoom_amount = self.camera_distance * self.zoom_sensitivity * scroll_y
        self.camera_distance -= zoom_amount
        self.camera_distance = max(self.near_plane * 2.0, self.camera_distance) # Prevent going too close

    def on_resize(self, width, height):
        if height == 0: height = 1 # Avoid division by zero
        self.ctx.viewport = (0, 0, width, height)
        # Projection matrix aspect ratio will be updated in _update_camera_matrices called by on_draw.

    def run_events_and_draw_frame(self):
        """Called by external loop to process events and draw one frame."""
        if self.has_exit: # Check if window has been closed
            return False # Indicate to external loop that we should stop

        self.dispatch_events()  # Process pyglet events (updates mouse state, calls handlers)
        
        # Manually trigger on_draw and swap buffers
        # This ensures rendering happens when called from an external loop
        if self.visible:
            self.dispatch_event('on_draw') # Process any drawing
            self.flip() # Swap buffers
        return True


def init():
    """Initializes the ModernGL display window and renderer."""
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
    # This init does not call pyglet.app.run()

def update(coords):
    """
    Updates the display with new coordinates and renders a frame.
    Called from the main simulation loop.
    Returns False if the display window has been closed.
    """
    if _renderer_instance:
        if _renderer_instance.has_exit: # Check if user closed the window
            return False 
        _renderer_instance.set_coords(coords)
        return _renderer_instance.run_events_and_draw_frame()
    else:
        print("Display Error: ModernGL renderer not initialized. Call init() first.", file=sys.stderr)
        return False # Cannot continue if not initialized

def cleanup():
    """ Optional: Cleans up resources if needed. """
    global _renderer_instance
    if _renderer_instance:
        # Pyglet window closes, ModernGL context should be released with it or explicitly
        if hasattr(_renderer_instance.ctx, 'release'):
             _renderer_instance.ctx.release()
        print("Display: Renderer cleaned up.")
        _renderer_instance = None


# --- Example Standalone Usage (for testing display.py itself) ---
if __name__ == '__main__':
    print("Running display.py standalone test...")
    init()

    if _renderer_instance:
        num_test_points = 100
        # Initial test coordinates: random points in a sphere
        phi = np.random.uniform(0, np.pi, num_test_points)
        theta = np.random.uniform(0, 2 * np.pi, num_test_points)
        radius = np.random.uniform(WORLD_VIEW_RADIUS * 0.1, WORLD_VIEW_RADIUS * 0.8, num_test_points)
        
        test_coords = np.zeros((num_test_points, 3), dtype='f4')
        test_coords[:,0] = radius * np.sin(phi) * np.cos(theta) # x
        test_coords[:,1] = radius * np.sin(phi) * np.sin(theta) # y
        test_coords[:,2] = radius * np.cos(phi)                   # z

        # Store angular velocities for animation
        angular_velocities = np.random.uniform(-60, 60, num_test_points) # degrees per second

        def animate_for_standalone(dt):
            global test_coords
            # Rotate points around Y axis at different speeds for a simple visual
            for i in range(num_test_points):
                angle_rad = np.radians(angular_velocities[i] * dt)
                rot_matrix = Matrix44.from_y_rotation(angle_rad, dtype='f4')
                point_vec4 = Vector3(test_coords[i,:]) 
                # pyrr matrix multiplication with vector needs homogeneous coord if vector is on right
                # or, handle as 3x3 rotation matrix on vec3.
                # For Matrix44 * Vector3 (position), pyrr does M * [v.x, v.y, v.z, 1.0]
                rotated_point = matrix44.apply_to_vector(rot_matrix, point_vec4)
                test_coords[i,:] = rotated_point[:3]
            
            if not update(test_coords): # update() now returns False if window closed
                pyglet.app.exit() # Exit the pyglet app loop

        if not (hasattr(pyglet.app, 'event_loop') and pyglet.app.event_loop.is_running):
            print("Display: Using pyglet.app.run() for standalone test.")
            pyglet.clock.schedule_interval(animate_for_standalone, 1/60.0)
            try:
                pyglet.app.run()
            except KeyboardInterrupt:
                print("Display: Exiting standalone test via KeyboardInterrupt.")
            finally:
                cleanup() # Ensure cleanup on exit
        else:
            print("Display: Pyglet app seems to be running or managed externally. Standalone test might not behave as expected without app.run().")
            # For an already running loop (e.g. in an IDE with pyglet integration), scheduling might be enough.
            # However, the init/update is designed for an external Python loop (like user's main.py).
    else:
        print("Display: Failed to initialize renderer for standalone test.")

    print("Display: Standalone test finished.")