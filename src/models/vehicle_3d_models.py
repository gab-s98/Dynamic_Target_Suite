"""
vehicle_3d_models.py
====================
Parametric, solid volumetric 3D OpenGL geometry models for Dynamic Target Suite.
Contains:
1. FSAECar3DModel: Classic Formula Student / FSAE racing car (clean, streamlined, high-FPS).
2. SUVCar3DModel: B-Segment compact road SUV (clean, solid boxy 'a rettangoloni' architecture).

Reference frame:
  +X: Forward (towards front bumper)
  +Y: Left side
  +Z: Upwards (sky)
  Z = 0.0: Ground plane
"""

import os
import math
import numpy as np

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False


# =============================================================================
# HELPER FUNCTIONS FOR 3D MESH AND GEOMETRY GENERATION
# =============================================================================

def find_fsae_obj_file():
    """Searches for an external FSAE car mesh in standard directories."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "models", "fsae_car.obj"),
        os.path.join(os.path.dirname(__file__), "assets", "fsae_car.obj"),
        os.path.join(os.path.dirname(__file__), "fsae_car.obj"),
        os.path.join(os.getcwd(), "models", "fsae_car.obj"),
        os.path.join(os.getcwd(), "assets", "fsae_car.obj"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def parse_obj_mesh(filepath: str):
    """
    Parses a Wavefront .OBJ file into vertices, normals, and triangulated faces.
    Supports comments, vertices (v), vertex normals (vn), and polygonal faces (f).
    Triangulates polygons via triangle fans and computes facet normals if missing.
    """
    vertices = []
    normals = []
    triangles = []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                cmd = parts[0].lower()
                if cmd == 'v':
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif cmd == 'vn':
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif cmd == 'f':
                    face_verts = []
                    face_norms = []
                    for tok in parts[1:]:
                        vals = tok.split('/')
                        vi = int(vals[0])
                        vi = vi - 1 if vi > 0 else len(vertices) + vi
                        face_verts.append(vi)
                        if len(vals) >= 3 and vals[2]:
                            ni = int(vals[2])
                            ni = ni - 1 if ni > 0 else len(normals) + ni
                            face_norms.append(ni)
                        else:
                            face_norms.append(None)

                    for i in range(1, len(face_verts) - 1):
                        v_tri = (face_verts[0], face_verts[i], face_verts[i + 1])
                        n_tri = (face_norms[0], face_norms[i], face_norms[i + 1])
                        triangles.append((v_tri, n_tri))

        processed_triangles = []
        for (v0_idx, v1_idx, v2_idx), (n0_idx, n1_idx, n2_idx) in triangles:
            p0 = vertices[v0_idx]
            p1 = vertices[v1_idx]
            p2 = vertices[v2_idx]

            if n0_idx is not None and n1_idx is not None and n2_idx is not None:
                n0 = normals[n0_idx]
                n1 = normals[n1_idx]
                n2 = normals[n2_idx]
            else:
                u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
                v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]]
                nx = u[1] * v[2] - u[2] * v[1]
                ny = u[2] * v[0] - u[0] * v[2]
                nz = u[0] * v[1] - u[1] * v[0]
                length = math.sqrt(nx * nx + ny * ny + nz * nz)
                if length > 1e-7:
                    nx /= length
                    ny /= length
                    nz /= length
                else:
                    nx, ny, nz = 0.0, 0.0, 1.0
                n0 = n1 = n2 = [nx, ny, nz]

            processed_triangles.append(((p0, p1, p2), (n0, n1, n2)))

        return {
            'filepath': filepath,
            'vertex_count': len(vertices),
            'triangle_count': len(processed_triangles),
            'triangles': processed_triangles
        }
    except Exception as e:
        print(f"Warning: Failed to parse OBJ file {filepath}: {e}")
        return None


_TUBE_N_SIDES = 6
_TUBE_ANGLES = [2.0 * math.pi * i / _TUBE_N_SIDES for i in range(_TUBE_N_SIDES + 1)]
_TUBE_COS = [math.cos(a) for a in _TUBE_ANGLES]
_TUBE_SIN = [math.sin(a) for a in _TUBE_ANGLES]


def _draw_tube_strut(p1, p2, radius: float = 0.008, n_sides: int = 6, col=None):
    """
    Renders a solid 3D regular prism cylinder (tube/strut) connecting p1 to p2.
    Uses precomputed trigonometric tables and GL_QUAD_STRIP for maximum GPU performance.
    """
    if not HAS_OPENGL:
        return
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-6:
        return

    ax, ay, az = dx / length, dy / length, dz / length

    if abs(ax) < 0.9:
        rx, ry, rz = 1.0, 0.0, 0.0
    else:
        rx, ry, rz = 0.0, 1.0, 0.0

    ux = ay * rz - az * ry
    uy = az * rx - ax * rz
    uz = ax * ry - ay * rx
    ul = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / ul, uy / ul, uz / ul

    vx = ay * uz - az * uy
    vy = az * ux - ax * uz
    vz = ax * uy - ay * ux

    if col is not None:
        glColor3f(*col)

    if n_sides == 6:
        cos_tab, sin_tab = _TUBE_COS, _TUBE_SIN
        n_pts = 7
    else:
        angles = [2.0 * math.pi * i / n_sides for i in range(n_sides + 1)]
        cos_tab = [math.cos(a) for a in angles]
        sin_tab = [math.sin(a) for a in angles]
        n_pts = n_sides + 1

    glBegin(GL_QUAD_STRIP)
    for i in range(n_pts):
        ca, sa = cos_tab[i], sin_tab[i]
        nx = ca * ux + sa * vx
        ny = ca * uy + sa * vy
        nz = ca * uz + sa * vz
        glNormal3f(nx, ny, nz)
        cx, cy, cz = radius * nx, radius * ny, radius * nz
        glVertex3f(p1[0] + cx, p1[1] + cy, p1[2] + cz)
        glVertex3f(p2[0] + cx, p2[1] + cy, p2[2] + cz)
    glEnd()


# =============================================================================
# 1. CLASSIC FORMULA STUDENT (FSAE) RACING CAR 3D MODEL
# =============================================================================
class FSAECar3DModel:
    """
    High-fidelity, Low-Poly (3,000-8,000 tris) 3D model for Formula Student / FSAE single-seater.
    Features:
    - External Wavefront .obj support with transparent procedural fallback.
    - Solid Hoosier 16x7.5-10 wheels (profiled tread, rounded shoulders, recessed rim, 5 alloy spokes, rotor, caliper).
    - True 3D volumetric tubular suspension wishbones (upper/lower A-arms, pushrod, tie-rod).
    - Multi-element front & rear wings with sculpted endplates, footplates, and swan-neck pylons.
    - Hollow cockpit with bucket seat, 5-point harness safety belts, headrest, and 3D tubular roll hoop.
    - FSAE butterfly steering wheel with integrated LCD racing dash, rotating dynamically with steer angle.
    - Precompiled OpenGL Display List (body_display_list) for rock-solid 120+ FPS.
    """

    def __init__(self):
        # Default livery colors
        self.col_body = (0.024, 0.322, 0.867)        # Formula Blue
        self.col_carbon = (0.098, 0.118, 0.141)      # Carbon Fiber Black
        self.col_carbon_light = (0.16, 0.18, 0.20)  # Carbon Accent Panel
        self.col_accent = (0.906, 0.188, 0.188)      # Racing Red Accent
        self.col_titanium = (0.686, 0.714, 0.733)    # Satin Titanium / Pushrods
        self.col_wishbone = (0.16, 0.17, 0.19)      # Matte Black Wishbones
        self.col_helmet = (0.945, 0.769, 0.059)      # Gold Helmet
        self.col_alloy = (0.82, 0.84, 0.88)          # Silver Alloy Rim
        self.col_rim_inner = (0.13, 0.14, 0.16)      # Recessed Wheel Well Dark Anthracite
        self.col_tread = (0.12, 0.13, 0.15)          # Hoosier Matte Tread Rubber
        self.col_brake_disc = (0.74, 0.75, 0.78)     # Steel Ventilated Rotor
        self.col_caliper = (0.88, 0.68, 0.10)        # Gold Brembo Caliper
        self.col_seat = (0.07, 0.08, 0.10)           # Carbon Bucket Seat
        self.col_belts = (0.82, 0.12, 0.12)          # Sabelt Red 5-Point Belts
        self.col_buckle = (0.78, 0.80, 0.82)         # Metal Rotary Buckle
        self.col_headrest = (0.12, 0.12, 0.14)       # High-Density Impact Foam
        self.col_dash_screen = (0.04, 0.08, 0.18)    # LCD Screen Navy Blue

        self.body_display_list = None
        self.ghost_display_list = None
        self.wheel_display_list_left = None
        self.wheel_display_list_right = None
        self.caliper_display_list_left = None
        self.caliper_display_list_right = None

        # Check for optional external OBJ mesh
        self.obj_path = find_fsae_obj_file()
        self.obj_mesh = parse_obj_mesh(self.obj_path) if self.obj_path else None
        if self.obj_mesh is not None:
            print(f"[FSAECar3DModel] Loaded external mesh: {self.obj_path} ({self.obj_mesh['triangle_count']} tris)")

    def invalidate_display_list(self):
        if HAS_OPENGL:
            for attr in ['body_display_list', 'ghost_display_list', 'wheel_display_list_left',
                         'wheel_display_list_right', 'caliper_display_list_left', 'caliper_display_list_right']:
                dl = getattr(self, attr, None)
                if dl is not None:
                    try:
                        glDeleteLists(dl, 1)
                    except Exception:
                        pass
                    setattr(self, attr, None)

    def compile_body_display_list(self):
        if not HAS_OPENGL:
            return
        try:
            from PyQt5.QtGui import QOpenGLContext
            if QOpenGLContext.currentContext() is None:
                return
            if self.body_display_list is not None:
                try:
                    glDeleteLists(self.body_display_list, 1)
                except Exception:
                    pass
            self.body_display_list = glGenLists(1)
            glNewList(self.body_display_list, GL_COMPILE)

            if self.obj_mesh is not None:
                self._draw_obj_mesh()
            else:
                self._draw_monocoque()
                self._draw_cockpit_interior()
                self._draw_sidepods()
                self._draw_front_wing()
                self._draw_rear_wing()

            glEndList()
        except Exception:
            self.body_display_list = None

    def compile_ghost_display_list(self):
        if not HAS_OPENGL:
            return
        try:
            from PyQt5.QtGui import QOpenGLContext
            if QOpenGLContext.currentContext() is None:
                return
            if self.ghost_display_list is not None:
                try:
                    glDeleteLists(self.ghost_display_list, 1)
                except Exception:
                    pass
            self.ghost_display_list = glGenLists(1)
            glNewList(self.ghost_display_list, GL_COMPILE)

            glColor4f(0.20, 0.75, 0.95, 0.35)
            if self.obj_mesh is not None:
                self._draw_obj_mesh()
            else:
                self._draw_monocoque()
                self._draw_sidepods()
                self._draw_front_wing()
                self._draw_rear_wing()

            glEndList()
        except Exception:
            self.ghost_display_list = None

    def compile_wheel_display_lists(self):
        if not HAS_OPENGL:
            return
        try:
            from PyQt5.QtGui import QOpenGLContext
            if QOpenGLContext.currentContext() is None:
                return
            for attr in ['wheel_display_list_left', 'wheel_display_list_right',
                         'caliper_display_list_left', 'caliper_display_list_right']:
                dl = getattr(self, attr, None)
                if dl is not None:
                    try:
                        glDeleteLists(dl, 1)
                    except Exception:
                        pass
                    setattr(self, attr, None)

            self.wheel_display_list_left = glGenLists(1)
            glNewList(self.wheel_display_list_left, GL_COMPILE)
            self._draw_wheel_rim_and_tire(is_left=True)
            glEndList()

            self.wheel_display_list_right = glGenLists(1)
            glNewList(self.wheel_display_list_right, GL_COMPILE)
            self._draw_wheel_rim_and_tire(is_left=False)
            glEndList()

            self.caliper_display_list_left = glGenLists(1)
            glNewList(self.caliper_display_list_left, GL_COMPILE)
            self._draw_caliper_geom(is_left=True)
            glEndList()

            self.caliper_display_list_right = glGenLists(1)
            glNewList(self.caliper_display_list_right, GL_COMPILE)
            self._draw_caliper_geom(is_left=False)
            glEndList()
        except Exception:
            self.wheel_display_list_left = None
            self.wheel_display_list_right = None
            self.caliper_display_list_left = None
            self.caliper_display_list_right = None

    def set_livery_preset(self, idx: int):
        if idx == 0:    # Formula Blue
            self.col_body = (0.024, 0.322, 0.867)
            self.col_accent = (0.906, 0.188, 0.188)
        elif idx == 1:  # Rosso Corsa
            self.col_body = (0.85, 0.12, 0.12)
            self.col_accent = (0.95, 0.95, 0.95)
        elif idx == 2:  # British Racing Green
            self.col_body = (0.08, 0.45, 0.28)
            self.col_accent = (0.95, 0.77, 0.15)
        elif idx == 3:  # Stealth Carbon / Gold
            self.col_body = (0.12, 0.14, 0.16)
            self.col_accent = (0.95, 0.77, 0.15)
        elif idx == 4:  # Modena Yellow
            self.col_body = (0.95, 0.77, 0.06)
            self.col_accent = (0.12, 0.14, 0.16)
        self.invalidate_display_list()

    def draw_ghost(self, state=None):
        """Renders the translucent 3D Ghost Car with satin cyan alpha blending via precompiled Display List."""
        if not HAS_OPENGL:
            return
        try:
            from PyQt5.QtGui import QOpenGLContext
            if QOpenGLContext.currentContext() is None:
                return
        except Exception:
            pass

        if self.ghost_display_list is None:
            self.compile_ghost_display_list()

        if self.ghost_display_list is not None:
            glCallList(self.ghost_display_list)
        else:
            old_body = self.col_body
            old_carbon = self.col_carbon
            old_accent = self.col_accent
            self.col_body = (0.20, 0.75, 0.95)
            self.col_carbon = (0.15, 0.60, 0.85)
            self.col_accent = (0.50, 0.90, 1.0)

            glColor4f(0.20, 0.75, 0.95, 0.35)
            if self.obj_mesh is not None:
                self._draw_obj_mesh()
            else:
                self._draw_monocoque()
                self._draw_sidepods()
                self._draw_front_wing()
                self._draw_rear_wing()

            self.col_body = old_body
            self.col_carbon = old_carbon
            self.col_accent = old_accent

    def draw(self, state=None, is_cockpit=False, skip_ao=False):
        if not HAS_OPENGL:
            return
        try:
            from PyQt5.QtGui import QOpenGLContext
            if QOpenGLContext.currentContext() is None:
                return
        except Exception:
            pass
        if state is None:
            state = {}

        steer_deg = float(state.get('steer_deg', 0.0))
        steer_rad = np.deg2rad(steer_deg)

        z_fl = float(state.get('z_fl', 0.0)) / 1000.0 if abs(state.get('z_fl', 0.0)) > 0.5 else float(state.get('z_fl', 0.0))
        z_fr = float(state.get('z_fr', 0.0)) / 1000.0 if abs(state.get('z_fr', 0.0)) > 0.5 else float(state.get('z_fr', 0.0))
        z_rl = float(state.get('z_rl', 0.0)) / 1000.0 if abs(state.get('z_rl', 0.0)) > 0.5 else float(state.get('z_rl', 0.0))
        z_rr = float(state.get('z_rr', 0.0)) / 1000.0 if abs(state.get('z_rr', 0.0)) > 0.5 else float(state.get('z_rr', 0.0))

        # Vehicle dimensions (FSAE ET-18 standard: L=1.53, W=1.25)
        L = 1.53
        W = 1.25
        a = L * 0.53
        b = L * 0.47
        tf_half = W / 2.0
        tr_half = 1.20 / 2.0

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)

        # 1. Static Body & Wheel Geometry (using compiled display lists)
        if self.body_display_list is None:
            self.compile_body_display_list()
        if self.wheel_display_list_left is None:
            self.compile_wheel_display_lists()
        if self.body_display_list is not None:
            glCallList(self.body_display_list)
        else:
            if self.obj_mesh is not None:
                self._draw_obj_mesh()
            else:
                self._draw_monocoque()
                self._draw_cockpit_interior()
                self._draw_sidepods()
                self._draw_front_wing()
                self._draw_rear_wing()

        # 2. Butterfly Steering Wheel (Dynamic real-time rotation)
        self._draw_fsae_steering_wheel(steer_deg, is_cockpit)

        # 3. Driver Helmet (Only in Chase / Third-Person view)
        if not is_cockpit:
            self._draw_driver_helmet()

        # 4. Wheels & 3D Suspension Wishbones
        steer_fl = -steer_rad * (0.90 if steer_rad > 0 else 1.10)
        steer_fr = -steer_rad * (1.10 if steer_rad > 0 else 0.90)

        wh_ang = [
            float(state.get('wheel_angle_fl', 0.0)),
            float(state.get('wheel_angle_fr', 0.0)),
            float(state.get('wheel_angle_rl', 0.0)),
            float(state.get('wheel_angle_rr', 0.0))
        ]

        # Front wheels (always visible from cockpit and chase)
        self._draw_wheel_and_suspension(a, tf_half, steer_fl, z_fl, True, True, wh_ang[0])
        self._draw_wheel_and_suspension(a, -tf_half, steer_fr, z_fr, True, False, wh_ang[1])

        # Rear wheels (visible in mirrors and side glances in open-wheel cockpit)
        self._draw_wheel_and_suspension(-b, tr_half, 0.0, z_rl, False, True, wh_ang[2])
        self._draw_wheel_and_suspension(-b, -tr_half, 0.0, z_rr, False, False, wh_ang[3])

    def _draw_obj_mesh(self):
        """Renders the parsed external OBJ mesh geometry."""
        if self.obj_mesh is None:
            return
        glColor3f(*self.col_body)
        glBegin(GL_TRIANGLES)
        for (p0, p1, p2), (n0, n1, n2) in self.obj_mesh['triangles']:
            glNormal3f(*n0); glVertex3f(*p0)
            glNormal3f(*n1); glVertex3f(*p1)
            glNormal3f(*n2); glVertex3f(*p2)
        glEnd()

    def _draw_monocoque(self):
        """Renders the tapered aerodynamic nosecone, monocoque flanks, undertray, and diffuser."""
        # 1. Front Crashbox / Impact Attenuator Tip
        glColor3f(0.12, 0.13, 0.15)
        glBegin(GL_QUADS)
        # Front Impact Face
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(1.38, -0.08, 0.03)
        glVertex3f(1.38, 0.08, 0.03)
        glVertex3f(1.38, 0.08, 0.13)
        glVertex3f(1.38, -0.08, 0.13)
        # Crashbox Top
        glNormal3f(0.1, 0.0, 0.9)
        glVertex3f(1.38, -0.08, 0.13)
        glVertex3f(1.38, 0.08, 0.13)
        glVertex3f(1.22, 0.10, 0.14)
        glVertex3f(1.22, -0.10, 0.14)
        # Crashbox Sides
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(1.38, 0.08, 0.03)
        glVertex3f(1.22, 0.10, 0.03)
        glVertex3f(1.22, 0.10, 0.14)
        glVertex3f(1.38, 0.08, 0.13)
        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(1.38, -0.08, 0.03)
        glVertex3f(1.38, -0.08, 0.13)
        glVertex3f(1.22, -0.10, 0.14)
        glVertex3f(1.22, -0.10, 0.03)
        # Crashbox Bottom
        glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(1.38, -0.08, 0.03)
        glVertex3f(1.22, -0.10, 0.03)
        glVertex3f(1.22, 0.10, 0.03)
        glVertex3f(1.38, 0.08, 0.03)
        glEnd()

        # 2. Sleek Tapered Nosecone (7 lofted aerodynamic cross-sections)
        sections = [
            (1.36, 0.08, 0.03, 0.12),
            (1.22, 0.10, 0.03, 0.14),
            (1.08, 0.13, 0.02, 0.17),
            (0.95, 0.16, 0.02, 0.20),
            (0.81, 0.19, 0.02, 0.23),
            (0.63, 0.22, 0.01, 0.25),
            (0.45, 0.24, 0.01, 0.27)
        ]

        glColor3f(*self.col_body)
        glBegin(GL_QUADS)
        for i in range(len(sections) - 1):
            x0, w0, zb0, zt0 = sections[i]
            x1, w1, zb1, zt1 = sections[i + 1]

            # Top Deck
            glNormal3f(0.15, 0.0, 0.95)
            glVertex3f(x0, -w0, zt0)
            glVertex3f(x0, w0, zt0)
            glVertex3f(x1, w1, zt1)
            glVertex3f(x1, -w1, zt1)

            # Left Flank
            glNormal3f(0.10, 0.95, 0.15)
            glVertex3f(x0, w0, zb0)
            glVertex3f(x1, w1, zb1)
            glVertex3f(x1, w1, zt1)
            glVertex3f(x0, w0, zt0)

            # Right Flank
            glNormal3f(0.10, -0.95, 0.15)
            glVertex3f(x0, -w0, zb0)
            glVertex3f(x0, -w0, zt0)
            glVertex3f(x1, -w1, zt1)
            glVertex3f(x1, -w1, zb1)

            # Under-nose Aerodynamic Channel
            glNormal3f(0.0, 0.0, -1.0)
            glVertex3f(x0, -w0, zb0)
            glVertex3f(x1, -w1, zb1)
            glVertex3f(x1, w1, zb1)
            glVertex3f(x0, w0, zb0)
        glEnd()

        # 3. Monocoque Cockpit Surround & Rear Powertrain Cowl
        glColor3f(*self.col_body)
        glBegin(GL_QUADS)
        # Cockpit outer flanks (x=0.45 to x=-0.32)
        glNormal3f(0.0, 1.0, 0.1)
        glVertex3f(0.45, 0.24, 0.01)
        glVertex3f(-0.32, 0.25, 0.01)
        glVertex3f(-0.32, 0.25, 0.26)
        glVertex3f(0.45, 0.24, 0.27)

        glNormal3f(0.0, -1.0, 0.1)
        glVertex3f(0.45, -0.24, 0.01)
        glVertex3f(0.45, -0.24, 0.27)
        glVertex3f(-0.32, -0.25, 0.26)
        glVertex3f(-0.32, -0.25, 0.01)

        # Rear Engine Cover / Powertrain Cowl (x=-0.32 to x=-0.72)
        glNormal3f(-0.2, 0.0, 0.95)
        glVertex3f(-0.32, -0.22, 0.26)
        glVertex3f(-0.32, 0.22, 0.26)
        glVertex3f(-0.72, 0.16, 0.19)
        glVertex3f(-0.72, -0.16, 0.19)

        # Rear Bulkhead Face
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(-0.72, -0.16, 0.01)
        glVertex3f(-0.72, 0.16, 0.01)
        glVertex3f(-0.72, 0.16, 0.19)
        glVertex3f(-0.72, -0.16, 0.19)
        glEnd()

        # 4. Full Carbon Undertray Splitter & Floor (x=1.25 to x=-0.42)
        glColor3f(*self.col_carbon)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(1.25, 0.20, 0.01)
        glVertex3f(1.25, -0.20, 0.01)
        glVertex3f(0.40, -0.54, 0.01)
        glVertex3f(0.40, 0.54, 0.01)

        glVertex3f(0.40, 0.54, 0.01)
        glVertex3f(0.40, -0.54, 0.01)
        glVertex3f(-0.42, -0.50, 0.01)
        glVertex3f(-0.42, 0.50, 0.01)
        glEnd()

        # 5. Rear Upswept Diffuser (x=-0.42 to x=-0.96, upswept at ~14 degrees)
        glColor3f(*self.col_carbon)
        glBegin(GL_QUADS)
        glNormal3f(-0.24, 0.0, -0.97)
        glVertex3f(-0.42, 0.38, 0.01)
        glVertex3f(-0.42, -0.38, 0.01)
        glVertex3f(-0.96, -0.36, 0.15)
        glVertex3f(-0.96, 0.36, 0.15)
        glEnd()

        # 6. Four Aerodynamic Vertical Diffuser Strakes / Fins
        glColor3f(*self.col_accent)
        for y_fin in [-0.25, -0.09, 0.09, 0.25]:
            glBegin(GL_QUADS)
            glNormal3f(0.0, 1.0 if y_fin > 0 else -1.0, 0.0)
            glVertex3f(-0.42, y_fin, 0.01)
            glVertex3f(-0.96, y_fin, 0.15)
            glVertex3f(-0.96, y_fin, 0.10)
            glVertex3f(-0.42, y_fin, -0.03)
            glEnd()

    def _draw_cockpit_interior(self):
        """Renders the hollow cockpit tub, bucket seat, 5-point harness, and 3D tubular roll hoop."""
        # 1. Hollow Cockpit Aperture Rim & Tub Interior
        glColor3f(0.08, 0.09, 0.11)
        glBegin(GL_QUADS)
        # Interior floor
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(0.35, -0.19, 0.02)
        glVertex3f(0.35, 0.19, 0.02)
        glVertex3f(-0.30, 0.19, 0.02)
        glVertex3f(-0.30, -0.19, 0.02)

        # Left tub inner wall
        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(0.35, 0.19, 0.02)
        glVertex3f(-0.30, 0.19, 0.02)
        glVertex3f(-0.30, 0.21, 0.26)
        glVertex3f(0.35, 0.21, 0.26)

        # Right tub inner wall
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(0.35, -0.19, 0.02)
        glVertex3f(0.35, -0.21, 0.26)
        glVertex3f(-0.30, -0.21, 0.26)
        glVertex3f(-0.30, -0.19, 0.02)

        # Dashboard firewall cowl
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(0.36, -0.21, 0.02)
        glVertex3f(0.36, 0.21, 0.02)
        glVertex3f(0.36, 0.21, 0.27)
        glVertex3f(0.36, -0.21, 0.27)
        glEnd()

        # 2. Ergonomic Carbon Bucket Seat (Reclined FSAE position)
        glColor3f(*self.col_seat)
        glBegin(GL_QUADS)
        # Seat Pan Cushion
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(0.18, -0.16, 0.04)
        glVertex3f(0.18, 0.16, 0.04)
        glVertex3f(-0.08, 0.16, 0.04)
        glVertex3f(-0.08, -0.16, 0.04)

        # Reclined Backrest (~45 degree angle)
        glNormal3f(0.65, 0.0, 0.75)
        glVertex3f(-0.08, -0.16, 0.04)
        glVertex3f(-0.08, 0.16, 0.04)
        glVertex3f(-0.30, 0.16, 0.35)
        glVertex3f(-0.30, -0.16, 0.35)

        # Left Lateral Rib/Thigh Bolster
        glNormal3f(0.0, -0.9, 0.3)
        glVertex3f(0.15, 0.16, 0.04)
        glVertex3f(-0.15, 0.16, 0.04)
        glVertex3f(-0.15, 0.18, 0.14)
        glVertex3f(0.15, 0.18, 0.14)

        # Right Lateral Bolster
        glNormal3f(0.0, 0.9, 0.3)
        glVertex3f(0.15, -0.16, 0.04)
        glVertex3f(0.15, -0.18, 0.14)
        glVertex3f(-0.15, -0.18, 0.14)
        glVertex3f(-0.15, -0.16, 0.04)
        glEnd()

        # 3. 5-Point Harness Racing Belts (Sabelt Red Straps)
        glColor3f(*self.col_belts)
        glBegin(GL_QUADS)
        # Left shoulder strap
        glNormal3f(0.6, 0.0, 0.8)
        glVertex3f(-0.28, 0.03, 0.33)
        glVertex3f(-0.28, 0.07, 0.33)
        glVertex3f(0.03, 0.03, 0.09)
        glVertex3f(0.03, 0.01, 0.09)
        # Right shoulder strap
        glVertex3f(-0.28, -0.07, 0.33)
        glVertex3f(-0.28, -0.03, 0.33)
        glVertex3f(0.03, -0.01, 0.09)
        glVertex3f(0.03, -0.03, 0.09)
        # Left lap strap
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(-0.06, 0.17, 0.06)
        glVertex3f(-0.03, 0.17, 0.06)
        glVertex3f(0.04, 0.03, 0.09)
        glVertex3f(0.02, 0.03, 0.09)
        # Right lap strap
        glVertex3f(-0.06, -0.17, 0.06)
        glVertex3f(0.02, -0.03, 0.09)
        glVertex3f(0.04, -0.03, 0.09)
        glVertex3f(-0.03, -0.17, 0.06)
        glEnd()

        # Central Rotary Quick-Release Buckle
        glColor3f(*self.col_buckle)
        glBegin(GL_QUADS)
        glNormal3f(0.5, 0.0, 0.86)
        glVertex3f(0.03, -0.025, 0.095)
        glVertex3f(0.03, 0.025, 0.095)
        glVertex3f(0.06, 0.025, 0.095)
        glVertex3f(0.06, -0.025, 0.095)
        glEnd()

        # 4. Padded Headrest on Firewall Bulkhead
        glColor3f(*self.col_headrest)
        glBegin(GL_QUADS)
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(-0.31, -0.08, 0.36)
        glVertex3f(-0.31, 0.08, 0.36)
        glVertex3f(-0.31, 0.08, 0.46)
        glVertex3f(-0.31, -0.08, 0.46)
        # Headrest Top
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(-0.31, -0.08, 0.46)
        glVertex3f(-0.31, 0.08, 0.46)
        glVertex3f(-0.34, 0.08, 0.46)
        glVertex3f(-0.34, -0.08, 0.46)
        glEnd()

        # 5. Volumetric 3D Tubular Main Roll-Hoop Arch (8-segment curved arc)
        arch_pts = [
            (-0.32, -0.22, 0.20),
            (-0.32, -0.20, 0.35),
            (-0.32, -0.15, 0.48),
            (-0.32, -0.08, 0.55),
            (-0.32, 0.00, 0.56),
            (-0.32, 0.08, 0.55),
            (-0.32, 0.15, 0.48),
            (-0.32, 0.20, 0.35),
            (-0.32, 0.22, 0.20)
        ]
        for i in range(len(arch_pts) - 1):
            _draw_tube_strut(arch_pts[i], arch_pts[i + 1], radius=0.013, n_sides=6, col=self.col_titanium)

        # Twin Rear Diagonal Braces (from roll-hoop shoulders down to engine bulkhead)
        _draw_tube_strut((-0.32, 0.10, 0.52), (-0.68, 0.16, 0.19), radius=0.011, n_sides=6, col=self.col_titanium)
        _draw_tube_strut((-0.32, -0.10, 0.52), (-0.68, -0.16, 0.19), radius=0.011, n_sides=6, col=self.col_titanium)

        # Front Roll Hoop structure under dashboard
        _draw_tube_strut((0.36, -0.20, 0.26), (0.36, 0.20, 0.26), radius=0.010, n_sides=6, col=self.col_titanium)

        # 6. Aerodynamic Carbon Side Mirrors (Visible in Cockpit POV)
        for side in [1.0, -1.0]:
            ym = side * 0.28
            # Mirror carbon housing
            glColor3f(*self.col_carbon)
            glBegin(GL_QUADS)
            glNormal3f(0.3, side * 0.8, 0.5)
            glVertex3f(0.34, ym, 0.28)
            glVertex3f(0.28, ym + side * 0.07, 0.28)
            glVertex3f(0.28, ym + side * 0.07, 0.35)
            glVertex3f(0.34, ym, 0.35)
            # Front aerodynamic cone
            glNormal3f(1.0, 0.0, 0.0)
            glVertex3f(0.34, ym, 0.28)
            glVertex3f(0.34, ym, 0.35)
            glVertex3f(0.37, ym + side * 0.035, 0.315)
            glVertex3f(0.37, ym + side * 0.035, 0.315)
            # Reflective Mirror Glass (Facing driver)
            glColor3f(0.85, 0.90, 0.98)
            glNormal3f(-0.95, -side * 0.2, 0.1)
            glVertex3f(0.28, ym + side * 0.005, 0.285)
            glVertex3f(0.28, ym + side * 0.065, 0.285)
            glVertex3f(0.28, ym + side * 0.065, 0.345)
            glVertex3f(0.28, ym + side * 0.005, 0.345)
            glEnd()
            # Mirror mounting stalk
            _draw_tube_strut((0.31, ym * 0.88, 0.25), (0.31, ym + side * 0.03, 0.28), radius=0.006, n_sides=6, col=self.col_carbon)

    def _draw_sidepods(self):
        """Renders undercut Coke-bottle sidepods with front radiator intake scoops and cooling vents."""
        for side in [1.0, -1.0]:
            y_in = side * 0.25
            y_out_front = side * 0.52
            y_out_mid = side * 0.48
            y_out_rear = side * 0.38  # Coke-bottle inward taper

            # 1. Front Radiator Intake Scoop (at x=0.38)
            glColor3f(0.08, 0.09, 0.11)
            glBegin(GL_QUADS)
            glNormal3f(-1.0, 0.0, 0.0)
            glVertex3f(0.38, y_in, 0.03)
            glVertex3f(0.38, y_out_front, 0.03)
            glVertex3f(0.38, y_out_front, 0.20)
            glVertex3f(0.38, y_in, 0.20)
            glEnd()

            # 2. Recessed Radiator Core Mesh (Dark metallic cooling fins)
            glColor3f(0.20, 0.22, 0.25)
            glBegin(GL_QUADS)
            glNormal3f(0.9, 0.0, 0.2)
            glVertex3f(0.33, y_in + side * 0.02, 0.04)
            glVertex3f(0.33, y_out_front - side * 0.03, 0.04)
            glVertex3f(0.33, y_out_front - side * 0.03, 0.19)
            glVertex3f(0.33, y_in + side * 0.02, 0.19)
            glEnd()

            # 3. Outer Flank Bodywork (with Coke-bottle undercut)
            glColor3f(*self.col_body)
            glBegin(GL_QUADS)
            # Front section outer wall
            glNormal3f(0.0, side, 0.0)
            glVertex3f(0.38, y_out_front, 0.03)
            glVertex3f(0.00, y_out_mid, 0.03)
            glVertex3f(0.00, y_out_mid, 0.20)
            glVertex3f(0.38, y_out_front, 0.20)

            # Rear section outer wall (tapering inward)
            glNormal3f(0.05, side, 0.0)
            glVertex3f(0.00, y_out_mid, 0.03)
            glVertex3f(-0.54, y_out_rear, 0.03)
            glVertex3f(-0.54, y_out_rear, 0.16)
            glVertex3f(0.00, y_out_mid, 0.20)

            # Sidepod Upper Deck
            glNormal3f(0.0, side * 0.2, 0.95)
            glVertex3f(0.38, y_in, 0.20)
            glVertex3f(0.38, y_out_front, 0.20)
            glVertex3f(0.00, y_out_mid, 0.20)
            glVertex3f(0.00, y_in, 0.20)

            glVertex3f(0.00, y_in, 0.20)
            glVertex3f(0.00, y_out_mid, 0.20)
            glVertex3f(-0.54, y_out_rear, 0.16)
            glVertex3f(-0.54, y_in, 0.16)
            glEnd()

            # 4. Louvered Top Cooling Vents (Accent highlight)
            glColor3f(*self.col_accent)
            glBegin(GL_QUADS)
            glNormal3f(0.0, 0.0, 1.0)
            for vx in [-0.08, -0.16, -0.24]:
                glVertex3f(vx, y_in + side * 0.04, 0.201)
                glVertex3f(vx, y_out_mid - side * 0.05, 0.201)
                glVertex3f(vx - 0.03, y_out_mid - side * 0.05, 0.201)
                glVertex3f(vx - 0.03, y_in + side * 0.04, 0.201)
            glEnd()

    def _draw_front_wing(self):
        """Renders the multi-element curved front wing with sculpted endplates, footplates, and pylons."""
        span_half = 0.64
        n_span = 8
        y_nodes = np.linspace(-span_half, span_half, n_span + 1)

        # 1. Mainplane Aerofoil Element (Multi-segment spanwise)
        glColor3f(*self.col_carbon)
        glBegin(GL_QUADS)
        for s in range(n_span):
            y_l, y_r = float(y_nodes[s]), float(y_nodes[s + 1])
            # Upper Surface
            glNormal3f(-0.15, 0.0, 0.98)
            glVertex3f(1.40, y_r, 0.03)
            glVertex3f(1.40, y_l, 0.03)
            glVertex3f(1.12, y_l, 0.06)
            glVertex3f(1.12, y_r, 0.06)
            # Lower Surface (Ground effect camber)
            glNormal3f(0.0, 0.0, -1.0)
            glVertex3f(1.40, y_l, 0.02)
            glVertex3f(1.40, y_r, 0.02)
            glVertex3f(1.12, y_r, 0.04)
            glVertex3f(1.12, y_l, 0.04)
            # Rounded Leading Edge
            glNormal3f(1.0, 0.0, 0.0)
            glVertex3f(1.40, y_l, 0.02)
            glVertex3f(1.40, y_r, 0.02)
            glVertex3f(1.40, y_r, 0.03)
            glVertex3f(1.40, y_l, 0.03)
        glEnd()

        # 2. Secondary Flap with trailing edge Gurney flap
        glColor3f(*self.col_carbon_light)
        glBegin(GL_QUADS)
        for s in range(n_span):
            y_l, y_r = float(y_nodes[s]), float(y_nodes[s + 1])
            glNormal3f(-0.35, 0.0, 0.93)
            glVertex3f(1.26, y_r, 0.08)
            glVertex3f(1.26, y_l, 0.08)
            glVertex3f(1.08, y_l, 0.14)
            glVertex3f(1.08, y_r, 0.14)
            # Vertical Gurney flap lip
            glNormal3f(-1.0, 0.0, 0.0)
            glVertex3f(1.08, y_l, 0.14)
            glVertex3f(1.08, y_r, 0.14)
            glVertex3f(1.08, y_r, 0.16)
            glVertex3f(1.08, y_l, 0.16)
        glEnd()

        # 3. Sculpted Vertical Endplates with Horizontal Footplates
        for side in [1.0, -1.0]:
            y_ep = side * span_half
            # Vertical Endplate
            glColor3f(*self.col_accent)
            glBegin(GL_QUADS)
            glNormal3f(0.0, side, 0.0)
            glVertex3f(1.44, y_ep, 0.01)
            glVertex3f(1.04, y_ep, 0.01)
            glVertex3f(1.04, y_ep, 0.22)
            glVertex3f(1.36, y_ep, 0.20)
            # Front Chamfer
            glVertex3f(1.44, y_ep, 0.01)
            glVertex3f(1.36, y_ep, 0.20)
            glVertex3f(1.44, y_ep, 0.12)
            glVertex3f(1.44, y_ep, 0.01)
            glEnd()

            # Horizontal Ground-Sealing Footplate
            glColor3f(*self.col_carbon)
            glBegin(GL_QUADS)
            glNormal3f(0.0, 0.0, 1.0)
            y_fp_out = y_ep + side * 0.05
            glVertex3f(1.42, y_ep, 0.015)
            glVertex3f(1.42, y_fp_out, 0.015)
            glVertex3f(1.08, y_fp_out, 0.015)
            glVertex3f(1.08, y_ep, 0.015)
            glEnd()

        # 4. Twin Nose Mounting Pylons (Attaching wing to nosecone)
        for side in [0.10, -0.10]:
            _draw_tube_strut((1.20, side, 0.06), (1.20, side, 0.14), radius=0.009, n_sides=6, col=self.col_carbon)

    def _draw_rear_wing(self):
        """Renders the elevated dual-element rear wing, slotted endplates, swan-neck pylons, and beam wing."""
        span_half = 0.54
        n_span = 8
        y_nodes = np.linspace(-span_half, span_half, n_span + 1)

        # 1. Mainplane Aerofoil (Cambered high-downforce profile)
        glColor3f(*self.col_carbon)
        glBegin(GL_QUADS)
        for s in range(n_span):
            y_l, y_r = float(y_nodes[s]), float(y_nodes[s + 1])
            glNormal3f(0.12, 0.0, 0.99)
            glVertex3f(-0.92, y_r, 0.64)
            glVertex3f(-0.92, y_l, 0.64)
            glVertex3f(-1.15, y_l, 0.60)
            glVertex3f(-1.15, y_r, 0.60)
            # Underside
            glNormal3f(0.0, 0.0, -1.0)
            glVertex3f(-0.92, y_l, 0.62)
            glVertex3f(-0.92, y_r, 0.62)
            glVertex3f(-1.15, y_r, 0.58)
            glVertex3f(-1.15, y_l, 0.58)
        glEnd()

        # 2. Upper Flap with Gurney flap
        glColor3f(*self.col_carbon_light)
        glBegin(GL_QUADS)
        for s in range(n_span):
            y_l, y_r = float(y_nodes[s]), float(y_nodes[s + 1])
            glNormal3f(0.25, 0.0, 0.96)
            glVertex3f(-1.00, y_r, 0.76)
            glVertex3f(-1.00, y_l, 0.76)
            glVertex3f(-1.18, y_l, 0.69)
            glVertex3f(-1.18, y_r, 0.69)
            # Gurney flap
            glNormal3f(-1.0, 0.0, 0.0)
            glVertex3f(-1.18, y_l, 0.69)
            glVertex3f(-1.18, y_r, 0.69)
            glVertex3f(-1.18, y_r, 0.72)
            glVertex3f(-1.18, y_l, 0.72)
        glEnd()

        # 3. Slotted Vertical Endplates
        for side in [1.0, -1.0]:
            y_ep = side * span_half
            glColor3f(*self.col_body)
            glBegin(GL_QUADS)
            glNormal3f(0.0, side, 0.0)
            glVertex3f(-0.88, y_ep, 0.46)
            glVertex3f(-1.22, y_ep, 0.46)
            glVertex3f(-1.22, y_ep, 0.82)
            glVertex3f(-0.88, y_ep, 0.80)
            glEnd()

            # Horizontal Pressure Relief Slits
            glColor3f(0.12, 0.13, 0.15)
            glBegin(GL_QUADS)
            glNormal3f(0.0, side, 0.0)
            for z_slit in [0.73, 0.77]:
                glVertex3f(-1.10, y_ep + side * 0.001, z_slit)
                glVertex3f(-1.20, y_ep + side * 0.001, z_slit)
                glVertex3f(-1.20, y_ep + side * 0.001, z_slit + 0.02)
                glVertex3f(-1.10, y_ep + side * 0.001, z_slit + 0.02)
            glEnd()

        # 4. Aerodynamic Swan-Neck Pylons (Hanging the wing from the upper surface)
        for side in [0.14, -0.14]:
            p_base = (-0.68, side, 0.19)
            p_mid = (-0.85, side, 0.52)
            p_top = (-1.00, side, 0.68)
            _draw_tube_strut(p_base, p_mid, radius=0.010, n_sides=6, col=self.col_carbon)
            _draw_tube_strut(p_mid, p_top, radius=0.010, n_sides=6, col=self.col_carbon)

        # 5. Lower Beam Wing (Diffuser upwash driver)
        glColor3f(*self.col_carbon)
        glBegin(GL_QUADS)
        glNormal3f(0.1, 0.0, 0.99)
        glVertex3f(-0.72, 0.36, 0.25)
        glVertex3f(-0.72, -0.36, 0.25)
        glVertex3f(-0.86, -0.36, 0.22)
        glVertex3f(-0.86, 0.36, 0.22)
        glEnd()

    def _draw_fsae_steering_wheel(self, steer_deg: float, is_cockpit: bool):
        """
        Renders the ergonomic FSAE butterfly steering wheel with integrated
        digital racing LCD dashboard display and multicolor LED shift-light strip.
        Rotates dynamically around the 18-degree inclined steering column axis.
        """
        glPushMatrix()
        # Position at driver hands / cockpit cowl
        glTranslatef(0.38, 0.0, 0.24)
        # 18-degree steering column inclination
        glRotatef(18.0, 0, 1, 0)
        # Real-time dynamic rotation around the column axis (corrected rotation sign)
        glRotatef(steer_deg * 2.0, 1, 0, 0)

        # 1. Butterfly Wheel Central Carbon Plate
        glColor3f(0.12, 0.13, 0.15)
        glBegin(GL_QUADS)
        glNormal3f(-1.0, 0.0, 0.0)
        # Center hub area
        glVertex3f(0.0, -0.07, -0.05)
        glVertex3f(0.0, 0.07, -0.05)
        glVertex3f(0.0, 0.07, 0.07)
        glVertex3f(0.0, -0.07, 0.07)
        # Left wing plate
        glVertex3f(0.0, 0.07, -0.04)
        glVertex3f(0.0, 0.12, -0.07)
        glVertex3f(0.0, 0.12, 0.06)
        glVertex3f(0.0, 0.07, 0.06)
        # Right wing plate
        glVertex3f(0.0, -0.12, -0.07)
        glVertex3f(0.0, -0.07, -0.04)
        glVertex3f(0.0, -0.07, 0.06)
        glVertex3f(0.0, -0.12, 0.06)
        glEnd()

        # 2. Left & Right Ergonomic Hand Grips (Molded rubber/Alcantara)
        glColor3f(0.10, 0.10, 0.12)
        glBegin(GL_QUADS)
        # Left grip front face
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(0.005, 0.10, -0.07)
        glVertex3f(0.005, 0.125, -0.07)
        glVertex3f(0.005, 0.125, 0.065)
        glVertex3f(0.005, 0.10, 0.065)
        # Right grip front face
        glVertex3f(0.005, -0.125, -0.07)
        glVertex3f(0.005, -0.10, -0.07)
        glVertex3f(0.005, -0.10, 0.065)
        glVertex3f(0.005, -0.125, 0.065)
        glEnd()

        # 3. Integrated Digital Racing LCD Display Screen (AIM / MoTeC style)
        # Screen Bezel
        glColor3f(0.05, 0.05, 0.06)
        glBegin(GL_QUADS)
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(0.003, -0.055, -0.005)
        glVertex3f(0.003, 0.055, -0.005)
        glVertex3f(0.003, 0.055, 0.055)
        glVertex3f(0.003, -0.055, 0.055)
        glEnd()

        # Screen Display Face (Illuminated Dark Navy Blue)
        glColor3f(*self.col_dash_screen)
        glBegin(GL_QUADS)
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(0.004, -0.050, 0.00)
        glVertex3f(0.004, 0.050, 0.00)
        glVertex3f(0.004, 0.050, 0.050)
        glVertex3f(0.004, -0.050, 0.050)
        glEnd()

        # Digital Tachometer RPM Bar (Green to Yellow to Red)
        glBegin(GL_QUADS)
        glNormal3f(-1.0, 0.0, 0.0)
        # Green zone (low RPM)
        glColor3f(0.15, 0.90, 0.25)
        glVertex3f(0.005, -0.045, 0.040)
        glVertex3f(0.005, -0.015, 0.040)
        glVertex3f(0.005, -0.015, 0.046)
        glVertex3f(0.005, -0.045, 0.046)
        # Yellow zone (mid RPM)
        glColor3f(0.95, 0.85, 0.10)
        glVertex3f(0.005, -0.012, 0.040)
        glVertex3f(0.005, 0.018, 0.040)
        glVertex3f(0.005, 0.018, 0.046)
        glVertex3f(0.005, -0.012, 0.046)
        # Red zone (peak RPM)
        glColor3f(0.95, 0.15, 0.15)
        glVertex3f(0.005, 0.021, 0.040)
        glVertex3f(0.005, 0.045, 0.040)
        glVertex3f(0.005, 0.045, 0.046)
        glVertex3f(0.005, 0.021, 0.046)
        glEnd()

        # Bold Gear Indicator ("1" in bright cyan/white)
        glColor3f(0.85, 0.95, 1.0)
        glBegin(GL_QUADS)
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(0.005, -0.006, 0.012)
        glVertex3f(0.005, 0.006, 0.012)
        glVertex3f(0.005, 0.006, 0.034)
        glVertex3f(0.005, -0.006, 0.034)
        glEnd()

        # 4. Multicolor LED Shift-Light Strip (8 LEDs along top edge)
        led_colors = [
            (0.10, 0.88, 0.20), (0.10, 0.88, 0.20), (0.10, 0.88, 0.20),  # Green
            (0.95, 0.85, 0.05), (0.95, 0.85, 0.05),                       # Yellow
            (0.95, 0.15, 0.15), (0.95, 0.15, 0.15),                       # Red
            (0.20, 0.60, 1.0)                                             # Shift Flash Blue
        ]
        glBegin(GL_QUADS)
        glNormal3f(-1.0, 0.0, 0.0)
        for i, col in enumerate(led_colors):
            y_led = -0.042 + i * 0.012
            glColor3f(*col)
            glVertex3f(0.004, y_led - 0.004, 0.060)
            glVertex3f(0.004, y_led + 0.004, 0.060)
            glVertex3f(0.004, y_led + 0.004, 0.066)
            glVertex3f(0.004, y_led - 0.004, 0.066)
        glEnd()

        # 5. Color Pushbuttons (DRS Cyan, Pit Limiter Yellow, Radio Green)
        buttons = [
            (0.08, -0.025, (0.0, 0.85, 1.0)),     # DRS
            (-0.08, -0.025, (0.95, 0.80, 0.05)),  # Pit Limiter
            (0.08, 0.025, (0.15, 0.85, 0.25)),    # Radio
            (-0.08, 0.025, (0.95, 0.15, 0.15))    # Neutral
        ]
        glBegin(GL_QUADS)
        glNormal3f(-1.0, 0.0, 0.0)
        for by, bz, bcol in buttons:
            glColor3f(*bcol)
            glVertex3f(0.004, by - 0.006, bz - 0.006)
            glVertex3f(0.004, by + 0.006, bz - 0.006)
            glVertex3f(0.004, by + 0.006, bz + 0.006)
            glVertex3f(0.004, by - 0.006, bz + 0.006)
        glEnd()

        # Steering Column Shaft
        _draw_tube_strut((0.0, 0.0, 0.0), (0.14, 0.0, 0.0), radius=0.016, n_sides=6, col=(0.20, 0.22, 0.25))

        glPopMatrix()

    def _draw_driver_helmet(self):
        """Renders the driver helmet in chase / third-person view."""
        glColor3f(*self.col_helmet)
        glPushMatrix()
        glTranslatef(-0.05, 0.0, 0.38)

        # Faceted helmet shell
        glBegin(GL_QUADS)
        # Top
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(0.08, 0.075, 0.09)
        glVertex3f(0.08, -0.075, 0.09)
        glVertex3f(-0.08, -0.075, 0.09)
        glVertex3f(-0.08, 0.075, 0.09)
        # Front
        glNormal3f(0.9, 0.0, 0.4)
        glVertex3f(0.10, 0.075, 0.02)
        glVertex3f(0.10, -0.075, 0.02)
        glVertex3f(0.08, -0.075, 0.09)
        glVertex3f(0.08, 0.075, 0.09)
        # Sides
        glNormal3f(0.0, 1.0, 0.1)
        glVertex3f(0.10, 0.075, 0.02)
        glVertex3f(0.08, 0.075, 0.09)
        glVertex3f(-0.08, 0.075, 0.09)
        glVertex3f(-0.10, 0.075, 0.02)
        glNormal3f(0.0, -1.0, 0.1)
        glVertex3f(0.10, -0.075, 0.02)
        glVertex3f(-0.10, -0.075, 0.02)
        glVertex3f(-0.08, -0.075, 0.09)
        glVertex3f(0.08, -0.075, 0.09)
        glEnd()

        # Dark Iridium Visor
        glColor3f(0.10, 0.12, 0.15)
        glBegin(GL_QUADS)
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(0.105, 0.070, 0.065)
        glVertex3f(0.105, -0.070, 0.065)
        glVertex3f(0.105, -0.070, 0.015)
        glVertex3f(0.105, 0.070, 0.015)
        glEnd()

        glPopMatrix()

    def _draw_wheel_and_suspension(self, wx, wy, steer_angle, z_disp, is_front, is_left, wh_rot_rad):
        """
        Renders true 3D volumetric tubular wishbones (upper/lower A-arms, pushrod, tie-rod)
        and solid Hoosier 16x7.5-10 racing wheels with rounded shoulders, recessed rim,
        5 alloy spokes, ventilated steel brake disc, and anodized brake caliper.
        """
        wz = 0.20 + z_disp
        sign_y = 1.0 if is_left else -1.0
        chassis_y = sign_y * (0.22 if is_front else 0.24)

        # 1. Upright Ball Joint Attachment Points
        y_upright = wy - sign_y * 0.065
        p_u_bj = (wx, y_upright, wz + 0.065)
        p_l_bj = (wx, y_upright, wz - 0.065)

        # 2. Chassis Wishbone Hardpoints
        p_u_fwd = (wx + 0.14, chassis_y, 0.22)
        p_u_aft = (wx - 0.14, chassis_y, 0.22)
        p_l_fwd = (wx + 0.16, chassis_y * 1.08, 0.04)
        p_l_aft = (wx - 0.16, chassis_y * 1.08, 0.04)

        # Draw Volumetric 3D Wishbone Struts
        _draw_tube_strut(p_u_fwd, p_u_bj, radius=0.0075, n_sides=6, col=self.col_wishbone)
        _draw_tube_strut(p_u_aft, p_u_bj, radius=0.0075, n_sides=6, col=self.col_wishbone)
        _draw_tube_strut(p_l_fwd, p_l_bj, radius=0.0075, n_sides=6, col=self.col_wishbone)
        _draw_tube_strut(p_l_aft, p_l_bj, radius=0.0075, n_sides=6, col=self.col_wishbone)

        # 3. Pushrod / Pullrod (Titanium satin link)
        p_rocker = (wx - 0.08, chassis_y * 0.85, 0.26)
        _draw_tube_strut(p_l_bj, p_rocker, radius=0.0070, n_sides=6, col=self.col_titanium)

        # 4. Tie-Rod / Steering Link
        if is_front:
            p_rack = (wx + 0.08, chassis_y, 0.12)
            p_steer_arm = (wx + 0.07, y_upright, wz + 0.01)
            _draw_tube_strut(p_rack, p_steer_arm, radius=0.0065, n_sides=6, col=self.col_wishbone)

        # 5. Upright Hub Carrier Block
        _draw_tube_strut(p_l_bj, p_u_bj, radius=0.012, n_sides=6, col=self.col_alloy)

        # 6. Solid Hoosier 16x7.5-10 Wheel Assembly
        glPushMatrix()
        glTranslatef(wx, wy, wz)
        if is_front:
            glRotatef(np.rad2deg(steer_angle), 0, 0, 1)

        # 6k. Anodized Brake Caliper (Fixed on upright, turns with steer but does not spin)
        if is_left:
            if self.caliper_display_list_left is not None:
                glCallList(self.caliper_display_list_left)
            else:
                self._draw_caliper_geom(is_left=True)
        else:
            if self.caliper_display_list_right is not None:
                glCallList(self.caliper_display_list_right)
            else:
                self._draw_caliper_geom(is_left=False)

        # Dynamic Wheel Spin
        glPushMatrix()
        glRotatef(np.rad2deg(wh_rot_rad), 0, 1 if is_left else -1, 0)

        if is_left:
            if self.wheel_display_list_left is not None:
                glCallList(self.wheel_display_list_left)
            else:
                self._draw_wheel_rim_and_tire(is_left=True)
        else:
            if self.wheel_display_list_right is not None:
                glCallList(self.wheel_display_list_right)
            else:
                self._draw_wheel_rim_and_tire(is_left=False)

        glPopMatrix()  # End spinning assembly
        glPopMatrix()  # End wheel assembly

    def _draw_wheel_rim_and_tire(self, is_left: bool):
        """Draws solid Hoosier 16x7.5-10 wheel rim and tire geometry (for display list compilation or fallback)."""
        sign_y = 1.0 if is_left else -1.0

        # Geometry parameters
        r_tread = 0.203     # Outer radius (16 inch tire)
        r_shoulder = 0.188  # Shoulder bevel radius
        r_rim = 0.127       # Rim bead radius (10 inch rim)
        r_barrel = 0.115    # Recessed rim barrel radius

        y_tread_out = sign_y * 0.055
        y_tread_in = -sign_y * 0.055
        y_sh_out = sign_y * 0.082
        y_sh_in = -sign_y * 0.082
        y_bead_out = sign_y * 0.078
        y_bead_in = -sign_y * 0.078
        y_barrel_deep = sign_y * 0.035

        n_seg = 22

        # Precompute radial angles
        cos_tab = [math.cos(2.0 * math.pi * i / n_seg) for i in range(n_seg + 1)]
        sin_tab = [math.sin(2.0 * math.pi * i / n_seg) for i in range(n_seg + 1)]

        # 6a. Main Tread Cylinder (Matte Rubber)
        glColor3f(*self.col_tread)
        glBegin(GL_QUAD_STRIP)
        for i in range(n_seg + 1):
            ca, sa = cos_tab[i], sin_tab[i]
            glNormal3f(ca, 0.0, sa)
            glVertex3f(r_tread * ca, y_tread_out, r_tread * sa)
            glVertex3f(r_tread * ca, y_tread_in, r_tread * sa)
        glEnd()

        # 6b. Outer Rounded Shoulder
        glBegin(GL_QUAD_STRIP)
        for i in range(n_seg + 1):
            ca, sa = cos_tab[i], sin_tab[i]
            glNormal3f(ca * 0.7, sign_y * 0.7, sa * 0.7)
            glVertex3f(r_tread * ca, y_tread_out, r_tread * sa)
            glVertex3f(r_shoulder * ca, y_sh_out, r_shoulder * sa)
        glEnd()

        # 6c. Inner Rounded Shoulder
        glBegin(GL_QUAD_STRIP)
        for i in range(n_seg + 1):
            ca, sa = cos_tab[i], sin_tab[i]
            glNormal3f(ca * 0.7, -sign_y * 0.7, sa * 0.7)
            glVertex3f(r_shoulder * ca, y_sh_in, r_shoulder * sa)
            glVertex3f(r_tread * ca, y_tread_in, r_tread * sa)
        glEnd()

        # 6d. Outer Sidewall (Tapering from shoulder to rim bead)
        glBegin(GL_QUAD_STRIP)
        for i in range(n_seg + 1):
            ca, sa = cos_tab[i], sin_tab[i]
            glNormal3f(0.0, sign_y, 0.0)
            glVertex3f(r_shoulder * ca, y_sh_out, r_shoulder * sa)
            glVertex3f(r_rim * ca, y_bead_out, r_rim * sa)
        glEnd()

        # 6e. Inner Sidewall
        glBegin(GL_QUAD_STRIP)
        for i in range(n_seg + 1):
            ca, sa = cos_tab[i], sin_tab[i]
            glNormal3f(0.0, -sign_y, 0.0)
            glVertex3f(r_rim * ca, y_bead_in, r_rim * sa)
            glVertex3f(r_shoulder * ca, y_sh_in, r_shoulder * sa)
        glEnd()

        # 6f. Polished Outer Rim Lip
        glColor3f(*self.col_alloy)
        glBegin(GL_QUAD_STRIP)
        for i in range(n_seg + 1):
            ca, sa = cos_tab[i], sin_tab[i]
            glNormal3f(0.0, sign_y, 0.0)
            glVertex3f(r_rim * ca, y_bead_out, r_rim * sa)
            glVertex3f(r_barrel * ca, y_bead_out, r_barrel * sa)
        glEnd()

        # 6g. Recessed Rim Barrel (Drop Center Wheel Well)
        glColor3f(*self.col_rim_inner)
        glBegin(GL_QUAD_STRIP)
        for i in range(n_seg + 1):
            ca, sa = cos_tab[i], sin_tab[i]
            glNormal3f(-ca, 0.0, -sa)
            glVertex3f(r_barrel * ca, y_bead_out, r_barrel * sa)
            glVertex3f(r_barrel * ca, y_barrel_deep, r_barrel * sa)
        glEnd()

        # 6h. 5 Sculpted Alloy Spokes
        glColor3f(*self.col_alloy)
        r_hub = 0.040
        spoke_y = y_barrel_deep + sign_y * 0.008
        for k in range(5):
            base_ang = 2.0 * math.pi * k / 5.0
            ang_l = base_ang - 0.16
            ang_r = base_ang + 0.16

            glBegin(GL_QUADS)
            glNormal3f(0.0, sign_y, 0.0)
            glVertex3f(r_hub * math.cos(ang_l), spoke_y, r_hub * math.sin(ang_l))
            glVertex3f(r_barrel * math.cos(ang_l), spoke_y, r_barrel * math.sin(ang_l))
            glVertex3f(r_barrel * math.cos(ang_r), spoke_y, r_barrel * math.sin(ang_r))
            glVertex3f(r_hub * math.cos(ang_r), spoke_y, r_hub * math.sin(ang_r))
            glEnd()

        # 6i. Center Hub Hex Nut
        glColor3f(0.25, 0.28, 0.32)
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0.0, sign_y, 0.0)
        glVertex3f(0.0, spoke_y + sign_y * 0.008, 0.0)
        for i in range(7):
            h_ang = i * 2.0 * math.pi / 6.0
            glVertex3f(0.026 * math.cos(h_ang), spoke_y + sign_y * 0.004, 0.026 * math.sin(h_ang))
        glEnd()

        # 6j. Ventilated Steel Brake Rotor Disc (Faces + Ventilated Rim Edge)
        glColor3f(*self.col_brake_disc)
        disc_y = y_tread_in + sign_y * 0.020
        disc_w = sign_y * 0.008
        r_disc_out = 0.110
        r_disc_in = 0.055
        glBegin(GL_QUAD_STRIP)
        for i in range(n_seg + 1):
            ca, sa = cos_tab[i], sin_tab[i]
            glNormal3f(0.0, -sign_y, 0.0)
            glVertex3f(r_disc_out * ca, disc_y, r_disc_out * sa)
            glVertex3f(r_disc_in * ca, disc_y, r_disc_in * sa)
        glEnd()

        glBegin(GL_QUAD_STRIP)
        for i in range(n_seg + 1):
            ca, sa = cos_tab[i], sin_tab[i]
            glNormal3f(ca, 0.0, sa)
            glVertex3f(r_disc_out * ca, disc_y, r_disc_out * sa)
            glVertex3f(r_disc_out * ca, disc_y + disc_w, r_disc_out * sa)
        glEnd()

    def _draw_caliper_geom(self, is_left: bool):
        """Draws anodized brake caliper geometry (for display list compilation or fallback)."""
        sign_y = 1.0 if is_left else -1.0
        y_tread_in = -sign_y * 0.055
        disc_y = y_tread_in + sign_y * 0.020
        glColor3f(*self.col_caliper)
        cal_y = disc_y + sign_y * 0.005
        # Caliper placed at upper-rear corner (45 degrees)
        ang_cal = math.pi * 0.75
        cx_cal = 0.095 * math.cos(ang_cal)
        cz_cal = 0.095 * math.sin(ang_cal)
        glBegin(GL_QUADS)
        glNormal3f(0.0, -sign_y, 0.0)
        glVertex3f(cx_cal - 0.035, cal_y, cz_cal - 0.025)
        glVertex3f(cx_cal + 0.035, cal_y, cz_cal - 0.025)
        glVertex3f(cx_cal + 0.035, cal_y, cz_cal + 0.025)
        glVertex3f(cx_cal - 0.035, cal_y, cz_cal + 0.025)
        glEnd()


# =============================================================================
# 2. B-SEGMENT COMPACT SUV 3D MODEL ("A RETTANGOLONI" BOXY GEOMETRIC DESIGN)
# =============================================================================
class SUVCar3DModel:
    """
    Parametric 3D Geometry Builder for a modern B-Segment Compact Road SUV.
    Engineered with clean, solid, boxy geometric volumes ('a rettangoloni'):
    - Clean rectangular lower chassis box (body color + charcoal lower rocker sill)
    - Clean rectangular upper cabin greenhouse (tinted glass & solid roof)
    - Clean front rectangular grille and twin crisp LED headlights
    - Clean rear horizontal LED light bar
    - Sized 215/60 R17 road wheels (Rw = 0.315 m) with front steering and rolling spin
    """

    def __init__(self):
        # Default livery colors (Urban Blue Metallic)
        self.col_body = (0.16, 0.38, 0.65)         # Metallic Slate Blue
        self.col_cladding = (0.12, 0.13, 0.15)     # Dark Charcoal Lower Sill
        self.col_glass = (0.08, 0.10, 0.14)        # Tinted Window Glass
        self.col_grille = (0.05, 0.06, 0.07)       # Obsidian Front Grille
        self.col_lights_front = (0.88, 0.95, 1.0)  # Cool White / Ice Blue LED
        self.col_lights_rear = (0.95, 0.12, 0.12)   # Bright Ruby Red LED
        self.col_alloy = (0.82, 0.84, 0.88)        # Silver Metallic Alloy
        self.col_accent = (0.88, 0.95, 1.0)        # Accent Light / Detail Color
        self.body_display_list = None

    def invalidate_display_list(self):
        if HAS_OPENGL and self.body_display_list is not None:
            try:
                glDeleteLists(self.body_display_list, 1)
            except Exception:
                pass
        self.body_display_list = None

    def compile_body_display_list(self):
        if not HAS_OPENGL:
            return
        try:
            from PyQt5.QtGui import QOpenGLContext
            if QOpenGLContext.currentContext() is None:
                return
            self.invalidate_display_list()
            self.body_display_list = glGenLists(1)
            glNewList(self.body_display_list, GL_COMPILE)
            self._draw_lower_chassis_box()
            self._draw_cabin_greenhouse()
            self._draw_front_face()
            self._draw_rear_face()
            glEndList()
        except Exception:
            self.body_display_list = None

    def set_livery_preset(self, idx: int):
        if idx == 0:  # Urban Slate Blue
            self.col_body = (0.16, 0.38, 0.65)
        elif idx == 1:  # Rosso Corsa / Sport
            self.col_body = (0.82, 0.15, 0.12)
        elif idx == 2:  # Adventure Khaki Green
            self.col_body = (0.18, 0.32, 0.22)
        elif idx == 3:  # Platinum Grey
            self.col_body = (0.35, 0.38, 0.42)
        elif idx == 4:  # Sunset Orange
            self.col_body = (0.88, 0.38, 0.08)
        self.invalidate_display_list()

    def draw_ghost(self, state=None):
        """Renders the translucent 3D Ghost Car with satin cyan alpha blending."""
        if not HAS_OPENGL:
            return
        try:
            from PyQt5.QtGui import QOpenGLContext
            if QOpenGLContext.currentContext() is None:
                return
        except Exception:
            pass

        old_body = self.col_body
        old_cladding = self.col_cladding
        self.col_body = (0.25, 0.80, 0.98)
        self.col_cladding = (0.15, 0.50, 0.70)

        glColor4f(0.25, 0.80, 0.98, 0.35)
        self._draw_lower_chassis_box()
        self._draw_cabin_greenhouse()
        self._draw_front_face()
        self._draw_rear_face()

        self.col_body = old_body
        self.col_cladding = old_cladding

    def draw(self, state=None, is_cockpit=False, skip_ao=False):
        if not HAS_OPENGL:
            return
        try:
            from PyQt5.QtGui import QOpenGLContext
            if QOpenGLContext.currentContext() is None:
                return
        except Exception:
            pass
        if state is None:
            state = {}

        steer_deg = float(state.get('steer_deg', 0.0))
        steer_rad = np.deg2rad(steer_deg)

        z_fl = float(state.get('z_fl', 0.0)) / 1000.0 if abs(state.get('z_fl', 0.0)) > 0.5 else float(state.get('z_fl', 0.0))
        z_fr = float(state.get('z_fr', 0.0)) / 1000.0 if abs(state.get('z_fr', 0.0)) > 0.5 else float(state.get('z_fr', 0.0))
        z_rl = float(state.get('z_rl', 0.0)) / 1000.0 if abs(state.get('z_rl', 0.0)) > 0.5 else float(state.get('z_rl', 0.0))
        z_rr = float(state.get('z_rr', 0.0)) / 1000.0 if abs(state.get('z_rr', 0.0)) > 0.5 else float(state.get('z_rr', 0.0))

        # SUV Geometry: Wheelbase L = 2.56m, Track = 1.54m
        a = 1.024
        b = 1.536
        tf_half = 0.77

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)

        # 1-4. Static Body Geometry (using compiled display list)
        if self.body_display_list is None:
            self.compile_body_display_list()
        if self.body_display_list is not None:
            glCallList(self.body_display_list)
        else:
            self._draw_lower_chassis_box()
            self._draw_cabin_greenhouse()
            self._draw_front_face()
            self._draw_rear_face()

        # 5. Wheels (4x 215/60 R17 Road Wheels with dynamic steering & spin)
        steer_fl = -steer_rad * (0.95 if steer_rad > 0 else 1.05)
        steer_fr = -steer_rad * (1.05 if steer_rad > 0 else 0.95)

        wh_ang = [
            float(state.get('wheel_angle_fl', 0.0)),
            float(state.get('wheel_angle_fr', 0.0)),
            float(state.get('wheel_angle_rl', 0.0)),
            float(state.get('wheel_angle_rr', 0.0))
        ]

        self._draw_suv_wheel(a, tf_half, steer_fl, z_fl, True, True, wh_ang[0])
        self._draw_suv_wheel(a, -tf_half, steer_fr, z_fr, True, False, wh_ang[1])
        if not is_cockpit:
            self._draw_suv_wheel(-b, tf_half, 0.0, z_rl, False, True, wh_ang[2])
            self._draw_suv_wheel(-b, -tf_half, 0.0, z_rr, False, False, wh_ang[3])

    def _draw_lower_chassis_box(self):
        """Clean rectangular body volume with dark protective lower rocker."""
        x_front = 1.45
        x_rear = -2.05
        y_half = 0.82
        z_bottom = 0.22
        z_sill = 0.42
        z_belt = 0.85

        # 1. Dark Protective Lower Sill (Cladding)
        glColor3f(*self.col_cladding)
        glBegin(GL_QUADS)
        # Left sill
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(x_front, y_half, z_sill)
        glVertex3f(x_rear, y_half, z_sill)
        glVertex3f(x_rear, y_half, z_bottom)
        glVertex3f(x_front, y_half, z_bottom)
        # Right sill
        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(x_front, -y_half, z_sill)
        glVertex3f(x_front, -y_half, z_bottom)
        glVertex3f(x_rear, -y_half, z_bottom)
        glVertex3f(x_rear, -y_half, z_sill)
        # Bottom undertray
        glNormal3f(0.0, 0.0, -1.0)
        glVertex3f(x_front, y_half, z_bottom)
        glVertex3f(x_rear, y_half, z_bottom)
        glVertex3f(x_rear, -y_half, z_bottom)
        glVertex3f(x_front, -y_half, z_bottom)
        # Front bumper lower
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(x_front, y_half, z_sill)
        glVertex3f(x_front, -y_half, z_sill)
        glVertex3f(x_front, -y_half, z_bottom)
        glVertex3f(x_front, y_half, z_bottom)
        # Rear bumper lower
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(x_rear, -y_half, z_sill)
        glVertex3f(x_rear, y_half, z_sill)
        glVertex3f(x_rear, y_half, z_bottom)
        glVertex3f(x_rear, -y_half, z_bottom)
        glEnd()

        # 2. Main Body Hull (Livery color)
        glColor3f(*self.col_body)
        glBegin(GL_QUADS)
        # Left side panel
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(x_front, y_half, z_belt)
        glVertex3f(x_rear, y_half, z_belt)
        glVertex3f(x_rear, y_half, z_sill)
        glVertex3f(x_front, y_half, z_sill)
        # Right side panel
        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(x_front, -y_half, z_belt)
        glVertex3f(x_front, -y_half, z_sill)
        glVertex3f(x_rear, -y_half, z_sill)
        glVertex3f(x_rear, -y_half, z_belt)
        # Hood Top Deck (Front hood plane)
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(x_front, y_half, z_belt)
        glVertex3f(x_front, -y_half, z_belt)
        glVertex3f(0.35, -y_half, z_belt)
        glVertex3f(0.35, y_half, z_belt)
        # Rear tailgate lower panel
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(x_rear, y_half, z_belt)
        glVertex3f(x_rear, -y_half, z_belt)
        glVertex3f(x_rear, -y_half, z_sill)
        glVertex3f(x_rear, y_half, z_sill)
        glEnd()

    def _draw_cabin_greenhouse(self):
        """Clean upper rectangular cabin volume with tinted windows and flat roof."""
        x_ws_base = 0.35
        x_roof_front = 0.05
        x_rear = -1.95
        y_cab = 0.70
        z_belt = 0.85
        z_roof = 1.46

        # 1. Raked Tinted Windshield
        glColor3f(*self.col_glass)
        glBegin(GL_QUADS)
        glNormal3f(0.65, 0.0, 0.75)
        glVertex3f(x_ws_base, y_cab, z_belt)
        glVertex3f(x_ws_base, -y_cab, z_belt)
        glVertex3f(x_roof_front, -y_cab, z_roof)
        glVertex3f(x_roof_front, y_cab, z_roof)
        glEnd()

        # 2. Side Windows (Left & Right)
        glBegin(GL_QUADS)
        # Left windows
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(x_ws_base, y_cab, z_belt)
        glVertex3f(x_roof_front, y_cab, z_roof)
        glVertex3f(x_rear, y_cab, z_roof)
        glVertex3f(x_rear, y_cab, z_belt)
        # Right windows
        glNormal3f(0.0, -1.0, 0.0)
        glVertex3f(x_ws_base, -y_cab, z_belt)
        glVertex3f(x_rear, -y_cab, z_belt)
        glVertex3f(x_rear, -y_cab, z_roof)
        glVertex3f(x_roof_front, -y_cab, z_roof)
        # Rear Window (Vertical tailgate glass)
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(x_rear, y_cab, z_roof)
        glVertex3f(x_rear, -y_cab, z_roof)
        glVertex3f(x_rear, -y_cab, z_belt)
        glVertex3f(x_rear, y_cab, z_belt)
        glEnd()

        # 3. Clean Flat Roof (Body color)
        glColor3f(*self.col_body)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 0.0, 1.0)
        glVertex3f(x_roof_front, y_cab, z_roof)
        glVertex3f(x_roof_front, -y_cab, z_roof)
        glVertex3f(x_rear, -y_cab, z_roof)
        glVertex3f(x_rear, y_cab, z_roof)
        glEnd()

    def _draw_front_face(self):
        """Clean front fascia with dark grille box and twin LED headlights."""
        x_front = 1.455
        z_belt = 0.85
        z_sill = 0.42

        # Obsidian Black Center Grille
        glColor3f(*self.col_grille)
        glBegin(GL_QUADS)
        glNormal3f(1.0, 0.0, 0.0)
        glVertex3f(x_front, 0.45, z_belt - 0.02)
        glVertex3f(x_front, -0.45, z_belt - 0.02)
        glVertex3f(x_front, -0.45, z_sill + 0.08)
        glVertex3f(x_front, 0.45, z_sill + 0.08)
        glEnd()

        # Twin Rectangular LED Headlights
        glColor3f(*self.col_lights_front)
        glBegin(GL_QUADS)
        glNormal3f(1.0, 0.0, 0.0)
        # Left Headlight
        glVertex3f(x_front, 0.76, z_belt - 0.03)
        glVertex3f(x_front, 0.48, z_belt - 0.03)
        glVertex3f(x_front, 0.48, z_belt - 0.18)
        glVertex3f(x_front, 0.76, z_belt - 0.18)
        # Right Headlight
        glVertex3f(x_front, -0.48, z_belt - 0.03)
        glVertex3f(x_front, -0.76, z_belt - 0.03)
        glVertex3f(x_front, -0.76, z_belt - 0.18)
        glVertex3f(x_front, -0.48, z_belt - 0.18)
        glEnd()

    def _draw_rear_face(self):
        """Full-width horizontal red LED taillight bar."""
        x_rear = -2.055
        z_light_top = 0.83
        z_light_bot = 0.74

        glColor3f(*self.col_lights_rear)
        glBegin(GL_QUADS)
        glNormal3f(-1.0, 0.0, 0.0)
        glVertex3f(x_rear, 0.76, z_light_top)
        glVertex3f(x_rear, -0.76, z_light_top)
        glVertex3f(x_rear, -0.76, z_light_bot)
        glVertex3f(x_rear, 0.76, z_light_bot)
        glEnd()

    def _draw_suv_wheel(self, wx, wy, steer_angle, z_disp, is_front, is_left, wh_rot_rad):
        """Clean 215/60 R17 road wheel with alloy rim."""
        wz = 0.315 + z_disp

        glPushMatrix()
        glTranslatef(wx, wy, wz)
        if is_front:
            glRotatef(np.rad2deg(steer_angle), 0, 0, 1)

        # Dynamic Wheel Rolling Spin
        glPushMatrix()
        glRotatef(np.rad2deg(wh_rot_rad), 0, 1 if is_left else -1, 0)

        # Tire Tread Cylinder (215/60 R17, Rw = 0.315m, width = 0.22m)
        w_rad = 0.315
        w_wid = 0.22
        n_seg = 20

        glColor3f(0.12, 0.13, 0.15)
        glBegin(GL_QUAD_STRIP)
        for i in range(n_seg + 1):
            ang = i * 2.0 * math.pi / n_seg
            cz = w_rad * math.sin(ang)
            cx = w_rad * math.cos(ang)
            glNormal3f(math.cos(ang), 0.0, math.sin(ang))
            glVertex3f(cx, w_wid / 2.0, cz)
            glVertex3f(cx, -w_wid / 2.0, cz)
        glEnd()

        # Silver Alloy Rim Face with 5 Clean Spokes
        out_side = (w_wid / 2.0) if is_left else (-w_wid / 2.0)
        glColor3f(*self.col_alloy)
        glLineWidth(2.4)
        glBegin(GL_LINES)
        for i in range(5):
            ang = i * 2.0 * math.pi / 5.0
            glVertex3f(0.0, out_side, 0.0)
            glVertex3f(0.18 * math.cos(ang), out_side, 0.18 * math.sin(ang))
        glEnd()

        glPopMatrix() # end rotating assembly
        glPopMatrix() # end wheel
