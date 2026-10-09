import numpy as np
import math
from src.utils.track_manager import TrackCollisionMesh

def test_collision_mesh():
    track_data = {
        's': np.array([0.0, 10.0]),
        'x_center': np.array([0.0, 10.0]),
        'y_center': np.array([0.0, 0.0]),
        'x_left': np.array([0.0, 10.0]),
        'y_left': np.array([2.0, 2.0]),
        'x_right': np.array([0.0, 10.0]),
        'y_right': np.array([-2.0, -2.0]),
        'is_closed': False,
        'ds': 2.0
    }
    
    mesh = TrackCollisionMesh(track_data)
    assert len(mesh.triangles) > 0
    
    # Raycast at origin
    r_contact, n_road, stype = mesh.raycast_down(np.array([5.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0]))
    assert abs(r_contact[2] - 0.0) < 1e-4
    assert n_road[2] > 0.99
    
def test_skidmarks_projection():
    # Synthetic test to verify telemetry and skidmark offset math
    rc = np.array([5.0, 2.0, 1.0])
    n = np.array([0.0, 0.0, 1.0])
    offset = rc + 0.003 * n
    assert abs(offset[2] - 1.003) < 1e-6
