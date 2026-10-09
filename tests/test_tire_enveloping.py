import unittest
import numpy as np
from src.core.road_profile import RoadProfile, RoadProfileConfig

class TestTireEnvelopingAndISO8608(unittest.TestCase):
    def setUp(self):
        self.s_arr = np.linspace(0, 100, 100)
        self.config = RoadProfileConfig(
            enable_macro=False, 
            enable_micro=True, 
            micro_roughness_amp_mm=5.0, 
            
        )
        self.rp = RoadProfile.flat(self.s_arr, config=self.config)

    def test_iso8608_generation(self):
        """Verifies that the fast ISO 8608 approximation generates non-zero roughness."""
        z_micro = self.rp.get_micro_unevenness(s_query=10.0, lateral_offset_y=0.0)
        self.assertNotEqual(z_micro, 0.0)
        self.assertTrue(abs(z_micro) < 0.1) # Shouldn't exceed 10cm for Class C at 5mm base
        
    def test_tire_enveloping_filter(self):
        """Verifies that get_wheel_contact_heights applies the 3-point enveloping filter."""
        heights = self.rp.get_wheel_contact_heights(
            s_center=15.0, a_dist=1.5, b_dist=1.5, t_f=1.2, t_r=1.2
        )
        self.assertIn('fl', heights)
        self.assertIn('rr', heights)
        self.assertNotEqual(heights['fl'], 0.0)

    def test_micro_disabled(self):
        """Verifies that disabling micro roughness returns exactly 0."""
        self.config.enable_micro = False
        z_micro = self.rp.get_micro_unevenness(s_query=10.0, lateral_offset_y=0.0)
        self.assertEqual(z_micro, 0.0)
        
        heights = self.rp.get_wheel_contact_heights(
            s_center=15.0, a_dist=1.5, b_dist=1.5, t_f=1.2, t_r=1.2
        )
        self.assertEqual(heights['fl'], 0.0)

if __name__ == '__main__':
    unittest.main()
