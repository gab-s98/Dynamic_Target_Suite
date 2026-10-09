import dataclasses
import numpy as np

@dataclasses.dataclass
class VehicleParams:
    """
    Data structure containing the physical parameters of the vehicle.
    Parameters are grouped by model tier:
    
    3-DOF:  m, Iz, g, L, t_f, t_r, h_cg, R_w, wd_f, rho, S, Cz, aero_balance
            + LMUX/LMUY (uncertainty), tire pressure (scaling)
    7-DOF:  All 3-DOF + Iw, brake_bias, t_split_rear
            + camber_f_static, camber_r_static, toe_f_static, toe_r_static
    14-DOF: All 7-DOF + unsprung masses, roll/ride stiffness & damping
            + kinematic polynomial maps (from Curve Digitizer)
    """
    # -----------------------------------------------------------------------
    # TIER 0 — Mass & Inertia (all models)
    # -----------------------------------------------------------------------
    m: float = 300.0      # Total mass [kg] (Baseline with driver included)
    Iz: float = 125.0     # Yaw moment of inertia [kg·m²]
    Ix: float = 38.0      # Roll moment of inertia [kg·m²]
    Iy: float = 115.0     # Pitch moment of inertia [kg·m²]
    Iw: float = 1.2       # Wheel rotational inertia baseline [kg·m²]
    Iw_f: float = 0.30    # Front wheel rotational inertia [kg·m²] (unpowered 10" rim + slick + disc + hub)
    Iw_r: float = 1.20    # Rear wheel rotational inertia [kg·m²] (driven assembly + motor rotor)
    Ixz: float = 0.0      # Cross-product of inertia [kg·m²] (roll-yaw coupling)
    g: float = 9.81       # Gravity [m/s²]

    # -----------------------------------------------------------------------
    # TIER 0 — Geometry (all models)
    # -----------------------------------------------------------------------
    L:    float = 1.550   # Wheelbase [m]
    t_f:  float = 1.220   # Front track width [m]
    t_r:  float = 1.200   # Rear track width [m]
    h_cg: float = 0.300   # CoG height above ground [m]
    R_w:  float = 0.228   # Effective tire radius [m]

    # -----------------------------------------------------------------------
    # TIER 0 — MDO base parameters (3-DOF optimizer)
    # -----------------------------------------------------------------------
    wd_f:         float = 0.500  # Static weight distribution (front fraction)
    lltd:         float = 0.520  # Lateral Load Transfer Distribution (front fraction)
    aero_balance: float = 0.500  # Aero downforce balance (front fraction)

    # -----------------------------------------------------------------------
    # TIER 0 — Aerodynamics (all models)
    # -----------------------------------------------------------------------
    rho: float = 1.225    # Air density [kg/m³]
    S:   float = 1.150    # Reference area [m²]
    Cz:  float = 2.800    # Total downforce coefficient (negative lift)
    Cd:  float = 1.300    # Total drag coefficient
    crr: float = 0.015    # Rolling resistance coefficient [-]

    # -----------------------------------------------------------------------
    # TIER 0 — Tire Uncertainty (all models)
    # -----------------------------------------------------------------------
    # These are PERCENTAGE MULTIPLIERS applied on top of the LMUY/LMUX values
    # already present in the .tir file ([SCALING_COEFFICIENTS] section).
    #
    # Effective grip = (PDY1 + PDY2*dfz) * LMUY_tir_file * LMUY_here
    #
    # 1.0 = nominal (use the .tir file exactly as provided)
    # 0.9 = simulate 10% grip reduction (cold tires, rain, worn rubber)
    # 1.1 = simulate 10% grip increase (optimal temperature, new compound)
    #
    # To run an uncertainty sweep: keep 1.0 in VehicleParams and vary it
    # externally, or add it to the optimizer design variables.
    LMUX: float = 1.0     # Longitudinal friction multiplier (% on .tir LMUX) [-]
    LMUY: float = 1.0     # Lateral friction multiplier (% on .tir LMUY) [-]

    # Cold tire inflation pressure [bar] (FSAE typical: 0.85 bar / 12.3 psi)
    tire_pressure_f_bar: float = 0.85
    tire_pressure_r_bar: float = 0.85
    tire_vertical_stiffness_12psi: float = 75000.0 # Tire vertical stiffness [N/m]
    tire_file: str = "Simple_Load_Dependent.tir"

    # -----------------------------------------------------------------------
    # TIER 0 / 2 — Tire Relaxation Lengths (Contact Patch Transient Model)
    # -----------------------------------------------------------------------
    sigma_x: float = 0.15  # Longitudinal relaxation length [m]
    sigma_y: float = 0.22  # Lateral relaxation length [m]

    # -----------------------------------------------------------------------
    # TIER 1 — Powertrain & Braking (7-DOF+)
    # -----------------------------------------------------------------------
    vehicle_type:    str = "formula" # "formula" (FSAE single-seater) or "b_segment_suv" (Segment B compact SUV)
    drivetrain_type: str = "rwd"     # "rwd" (rear-wheel drive), "fwd" (front-wheel drive), or "awd"
    diff_type:       str = "open"    # "open", "spool", "salisbury_lsd", "active_tv"
    diff_mode:       str = "DUAL_MOTOR_OPEN" # "DUAL_MOTOR_OPEN", "VIRTUAL_SPOOL", "VIRTUAL_SALISBURY_LSD", "ACTIVE_TORQUE_VECTORING"
    diff_spool_k:    float = 80.0    # Virtual Spool internal torsional stiffness [Nm/(rad/s)]
    diff_lsd_preload: float = 35.0   # Salisbury LSD static preload torque [Nm]
    diff_lsd_mu_power: float = 0.40  # Salisbury LSD locking ratio under power / acceleration
    diff_lsd_mu_coast: float = 0.20  # Salisbury LSD locking ratio under coast / deceleration
    tv_kp:           float = 450.0   # Active TV yaw rate error proportional gain [Nm/(rad/s)]
    tv_ki:           float = 80.0    # Active TV yaw rate error integral gain [Nm/rad]
    tv_max_delta_mz: float = 350.0   # Maximum corrective yaw moment [Nm]
    tv_kus:          float = 0.0025  # Linear bicycle understeer gradient [rad/(m/s^2)]
    brake_bias:    float = 0.60  # Front braking fraction
    t_split_rear:  float = 0.50  # Rear torque split (0=full left, 1=full right)
    
    # Powertrain parameters (Generic prototype electric powertrain)
    motor_type:    str = "GENERIC_ELECTRIC_POWERTRAIN" # Electric motor model
    P_max_kw:      float = 70.0  # Total accumulator electrical power limit [kW]
    P_max_motor_kw: float = 45.0 # Single motor mechanical peak power [kW]
    T_max_motor:   float = 85.0  # Single motor peak torque [Nm]
    final_drive:   float = 4.0   # Final drive reduction ratio
    n_drivetrain:  float = 0.95  # Drivetrain mechanical efficiency
    T_brake_max_total: float = 1400.0  # Total maximum braking torque at all wheels [Nm]
    tire_thermal_mass: float = 2000.0  # Tire tread bulk thermal capacity [J/K] (FSAE: 2000 J/K; Road tire: 16000 J/K)
    tire_h_conv:       float = 8.5     # Tire convective cooling coefficient [W/K] (FSAE: 8.5 W/K; Road tire: 32 W/K)

    # -----------------------------------------------------------------------
    # TIER 1 — Static Kinematics (7-DOF+ design variables)
    # These are the "nominal" values at design ride height / zero roll.
    # In the 7-DOF trim, they directly affect slip angles and thus the lateral
    # force balance. Can be included as NLP design variables.
    # -----------------------------------------------------------------------
    camber_f_static: float = -1.5  # Front static camber [deg], negative = lean in
    camber_r_static: float = -1.0  # Rear static camber [deg]
    toe_f_static:    float =  0.0  # Front static toe [deg], positive = toe-out
    toe_r_static:    float =  0.1  # Rear static toe [deg], positive = toe-in

    # Suspension Kinematics: Caster & KPI for diagonal corner weight jacking
    caster_f:           float =  3.5   # Front Caster angle [deg]
    kpi_f:              float =  8.0   # Front Kingpin Inclination angle [deg]
    scrub_radius_f:     float =  0.035 # Front Scrub radius [m]
    mechanical_trail_f: float =  0.015 # Front Mechanical trail [m]
    spindle_offset_f:   float =  0.12  # Front spindle offset (king-pin axis -> wheel centre) [m]
    spindle_offset_r:   float =  0.10  # Rear spindle offset [m]

    # Steering system
    steering_ratio: float = 5.0           # Hand-wheel to road-wheel ratio [-]
    steering_wheel_max_deg: float = 125.0 # Max hand-wheel / pinion angle [deg]

    # -----------------------------------------------------------------------
    # TIER 1 — Roll Centers (7-DOF+, from geometry)
    # -----------------------------------------------------------------------
    q1: float = 0.050   # Front roll center height [m]
    q2: float = 0.070   # Rear roll center height [m]

    # -----------------------------------------------------------------------
    # TIER 2 — Unsprung Masses (14-DOF)
    # -----------------------------------------------------------------------
    m_us_f: float = 18.0    # Front unsprung mass per corner [kg] (hub + disc + caliper + upright)
    m_us_r: float = 16.0    # Rear unsprung mass per corner [kg]  (hub + disc + upright)
    C_tire_z: float = 500.0  # Vertical tire damping coefficient [N·s/m]

    # -----------------------------------------------------------------------
    # TIER 2 — Roll Stiffness & Damping (14-DOF design variables)
    # -----------------------------------------------------------------------
    K_phi_f: float = 30000.0  # Front roll stiffness [Nm/rad]
    K_phi_r: float = 28000.0  # Rear roll stiffness [Nm/rad]
    K_phi_arb_f: float = 10000.0  # Front ARB lumped roll stiffness [Nm/rad]
    K_phi_arb_r: float = 7000.0   # Rear ARB lumped roll stiffness [Nm/rad]
    Kw_arb_f: float = 0.0         # Front ARB differential wheel rate [N/m] (if 0, derived from K_phi_arb_f / t_f^2)
    Kw_arb_r: float = 0.0         # Rear ARB differential wheel rate [N/m] (if 0, derived from K_phi_arb_r / t_r^2)
    C_phi_f: float = 800.0    # Front roll damping [Nm·s/rad]
    C_phi_r: float = 600.0    # Rear roll damping [Nm·s/rad]

    # -----------------------------------------------------------------------
    # TIER 2 — Wheel Rate / Ride Stiffness (14-DOF design variables)
    # Wheel rate = spring rate at the wheel: K_z = k_spring * 1000 * MR_damper^2 [N/m]
    # -----------------------------------------------------------------------
    K_z_f: float = 35000.0   # Front wheel rate [N/m]
    K_z_r: float = 35000.0   # Rear wheel rate [N/m]
    C_z_f: float = 1500.0    # Front ride damping [N·s/m]
    C_z_r: float = 1200.0    # Rear ride damping [N·s/m]

    # -----------------------------------------------------------------------
    # TIER 2 — Real Damper Limits, Bump Stops & Physical Springs (14-DOF)
    # -----------------------------------------------------------------------
    damper_stroke_total_mm: float = 57.0         # Total physical damper stroke [mm]
    damper_stroke_bump_max_m: float = 0.060      # Max wheel travel in bump from 1g [m]
    damper_stroke_rebound_max_m: float = -0.012  # Max wheel travel in rebound from 1g [m]
    bumpstop_gap_f_m: float = 0.055             # Free wheel travel before bumpstop contact front [m]
    bumpstop_gap_r_m: float = 0.055             # Free wheel travel before bumpstop contact rear [m]
    bumpstop_contact_tail_mm: float = 5.0       # Damper stroke before full compression where bumpstop contacts [mm]
    bumpstop_pack_tail_mm: float = 3.5          # Damper stroke before full compression where solid pack occurs [mm]
    bumpstop_length_f_mm: float = 5.0           # Front bumpstop active contact range on damper shaft [mm]
    bumpstop_length_r_mm: float = 5.0           # Rear bumpstop active contact range on damper shaft [mm]
    bumpstop_k_f: float = 80000.0               # Bumpstop progressive stiffness front [N/m]
    bumpstop_k_r: float = 80000.0               # Bumpstop progressive stiffness rear [N/m]
    pushrod_offset_f_mm: float = 0.0            # Front pushrod / kinematic offset L0 [mm]
    pushrod_offset_r_mm: float = 0.0            # Rear pushrod / kinematic offset L0 [mm]
    spring_free_length_f_mm: float = 150.0      # Front spring free length L0 [mm]
    spring_free_length_r_mm: float = 150.0      # Rear spring free length L0 [mm]
    spring_rate_installed_f_Nmm: float = 55.0   # Installed physical spring rate front [N/mm]
    spring_rate_installed_r_Nmm: float = 55.0   # Installed physical spring rate rear [N/mm]
    spring_preload_f_mm: float = 5.0            # Front spring installed preload [mm]
    spring_preload_r_mm: float = 5.0            # Rear spring installed preload [mm]
    preload_f_N: float = 200.0                  # Front spring preload force [N]
    preload_r_N: float = 250.0                  # Rear spring preload force [N]
    preload_f_nom_N: float = 200.0              # Front spring nominal preload force [N]
    preload_r_nom_N: float = 250.0              # Rear spring nominal preload force [N]
    h_cg_nominal: float = 0.280                 # Nominal CoG height at nominal ride height [m]
    nominal_ride_height_m: float = 0.050        # Nominal ground clearance [m] (5 cm)
    motion_ratio_f: float = 0.793775            # Front suspension damper motion ratio MR = d_damper / d_wheel [-]
    motion_ratio_r: float = 0.799511            # Rear suspension damper motion ratio MR = d_damper / d_wheel [-]
    motion_ratio_arb_f: float = 0.399877        # Front ARB motion ratio MR_arb = d_arb / d_wheel [-]
    motion_ratio_arb_r: float = 0.404921        # Rear ARB motion ratio MR_arb = d_arb / d_wheel [-]
    static_ride_height_f_m: float = 0.050       # Static ground clearance front [m] (50 mm / 5 cm)
    static_ride_height_r_m: float = 0.050       # Static ground clearance rear [m] (50 mm / 5 cm)

    # -----------------------------------------------------------------------
    # TIER 2 — Non-Linear 4-Way Damper Model (Öhlins TTX25 MkII FSAE)
    # -----------------------------------------------------------------------
    damper_mode: str = 'linear'                 # 'linear' (uses C_phi/C_z constant) or 'ttx25_nonlinear'
    damper_valving_f: str = 'C12/R12'           # Front valving ('C11/R11' .. 'C23/R23')
    damper_valving_r: str = 'C12/R12'           # Rear valving
    damper_lsc_clicks_f: float = 10.0           # Front Low-Speed Comp (0=closed/stiff, 25=open/soft)
    damper_hsc_turns_f: float = 2.0             # Front High-Speed Comp (0=open/soft, 4.3=closed/stiff)
    damper_lsr_clicks_f: float = 10.0           # Front Low-Speed Reb (0=closed/stiff, 25=open/soft)
    damper_hsr_turns_f: float = 2.0             # Front High-Speed Reb (0=open/soft, 4.3=closed/stiff)
    damper_lsc_clicks_r: float = 10.0           # Rear Low-Speed Comp
    damper_hsc_turns_r: float = 2.0             # Rear High-Speed Comp
    damper_lsr_clicks_r: float = 10.0           # Rear Low-Speed Reb
    damper_hsr_turns_r: float = 2.0             # Rear High-Speed Reb

    # -----------------------------------------------------------------------
    # TIER 2 — Kinematic Maps (from Curve Digitizer, 14-DOF)
    # Each map is stored as a list of 4 polynomial coefficients [a3, a2, a1, a0]
    # for a 3rd-degree polynomial:  y = a3*x^3 + a2*x^2 + a1*x + a0
    #
    # "predisposed_for_optimization" means the optimizer can search around
    # the base coefficients (e.g. ±10% on a1 which is the linear gain),
    # but the full poly shape comes from the Digitizer.
    #
    # Independent variable for wheel-travel maps: z [m], positive = compression
    # Independent variable for steering maps: delta [rad]
    # -----------------------------------------------------------------------

    # Camber gain vs wheel travel [deg/m -> output in deg]
    camber_gain_f_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, -15.0, 0.0]
    )
    camber_gain_r_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, -12.0, 0.0]
    )

    # Bump-steer (toe change) vs wheel travel [deg/m -> output in deg]
    toe_gain_f_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )
    toe_gain_r_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )

    # Roll Center height vs chassis roll angle phi [deg -> output in m]
    rollcenter_f_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.050]  # h_rc_f = 0.050 m static
    )
    rollcenter_r_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.070]  # h_rc_r = 0.070 m static
    )

    # Roll Center Lateral Migration vs chassis roll angle phi [deg -> output in m]
    rollcenter_lat_f_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.050, 0.0]
    )
    rollcenter_lat_r_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.040, 0.0]
    )

    # Anti-dive fraction vs wheel travel (front brake in bump) [-]
    anti_dive_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.250]
    )

    # Anti-squat fraction vs wheel travel (rear drive in bump) [-]
    anti_squat_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.150]
    )

    # Anti-lift fraction vs wheel travel (rear brake in rebound) [-]
    anti_lift_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )

    # Damper motion ratio polynomial vs wheel travel [-]
    damper_motion_ratio_f_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.800]
    )
    damper_motion_ratio_r_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.800]
    )

    # ARB motion ratio polynomial vs wheel travel [-]
    arb_motion_ratio_f_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.400]
    )
    arb_motion_ratio_r_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.400]
    )

    # Ackermann percentage vs steering wheel/pinion angle [deg]
    # Poly3: [a3, a2, a1, a0] evaluated on delta_sw [deg] -> output in [%]
    ackermann_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.001, 0.0, 50.0]   # 50.0% nominal at delta=0
    )
    # Ackermann percentage scalar: 0.0 = parallel steer, 1.0 = full geometric Ackermann
    ackermann_pct: float = 0.50

    # Caster Trail (Mechanical Trail) vs steering wheel angle delta_sw [deg -> output in m]
    caster_trail_f_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.015]
    )

    # Scrub Radius vs steering wheel angle delta_sw [deg -> output in m]
    scrub_radius_f_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.035]
    )

    # Camber Steer Delta vs steering wheel angle delta_sw [deg -> output in deg]
    camber_steer_f_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )

    # Wheel Steer Angle vs Steering Wheel / Pinion Angle delta_sw [deg]
    # Poly3 for Left Front (FL) wheel: [a3, a2, a1, a0] evaluated on delta_sw [deg] -> delta_fl [deg]
    wheel_steer_f_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0]
    )

    # -----------------------------------------------------------------------
    # TIER 2 — Aero Maps (from CFD / Wind Tunnel — 14-DOF)
    # Sensitivity to front & rear ride heights (zf, zr in meters, + = bump/compression)
    # Cz(zf, zr) = Cz_0 + dCz_dzf * zf + dCz_dzr * zr + d2Cz * (zf - zr)^2
    # -----------------------------------------------------------------------
    use_aeromap: bool = False
    aeromap_Cz_coeffs: list = dataclasses.field(
        default_factory=lambda: [2.800, 0.0, 0.0, 0.0]
    )
    aeromap_Cd_coeffs: list = dataclasses.field(
        default_factory=lambda: [1.300, 0.0, 0.0, 0.0]
    )
    aeromap_bal_coeffs: list = dataclasses.field(
        default_factory=lambda: [0.500, 0.0, 0.0, 0.0]
    )

    # -----------------------------------------------------------------------
    # Powertrain & Energy / Consumption (AutoX & Endurance FSAE)
    # -----------------------------------------------------------------------
    battery_capacity_kwh: float = 6.8   # Accumulator energy capacity [kWh]
    battery_voltage_nom:  float = 400.0 # Nominal pack voltage [V]
    regen_max_kw:         float = 40.0  # Max regen power limit [kW]
    regen_efficiency:     float = 0.65  # Round-trip regen efficiency
    motor_base_eff:       float = 0.92  # Base electric motor efficiency
    inverter_eff:         float = 0.96  # Inverter efficiency

    def eval_poly3(self, coeffs, x):
        """Evaluate a 3rd-degree polynomial: y = a3*x^3 + a2*x^2 + a1*x + a0"""
        a3, a2, a1, a0 = coeffs
        return a3*x**3 + a2*x**2 + a1*x + a0

    def eval_aeromap(self, z_f: float, z_r: float):
        """
        Evaluate Aerodynamic Coefficients considering pitch/ride heights.
        Returns: (Cz, Cd, aero_balance)
        """
        if not self.use_aeromap:
            return self.Cz, self.Cd, self.aero_balance
            
        c_z0, d_zf_z, d_zr_z, d2_z = self.aeromap_Cz_coeffs
        Cz_val = max(c_z0 + d_zf_z * z_f + d_zr_z * z_r + d2_z * (z_f - z_r)**2, 0.1)
        
        c_d0, d_zf_d, d_zr_d, d2_d = self.aeromap_Cd_coeffs
        Cd_val = max(c_d0 + d_zf_d * z_f + d_zr_d * z_r + d2_d * (z_f - z_r)**2, 0.05)
        
        b_0, d_zf_b, d_zr_b, d2_b = self.aeromap_bal_coeffs
        bal_val = np.clip(b_0 + d_zf_b * z_f + d_zr_b * z_r + d2_b * (z_f - z_r)**2, 0.1, 0.9)
        
        return float(Cz_val), float(Cd_val), float(bal_val)

    def get_powertrain(self):
        """
        Returns the EMRAX188HV powertrain instance configured with vehicle parameters.
        """
        from src.models.powertrain_electric import EMRAX188HV
        return EMRAX188HV(
            gear_ratio=self.final_drive,
            p_elec_limit_kw=self.P_max_kw,
            n_drivetrain=self.n_drivetrain
        )

    @property
    def a(self) -> float:
        """Distance from front axle to CoG [m]: a = (1 - wd_f) * L"""
        return self.L * (1.0 - self.wd_f)

    @property
    def b(self) -> float:
        """Distance from rear axle to CoG [m]: b = wd_f * L"""
        return self.L * self.wd_f

    @property
    def wheelbase(self) -> float:
        return self.L

    @property
    def mass(self) -> float:
        return self.m

    @property
    def track_f(self) -> float:
        return self.t_f

    @property
    def track_r(self) -> float:
        return self.t_r

    @property
    def tire_radius(self) -> float:
        return self.R_w

    @property
    def cg_dist_f(self) -> float:
        return self.a

    @property
    def cg_dist_r(self) -> float:
        return self.b

    @property
    def C_alpha_f(self) -> float:
        return 120000.0

    @property
    def C_alpha_r(self) -> float:
        return 130000.0

    @property
    def spring_rate_f(self) -> float:
        return self.K_z_f

    @property
    def spring_rate_r(self) -> float:
        return self.K_z_r

    @property
    def road_wheel_max_deg(self) -> float:
        """Maximum road-wheel angle [deg] derived from steering system: steering_wheel_max / steering_ratio."""
        return self.steering_wheel_max_deg / max(self.steering_ratio, 0.1)

    @property
    def dynamic_index_z(self) -> float:
        """Yaw Dynamic Index: DI_z = Iz / (m * a * b)"""
        denom = self.m * self.a * self.b
        return float(self.Iz / max(denom, 1e-4))

    @property
    def z_wheel_tot_f(self) -> float:
        """Total theoretical geometric wheel travel front [m]: shock_stroke / MR."""
        mr = max(self.motion_ratio_f, 0.01)
        return (self.damper_stroke_total_mm / 1000.0) / mr

    @property
    def z_wheel_tot_r(self) -> float:
        """Total theoretical geometric wheel travel rear [m]: shock_stroke / MR."""
        mr = max(self.motion_ratio_r, 0.01)
        return (self.damper_stroke_total_mm / 1000.0) / mr

    @property
    def z_wheel_tot(self) -> float:
        """Average total geometric wheel travel [m]: shock_stroke / MR."""
        return 0.5 * (self.z_wheel_tot_f + self.z_wheel_tot_r)

    @property
    def z_wheel_max(self) -> float:
        """Maximum total geometric wheel travel [m]: shock_stroke / MR."""
        return self.z_wheel_tot

    def update_inertias_from_geometry(self, di_z: float = 1.0, di_y: float = 1.0, di_x: float = 0.25, ixz: float = 0.0):
        """
        Overrides the declared Ix, Iy, Iz, Ixz fields with values derived from vehicle geometry
        using the Dynamic Index formulation:

        - Iz  = di_z * m * a * b  (Yaw MoI;   typical FSAE: di_z ≈ 0.95 .. 1.15)
        - Iy  = di_y * m * a * b  (Pitch MoI; typical FSAE: di_y ≈ 0.90 .. 1.10)
        - Ix  = di_x * m * (t_avg/2)²  (Roll MoI; typical FSAE: di_x ≈ 0.20 .. 0.30)
        - Ixz = ixz                (Roll-yaw cross-inertia; ~0 for symmetric FSAE cars)

        Note: Ix, Iy, Iz, Ixz are declared dataclass fields with physical defaults.
        This method is optional — call it only when you want geometry-consistent inertias
        (e.g. after changing L, t_f, t_r, m, wd_f). If not called, the defaults from
        VehicleParams (Ix=35.0, Iy=110.0, Iz=120.0) are used directly.
        """
        self.Iz = float(di_z * self.m * self.a * self.b)
        self.Iy = float(di_y * self.m * self.a * self.b)
        t_avg = (self.t_f + self.t_r) / 2.0
        self.Ix = float(di_x * self.m * ((t_avg / 2.0)**2))
        self.Ixz = float(ixz)

    def get_heave_stiffness(self) -> float:
        """Total vertical heave stiffness K_heave [N/m] (2*K_z_f + 2*K_z_r)."""
        return 2.0 * self.K_z_f + 2.0 * self.K_z_r

    @property
    def spring_installed_length_f_mm(self) -> float:
        """Physical installed length of front spring at full droop / free shock [mm]: L_inst = L0 - Delta_L_pre."""
        return float(self.spring_free_length_f_mm - self.spring_preload_f_mm)

    @property
    def spring_installed_length_r_mm(self) -> float:
        """Physical installed length of rear spring at full droop / free shock [mm]: L_inst = L0 - Delta_L_pre."""
        return float(self.spring_free_length_r_mm - self.spring_preload_r_mm)

    def get_Kw_arb_f(self) -> float:
        """Front differential wheel rate from ARB [N/m]: Kw_arb = K_phi_arb / t_f^2."""
        if getattr(self, 'Kw_arb_f', 0.0) > 0.0:
            return float(self.Kw_arb_f)
        return float(getattr(self, 'K_phi_arb_f', 12000.0) / (self.t_f**2 + 1e-6))

    def get_Kw_arb_r(self) -> float:
        """Rear differential wheel rate from ARB [N/m]: Kw_arb = K_phi_arb / t_r^2."""
        if getattr(self, 'Kw_arb_r', 0.0) > 0.0:
            return float(self.Kw_arb_r)
        return float(getattr(self, 'K_phi_arb_r', 8000.0) / (self.t_r**2 + 1e-6))

    def set_Kphi_arb_from_Kw(self, kw_f: float, kw_r: float):
        """Set ARB roll stiffness [Nm/rad] from desired wheel rates [N/m]: K_phi = Kw * t^2."""
        self.Kw_arb_f = float(kw_f)
        self.Kw_arb_r = float(kw_r)
        self.K_phi_arb_f = float(kw_f * (self.t_f**2))
        self.K_phi_arb_r = float(kw_r * (self.t_r**2))

    def get_pitch_stiffness(self) -> float:
        """Total pitch stiffness K_theta [Nm/rad] around CoG."""
        return 2.0 * self.K_z_f * (self.a**2) + 2.0 * self.K_z_r * (self.b**2)

    def get_roll_stiffness_total(self) -> float:
        """Total chassis roll stiffness K_phi_tot [Nm/rad]."""
        return self.K_phi_f + self.K_phi_r

    @property
    def lltd_elastic(self) -> float:
        """Elastic roll moment distribution from front and rear roll stiffnesses."""
        k_tot = self.K_phi_f + self.K_phi_r
        return float(self.K_phi_f / k_tot) if k_tot > 0 else float(self.lltd)

    def get_effective_lltd(self, ay_g=1.0) -> float:
        """
        Calculates the true total Lateral Load Transfer Distribution (LLTD)
        including both elastic roll stiffnesses (K_phi_f, K_phi_r) and geometric roll center load transfer.
        """
        m = self.m
        m_s = m * (1.0 - 0.12)
        wd_f = self.wd_f
        m_f = m * wd_f
        m_r = m * (1.0 - wd_f)
        q1 = getattr(self, 'q1', 0.046)
        q2 = getattr(self, 'q2', 0.077)
        tf = max(self.t_f, 0.5)
        tr = max(self.t_r, 0.5)
        h_cg = getattr(self, 'h_cg', 0.28)
        
        # Roll axis height at CG
        h_ra = q1 + (q2 - q1) * (1.0 - wd_f)
        h_arm = max(h_cg - h_ra, 0.02)
        k_tot = max(self.K_phi_f + self.K_phi_r, 100.0)
        
        ay = ay_g * 9.81
        phi = (m_s * ay * h_arm) / max(k_tot - m_s * 9.81 * h_arm, 10.0)
        
        dFz_f = (self.K_phi_f * phi + m_f * ay * q1) / tf
        dFz_r = (self.K_phi_r * phi + m_r * ay * q2) / tr
        
        M_f = dFz_f * tf
        M_r = dFz_r * tr
        if (M_f + M_r) > 1e-3:
            return float(M_f / (M_f + M_r))
        return float(self.lltd_elastic)

    @property
    def K_z_f_derived(self) -> float:
        """Front Wheel Rate [N/m] derived from installed physical spring and motion ratio."""
        return float(self.spring_rate_installed_f_Nmm * 1000.0 * (self.motion_ratio_f ** 2))

    @property
    def K_z_r_derived(self) -> float:
        """Rear Wheel Rate [N/m] derived from installed physical spring and motion ratio."""
        return float(self.spring_rate_installed_r_Nmm * 1000.0 * (self.motion_ratio_r ** 2))

    def sync_wheel_rates_and_springs(self, priority='spring'):
        """
        Synchronizes Wheel Rate (K_z) and Installed Spring Rate (k_spring).
        If priority == 'spring', K_z is computed from installed spring: K_z = k_spring * 1000 * MR^2.
        If priority == 'wheel_rate', installed spring is computed from K_z: k_spring = K_z / (1000 * MR^2).
        """
        mr_f = max(self.motion_ratio_f, 0.01)
        mr_r = max(self.motion_ratio_r, 0.01)
        if priority == 'wheel_rate':
            self.spring_rate_installed_f_Nmm = float(self.K_z_f / (1000.0 * (mr_f ** 2)))
            self.spring_rate_installed_r_Nmm = float(self.K_z_r / (1000.0 * (mr_r ** 2)))
        else:
            self.K_z_f = float(self.spring_rate_installed_f_Nmm * 1000.0 * (mr_f ** 2))
            self.K_z_r = float(self.spring_rate_installed_r_Nmm * 1000.0 * (mr_r ** 2))

    def calc_damper_strokes_from_preload(self, corner='f') -> dict:
        """
        Calculates exact damper and wheel travel subdivision (Static Sag, Bump Stroke, Rebound Stroke, Bumpstop Gap)
        governed by the physical 57mm TTX25 stroke, spring preload, spring rate, and motion ratios.
        Tampone contacts in last 5mm of stroke, with solid pack at 3-4mm remaining.
        """
        mr_d = max(self.motion_ratio_f if corner == 'f' else self.motion_ratio_r, 0.01)
        k_s_Nmm = self.spring_rate_installed_f_Nmm if corner == 'f' else self.spring_rate_installed_r_Nmm
        preload_mm = getattr(self, f'spring_preload_{corner}_mm', 5.0)
        s_tot_mm = getattr(self, 'damper_stroke_total_mm', 57.0)
        bs_contact_tail_mm = getattr(self, 'bumpstop_contact_tail_mm', 5.0)
        bs_pack_tail_mm = getattr(self, 'bumpstop_pack_tail_mm', 3.5)

        # Corner static sprung mass weight
        m_sprung = self.m * (1.0 - 0.12)
        wd = self.wd_f if corner == 'f' else (1.0 - self.wd_f)
        f_z0_N = (m_sprung * self.g * wd) / 2.0  # [N]

        # Force acting on spring at damper level
        f_spring_static_N = f_z0_N / mr_d

        # Preload force when fully extended
        preload_n_attr = getattr(self, f'preload_{corner}_N', None)
        if preload_n_attr is not None and preload_n_attr > 0:
            f_preload_N = float(preload_n_attr)
        else:
            f_preload_N = k_s_Nmm * preload_mm

        # Additional compression from fully extended position (Damper Static Sag)
        if f_spring_static_N > f_preload_N:
            damper_sag_mm = (f_spring_static_N - f_preload_N) / max(k_s_Nmm, 0.1)
        else:
            damper_sag_mm = 0.0  # topped out against rebound stop

        # Available strokes from 1g static ride height
        damper_rebound_mm = damper_sag_mm
        damper_bump_mm = max(s_tot_mm - damper_sag_mm, 0.0)
        # Free damper travel before bumpstop contact (contacts in last 5mm)
        damper_bumpstop_gap_mm = max((s_tot_mm - bs_contact_tail_mm) - damper_sag_mm, 0.0)
        # Travel before solid bottoming (impaccamento solido a 3-4mm residui)
        damper_pack_gap_mm = max((s_tot_mm - bs_pack_tail_mm) - damper_sag_mm, 0.0)

        # Wheel level strokes: z_wheel = z_damper / MR
        s_tot_wheel_mm = s_tot_mm / mr_d
        wheel_sag_mm = damper_sag_mm / mr_d
        wheel_rebound_mm = damper_rebound_mm / mr_d
        wheel_bump_mm = damper_bump_mm / mr_d
        wheel_bumpstop_gap_mm = damper_bumpstop_gap_mm / mr_d
        wheel_pack_gap_mm = damper_pack_gap_mm / mr_d

        return {
            's_tot_damper_mm': float(s_tot_mm),
            's_tot_wheel_mm': float(s_tot_wheel_mm),
            'damper_sag_mm': float(damper_sag_mm),
            'wheel_sag_mm': float(wheel_sag_mm),
            'damper_bump_available_mm': float(damper_bump_mm),
            'damper_rebound_available_mm': float(damper_rebound_mm),
            'wheel_bump_available_mm': float(wheel_bump_mm),
            'wheel_rebound_available_mm': float(wheel_rebound_mm),
            'damper_bumpstop_gap_mm': float(damper_bumpstop_gap_mm),
            'wheel_bumpstop_gap_mm': float(wheel_bumpstop_gap_mm),
            'damper_pack_gap_mm': float(damper_pack_gap_mm),
            'wheel_pack_gap_mm': float(wheel_pack_gap_mm)
        }

    def update_suspension_limits(self):
        """Update wheel travel limits and bumpstop gaps directly from physical shock stroke and sag."""
        st_f = self.calc_damper_strokes_from_preload('f')
        st_r = self.calc_damper_strokes_from_preload('r')
        self.bumpstop_gap_f_m = float(st_f['wheel_bumpstop_gap_mm'] / 1000.0)
        self.bumpstop_gap_r_m = float(st_r['wheel_bumpstop_gap_mm'] / 1000.0)
        self.damper_stroke_bump_max_m = float(st_f['wheel_bump_available_mm'] / 1000.0)
        self.damper_stroke_rebound_max_m = -float(st_f['wheel_rebound_available_mm'] / 1000.0)

    def set_kinematic_map(self, map_name: str, coeffs: list, static_val: float = None):
        """
        Update a kinematic polynomial map by name.
        Called by the Curve Digitizer or GUI after fitting / JSON loading.
        
        map_name: camber_gain_f/r, toe_gain_f/r, rollcenter_f/r, rollcenter_lat_f/r,
                  anti_dive, anti_squat, anti_lift, damper_motion_ratio_f/r,
                  arb_motion_ratio_f/r, ackermann
        coeffs:   [a3, a2, a1, a0]
        static_val: optional nominal static value (or offset)
        """
        if not map_name:
            return
        base_name = map_name.replace('_poly3', '').strip()
        
        mapping = {
            'camber_gain_f': 'camber_gain_f_coeffs',
            'camber_gain_r': 'camber_gain_r_coeffs',
            'toe_gain_f': 'toe_gain_f_coeffs',
            'toe_gain_r': 'toe_gain_r_coeffs',
            'rollcenter_f': 'rollcenter_f_coeffs',
            'rollcenter_r': 'rollcenter_r_coeffs',
            'rollcenter_lat_f': 'rollcenter_lat_f_coeffs',
            'rollcenter_lat_r': 'rollcenter_lat_r_coeffs',
            'anti_dive': 'anti_dive_coeffs',
            'anti_squat': 'anti_squat_coeffs',
            'anti_lift': 'anti_lift_coeffs',
            'damper_motion_ratio_f': 'damper_motion_ratio_f_coeffs',
            'damper_motion_ratio_r': 'damper_motion_ratio_r_coeffs',
            'arb_motion_ratio_f': 'arb_motion_ratio_f_coeffs',
            'arb_motion_ratio_r': 'arb_motion_ratio_r_coeffs',
            'ackermann': 'ackermann_coeffs',
            'caster_trail_f': 'caster_trail_f_coeffs',
            'scrub_radius_f': 'scrub_radius_f_coeffs',
            'camber_steer_f': 'camber_steer_f_coeffs',
            'wheel_steer_f': 'wheel_steer_f_coeffs',
            'steering_ratio_f': 'wheel_steer_f_coeffs',
        }
        
        attr = mapping.get(base_name, f"{base_name}_coeffs")
        if hasattr(self, attr):
            setattr(self, attr, list(coeffs))
        else:
            raise ValueError(f"Unknown kinematic map: '{map_name}'. "
                             f"Valid: camber_gain_f/r, toe_gain_f/r, "
                             f"rollcenter_f/r, anti_dive, anti_squat, anti_lift, ackermann, "
                             f"caster_trail_f, scrub_radius_f, camber_steer_f, wheel_steer_f")
            
        # Update scalar nominal/static values
        sval = float(static_val) if static_val is not None else None
        if sval is None and len(coeffs) == 4 and base_name not in ['camber_gain_f', 'camber_gain_r', 'toe_gain_f', 'toe_gain_r', 'camber_steer_f', 'wheel_steer_f']:
            sval = float(coeffs[3])
            
        if sval is not None:
            if base_name == 'camber_gain_f':
                self.camber_f_static = sval
            elif base_name == 'camber_gain_r':
                self.camber_r_static = sval
            elif base_name == 'toe_gain_f':
                self.toe_f_static = sval
            elif base_name == 'toe_gain_r':
                self.toe_r_static = sval
            elif base_name == 'rollcenter_f':
                self.q1 = sval
            elif base_name == 'rollcenter_r':
                self.q2 = sval
            elif base_name == 'damper_motion_ratio_f':
                self.motion_ratio_f = sval
            elif base_name == 'damper_motion_ratio_r':
                self.motion_ratio_r = sval
            elif base_name == 'arb_motion_ratio_f':
                self.motion_ratio_arb_f = sval
            elif base_name == 'arb_motion_ratio_r':
                self.motion_ratio_arb_r = sval
            elif base_name == 'ackermann':
                self.ackermann_pct = (sval / 100.0) if sval > 1.0 else sval
            elif base_name == 'caster_trail_f':
                self.mechanical_trail_f = sval
            elif base_name == 'scrub_radius_f':
                self.scrub_radius_f = sval
            elif base_name == 'anti_dive':
                self.anti_dive_front = (sval / 100.0) if sval > 1.0 else sval
            elif base_name == 'anti_squat':
                self.anti_squat_rear = (sval / 100.0) if abs(sval) > 1.0 else sval
            elif base_name == 'anti_lift':
                self.anti_lift_rear = (sval / 100.0) if abs(sval) > 1.0 else sval

    def get_camber_curve(self, z_range_mm=(-35.0, 35.0), n_pts=71):
        """Returns (z_mm, cam_f_deg, cam_r_deg) across wheel travel z [mm]."""
        z_mm = np.linspace(z_range_mm[0], z_range_mm[1], n_pts)
        z_m = z_mm / 1000.0
        cam_f = np.array([self.eval_poly3(self.camber_gain_f_coeffs, zm) + self.camber_f_static for zm in z_m])
        cam_r = np.array([self.eval_poly3(self.camber_gain_r_coeffs, zm) + self.camber_r_static for zm in z_m])
        return z_mm, cam_f, cam_r

    def get_toe_curve(self, z_range_mm=(-35.0, 35.0), n_pts=71):
        """Returns (z_mm, toe_f_deg, toe_r_deg) across wheel travel z [mm]."""
        z_mm = np.linspace(z_range_mm[0], z_range_mm[1], n_pts)
        z_m = z_mm / 1000.0
        toe_f = np.array([self.eval_poly3(self.toe_gain_f_coeffs, zm) + self.toe_f_static for zm in z_m])
        toe_r = np.array([self.eval_poly3(self.toe_gain_r_coeffs, zm) + self.toe_r_static for zm in z_m])
        return z_mm, toe_f, toe_r

    def get_rollcenter_curve(self, phi_range_deg=(-2.0, 2.0), n_pts=71, z_range_mm=None):
        """Returns (phi_deg, h_rc_f_mm, h_rc_r_mm) across chassis roll angle phi [deg]."""
        r_range = z_range_mm if z_range_mm is not None else phi_range_deg
        phi_deg = np.linspace(r_range[0], r_range[1], n_pts)
        rc_f_mm = np.array([self.eval_poly3(self.rollcenter_f_coeffs, phi) * 1000.0 for phi in phi_deg])
        rc_r_mm = np.array([self.eval_poly3(self.rollcenter_r_coeffs, phi) * 1000.0 for phi in phi_deg])
        return phi_deg, rc_f_mm, rc_r_mm

    def get_rollcenter_lat_curve(self, phi_range_deg=(-2.0, 2.0), n_pts=71, dz_range_mm=None):
        """Returns (phi_deg, y_rc_f_mm, y_rc_r_mm) across chassis roll angle phi [deg]."""
        r_range = dz_range_mm if dz_range_mm is not None else phi_range_deg
        phi_deg = np.linspace(r_range[0], r_range[1], n_pts)
        y_rc_f = np.array([self.eval_poly3(self.rollcenter_lat_f_coeffs, phi) * 1000.0 for phi in phi_deg])
        y_rc_r = np.array([self.eval_poly3(self.rollcenter_lat_r_coeffs, phi) * 1000.0 for phi in phi_deg])
        return phi_deg, y_rc_f, y_rc_r

    def get_anti_features_curve(self, z_range_mm=(-35.0, 35.0), n_pts=71):
        """Returns (z_mm, anti_dive_pct, anti_squat_pct, anti_lift_pct) across wheel travel z [mm]."""
        z_mm = np.linspace(z_range_mm[0], z_range_mm[1], n_pts)
        z_m = z_mm / 1000.0
        dive = np.array([self.eval_poly3(self.anti_dive_coeffs, zm) for zm in z_m])
        squat = np.array([self.eval_poly3(self.anti_squat_coeffs, zm) for zm in z_m])
        lift = np.array([self.eval_poly3(self.anti_lift_coeffs, zm) for zm in z_m])
        if len(dive) > 0 and np.abs(dive).max() <= 1.5:
            dive = dive * 100.0
        if len(squat) > 0 and np.abs(squat).max() <= 1.5:
            squat = squat * 100.0
        if len(lift) > 0 and np.abs(lift).max() <= 1.5:
            lift = lift * 100.0
        return z_mm, dive, squat, lift

    def get_motion_ratio_curve(self, z_range_mm=(-35.0, 35.0), n_pts=71):
        """Returns (z_mm, mr_damper_f, mr_damper_r, mr_arb_f, mr_arb_r) across wheel travel z [mm]."""
        z_mm = np.linspace(z_range_mm[0], z_range_mm[1], n_pts)
        z_m = z_mm / 1000.0
        mr_df = np.array([self.eval_poly3(self.damper_motion_ratio_f_coeffs, zm) for zm in z_m])
        mr_dr = np.array([self.eval_poly3(self.damper_motion_ratio_r_coeffs, zm) for zm in z_m])
        mr_af = np.array([self.eval_poly3(self.arb_motion_ratio_f_coeffs, zm) for zm in z_m])
        mr_ar = np.array([self.eval_poly3(self.arb_motion_ratio_r_coeffs, zm) for zm in z_m])
        return z_mm, mr_df, mr_dr, mr_af, mr_ar

    def get_ackermann_curve(self, delta_range_deg=(-125.0, 125.0), n_pts=71):
        """Returns (delta_sw_deg, ack_pct) across steering wheel / pinion angle delta [deg]."""
        delta_deg = np.linspace(delta_range_deg[0], delta_range_deg[1], n_pts)
        ack_pct = np.array([self.eval_poly3(self.ackermann_coeffs, d) for d in delta_deg])
        # If coefficients were normalized to [0, 1], scale to percentage for display
        if len(ack_pct) > 0 and np.nanmax(np.abs(ack_pct)) <= 1.5:
            ack_pct = ack_pct * 100.0
        return delta_deg, ack_pct

    def get_steer_wheel_angles(self, delta_range_deg=(-125.0, 125.0), n_pts=71):
        """
        Calculates individual left and right road wheel steer angles (delta_fl, delta_fr)
        and difference (delta_fl - delta_fr) across steering wheel angle delta_sw [deg].
        Uses direct polynomial wheel steer curve if available, or smooth continuous Ackermann geometry.
        """
        delta_sw_deg = np.linspace(delta_range_deg[0], delta_range_deg[1], n_pts)

        # Check if wheel_steer_f_coeffs is populated (direct kinematics curve)
        w_coeffs = getattr(self, 'wheel_steer_f_coeffs', [0.0, 0.0, 0.0, 0.0])
        if any(abs(c) > 1e-6 for c in w_coeffs[:3]):
            # Left wheel FL directly from curve
            fl_deg = np.array([self.eval_poly3(w_coeffs, d) for d in delta_sw_deg])
            # Right wheel FR is bilateral mirror across vehicle centerline: delta_fr(d) = -delta_fl(-d)
            fr_deg = np.array([-self.eval_poly3(w_coeffs, -d) for d in delta_sw_deg])
            return delta_sw_deg, fl_deg, fr_deg, (fl_deg - fr_deg)

        sr = self.steering_ratio if self.steering_ratio > 0.1 else 4.72
        d_road_rad = np.deg2rad(delta_sw_deg / sr)
        tan_d = np.tan(d_road_rad)

        L = self.L
        tf = self.t_f
        ack_fl_rad = np.arctan2(L * tan_d, L - (tf / 2.0) * tan_d)
        ack_fr_rad = np.arctan2(L * tan_d, L + (tf / 2.0) * tan_d)

        # Dynamic Ackermann factor
        ack_pct_vals = np.array([self.eval_poly3(self.ackermann_coeffs, d) for d in delta_sw_deg])
        if len(ack_pct_vals) > 0 and np.nanmax(np.abs(ack_pct_vals)) > 1.5:
            ack_frac = ack_pct_vals / 100.0
        else:
            ack_frac = ack_pct_vals

        delta_fl_rad = d_road_rad + ack_frac * (ack_fl_rad - d_road_rad)
        delta_fr_rad = d_road_rad + ack_frac * (ack_fr_rad - d_road_rad)

        return delta_sw_deg, np.rad2deg(delta_fl_rad), np.rad2deg(delta_fr_rad), np.rad2deg(delta_fl_rad - delta_fr_rad)

    def eval_bumpstop_force(self, z_m: float, corner='f') -> float:
        """
        Evaluate non-linear progressive bump stop / rebound limit force [N] for wheel travel z_m [m].
        +z = bump (compression), -z = rebound (droop). Smooth non-linear progression without rigid steps.
        Tampone entra negli ultimi 5mm di stelo ammortizzatore, con impaccamento solido a 3-4 mm residui.
        """
        mr_d = max(self.motion_ratio_f if corner == 'f' else self.motion_ratio_r, 0.01)
        gap = getattr(self, f'bumpstop_gap_{corner}_m', None)
        if gap is None or gap <= 0.0:
            st = self.calc_damper_strokes_from_preload(corner)
            gap = float(st['wheel_bumpstop_gap_mm'] / 1000.0)

        gap_reb = abs(getattr(self, 'damper_stroke_rebound_max_m', 0.015))
        k_bs = getattr(self, f'bumpstop_k_{corner}', 80000.0)
        # Solid pack travel on wheel: remaining 3.5mm stroke (1.5mm bumpstop compression on damper)
        dz_pack = 0.0015 / mr_d

        f_bs = 0.0
        # Bump stop compression: progressive quadratic force + solid pack stiffness
        if z_m > gap:
            dz = z_m - gap
            f_bs += k_bs * dz + 50.0 * k_bs * (dz**2)
            if dz > dz_pack:
                dz_solid = dz - dz_pack
                f_bs += 2.0e6 * dz_solid + 5.0e7 * (dz_solid**2)

        # Rebound stop extension: progressive quadratic force
        elif z_m < -gap_reb:
            dz_reb = -z_m - gap_reb
            f_bs -= (k_bs * dz_reb + 50.0 * k_bs * (dz_reb**2))

        return float(f_bs)

    def update_static_cg_from_ride_height(self, rh_f_m: float, rh_r_m: float, rh_nom_m: float = 0.050) -> float:
        """
        Update static h_cg based on ground clearance variation from nominal 5cm (0.050 m).
        h_cg = h_cg_nom + delta_rh_cg.
        """
        rh_nom = getattr(self, 'nominal_ride_height_m', rh_nom_m)
        h_cg_nom = getattr(self, 'h_cg_nominal', 0.280)
        w_f = getattr(self, 'wd_f', 0.485)
        delta_rh = (1.0 - w_f) * (rh_f_m - rh_nom) + w_f * (rh_r_m - rh_nom)
        self.h_cg = float(max(0.05, h_cg_nom + delta_rh))
        self.static_ride_height_f_m = float(rh_f_m)
        self.static_ride_height_r_m = float(rh_r_m)
        return self.h_cg

    def eval_wheel_rate_installed(self, corner='f') -> float:
        """Calculate effective wheel rate K_w [N/m] from installed spring rate and motion ratio."""
        k_spr = (self.spring_rate_installed_f_Nmm if corner == 'f' else self.spring_rate_installed_r_Nmm) * 1000.0
        mr = self.motion_ratio_f if corner == 'f' else self.motion_ratio_r
        return float(k_spr * (mr**2))

    def eval_dynamic_ground_clearance(self, z_f_m: float, z_r_m: float) -> tuple:
        """
        Calculate dynamic ground clearances h_f, h_r [mm] and return (h_f_mm, h_r_mm, is_bottoming).
        """
        h_f = (self.static_ride_height_f_m - z_f_m) * 1000.0
        h_r = (self.static_ride_height_r_m - z_r_m) * 1000.0
        is_bottoming = (h_f <= 0.0 or h_r <= 0.0)
        return float(h_f), float(h_r), bool(is_bottoming)

    def to_dict(self) -> dict:
        """Serializes all parameters into a clean dictionary."""
        d = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if isinstance(val, (int, float, str, bool, list)):
                d[f.name] = val
        return d

    @classmethod
    def from_dict(cls, d: dict):
        """Creates a VehicleParams instance from dictionary, filtering unknown keys."""
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def save_json(self, filepath: str):
        """Saves vehicle parameters to a JSON file."""
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, filepath: str):
        """Loads vehicle parameters from a JSON file."""
        import json
        import os
        if not os.path.isfile(filepath):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
            cwd = os.getcwd()
            fname = os.path.basename(filepath)
            candidates = [
                os.path.join(repo_root, "data", "setups", fname),
                os.path.join(repo_root, "setups", fname),
                os.path.join(cwd, "data", "setups", fname),
                os.path.join(cwd, "setups", fname),
                os.path.join(repo_root, filepath),
                os.path.join(repo_root, "data", filepath),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    filepath = c
                    break
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def update_tire_relaxation(self, tp):
        """Updates sigma_x and sigma_y from TireParams if defined in .tir."""
        if tp is not None and hasattr(tp, 'get_relaxation_lengths'):
            self.sigma_x, self.sigma_y = tp.get_relaxation_lengths(self.sigma_x, self.sigma_y)

    def import_digitized_curves(self, folder_path: str) -> dict:
        """
        Scans folder_path for digitized polynomial curve JSON files (*_poly3.json or *.json)
        and automatically updates the corresponding kinematics maps on this VehicleParams instance.
        Returns a dict summarizing the loaded curves.
        """
        import os
        import json
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Directory non trovata: {folder_path}")

        loaded = {}
        for fname in os.listdir(folder_path):
            if fname.endswith('.json'):
                fpath = os.path.join(folder_path, fname)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    cname = data.get('curve_name')
                    if not cname:
                        base = fname.replace('.json', '')
                        if base.endswith('_poly3'):
                            base = base[:-6]
                        cname = base

                    coeffs = data.get('coeffs')
                    sval = data.get('static_val')
                    if coeffs and isinstance(coeffs, list):
                        self.set_kinematic_map(cname, coeffs, static_val=sval)
                        loaded[cname] = coeffs
                    elif sval is not None:
                        self.set_kinematic_map(cname, [0.0, 0.0, 0.0, float(sval)], static_val=sval)
                        loaded[cname] = float(sval)

                except Exception as err:
                    print(f"Errore lettura {fname}: {err}")

        return loaded

    @classmethod
    def nominal_fsae_params(cls) -> 'VehicleParams':
        """Factory method returning nominal parameters for FSAE prototype."""
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
        setups_dir = os.path.join(repo_root, 'data', 'setups')
        if os.path.isdir(setups_dir):
            # Check for any private/team setup first (e.g. starting with 'et')
            for f in sorted(os.listdir(setups_dir), reverse=True):
                if f.startswith('et') and f.endswith('.json'):
                    try:
                        vp = cls.load_json(os.path.join(setups_dir, f))
                        vp.update_suspension_limits()
                        return vp
                    except Exception:
                        pass
        # Fallback to public mockup
        for fname in ['fsae_mockup.json', 'default_vehicle.json']:
            setup_path = os.path.join(setups_dir, fname)
            if os.path.exists(setup_path):
                try:
                    vp = cls.load_json(setup_path)
                    vp.update_suspension_limits()
                    return vp
                except Exception:
                    pass
        return cls()

    @classmethod
    def segmento_b_suv(cls) -> 'VehicleParams':
        """
        Factory method returning nominal parameters for a modern B-Segment Road Car / Compact SUV:
        - Front-Wheel Drive (FWD) with open mechanical differential
        - Total mass: 1250 kg (with driver and fluids), 60% front weight distribution
        - Wheelbase: 2.56 m, Tracks: 1.54 m front and rear, CoG height: 0.56 m
        - Road tire dimensions: 215/60 R17 (Rw = 0.315 m), LMUX = 0.65, LMUY = 0.65
        - Soft compliant suspension (ride rates 28 N/mm front, 26 N/mm rear)
        - Passenger car steering rack (ratio 15.5:1, max handwheel +/- 500 deg)
        """
        return cls(
            vehicle_type="b_segment_suv",
            drivetrain_type="fwd",
            diff_type="open",
            # Mass & Inertia
            m=1250.0,
            Iz=1850.0,
            Ix=520.0,
            Iy=1700.0,
            Iw=11.5,
            Iw_f=11.5,
            Iw_r=3.5,
            Ixz=0.0,
            g=9.81,
            # Geometry
            L=2.56,
            t_f=1.54,
            t_r=1.54,
            h_cg=0.56,
            R_w=0.315,
            # MDO & Distribution
            wd_f=0.60,
            lltd=0.58,
            aero_balance=0.50,
            # Aerodynamics
            rho=1.225,
            S=2.25,
            Cz=0.02,
            Cd=0.34,
            # Tires & Uncertainty (Dedicated Road Touring tire: 215/60 R17, FNOMIN=3500N)
            tire_file="Road_215_60R17_Touring.tir",
            LMUX=1.0,
            LMUY=1.0,
            tire_pressure_f_bar=2.3,
            tire_pressure_r_bar=2.2,
            sigma_x=0.28,
            sigma_y=0.35,
            # Powertrain & Braking
            brake_bias=0.65,
            t_split_rear=0.50,
            P_max_kw=100.0,
            T_max_motor=230.0,
            final_drive=4.1,
            n_drivetrain=0.94,
            T_brake_max_total=4800.0,
            tire_thermal_mass=16000.0,
            tire_h_conv=32.0,
            # Static Kinematics
            camber_f_static=-0.5,
            camber_r_static=-0.8,
            toe_f_static=0.0,
            toe_r_static=0.15,
            caster_f=4.0,
            kpi_f=11.0,
            scrub_radius_f=0.010,
            mechanical_trail_f=0.028,
            spindle_offset_f=0.04,
            spindle_offset_r=0.0,
            # Steering system (Road car rack: 15.5:1, +/-500 deg handwheel)
            steering_ratio=15.5,
            steering_wheel_max_deg=500.0,
            # Roll Centers
            q1=0.080,
            q2=0.110,
            # Unsprung Masses
            m_us_f=42.0,
            m_us_r=38.0,
            # Roll Stiffness & Damping
            K_phi_f=45000.0,
            K_phi_r=32000.0,
            C_phi_f=3200.0,
            C_phi_r=2500.0,
            # Ride Stiffness & Damping
            K_z_f=28000.0,
            K_z_r=26000.0,
            C_z_f=2800.0,
            C_z_r=2400.0,
            # Damper Stroke & Bumpstops
            damper_stroke_total_mm=180.0,
            damper_stroke_bump_max_m=0.100,
            damper_stroke_rebound_max_m=-0.080,
            bumpstop_gap_f_m=0.055,
            bumpstop_gap_r_m=0.055,
            bumpstop_length_f_mm=45.0,
            bumpstop_length_r_mm=45.0,
            bumpstop_k_f=40000.0,
            bumpstop_k_r=40000.0,
            spring_rate_installed_f_Nmm=30.0,
            spring_rate_installed_r_Nmm=28.0,
            spring_preload_f_mm=15.0,
            spring_preload_r_mm=15.0,
            motion_ratio_f=0.96,
            motion_ratio_r=0.96,
            motion_ratio_arb_f=0.65,
            motion_ratio_arb_r=0.55,
            static_ride_height_f_m=0.170,
            static_ride_height_r_m=0.180,
            # Kinematic maps (road car MacPherson / semi-trailing)
            camber_gain_f_coeffs=[0.0, 0.0, -8.0, 0.0],
            camber_gain_r_coeffs=[0.0, 0.0, -6.0, 0.0],
            toe_gain_f_coeffs=[0.0, 0.0, 0.2, 0.0],
            toe_gain_r_coeffs=[0.0, 0.0, -0.3, 0.0],
            rollcenter_f_coeffs=[0.0, -0.0002, 0.0, 0.080],
            rollcenter_r_coeffs=[0.0, -0.0002, 0.0, 0.110],
            rollcenter_lat_f_coeffs=[0.0, 0.0, 0.030, 0.0],
            rollcenter_lat_r_coeffs=[0.0, 0.0, 0.020, 0.0],
            anti_dive_coeffs=[0.0, 0.0, 0.0, 0.20],
            anti_squat_coeffs=[0.0, 0.0, 0.0, 0.15],
            anti_lift_coeffs=[0.0, 0.0, 0.0, 0.10],
            damper_motion_ratio_f_coeffs=[0.0, 0.0, 0.0, 0.96],
            damper_motion_ratio_r_coeffs=[0.0, 0.0, 0.0, 0.96],
            arb_motion_ratio_f_coeffs=[0.0, 0.0, 0.0, 0.65],
            arb_motion_ratio_r_coeffs=[0.0, 0.0, 0.0, 0.55],
            ackermann_coeffs=[0.0, 0.0, 0.0, 70.0],
            ackermann_pct=0.70,
            caster_trail_f_coeffs=[0.0, 0.0, 0.0, 0.028],
            scrub_radius_f_coeffs=[0.0, 0.0, 0.0, 0.010],
            camber_steer_f_coeffs=[0.0, 0.0, -0.003, 0.0],
            wheel_steer_f_coeffs=[0.0, 0.0, 1.0 / 15.5, 0.0],
            # Powertrain & Battery (ICE / Hybrid / BEV road parameters)
            battery_capacity_kwh=40.0,
            battery_voltage_nom=400.0,
            regen_max_kw=30.0,
            regen_efficiency=0.70,
            motor_base_eff=0.90,
            inverter_eff=0.95
        )



# -----------------------------------------------------------------------
# Per-model parameter catalogue — used by the GUI to build the dynamic panel
# -----------------------------------------------------------------------

MODEL_PARAM_CATALOGUE = {
    "3-DOF": {
        "Geometry (Design Variables)": [
            ("Wheelbase L", "L", "m", 1.529),
            ("Front Track t_f", "t_f", "m", 1.252),
            ("Rear Track t_r", "t_r", "m", 1.136),
        ],
        "Mass Distribution (Design Variables)": [
            ("Weight Dist. Front wd_f", "wd_f", "-", 0.485),
        ],
        "Load Transfer (Design Variables)": [
            ("Lat. Load Transfer Dist. lltd", "lltd", "-", 0.525),
        ],
        "Roll Centers (Design Variables)": [
            ("Front Roll Center Height q1", "q1", "m", 0.0463),
            ("Rear Roll Center Height q2", "q2", "m", 0.07693),
        ],
        "Aerodynamics (Design Variables)": [
            ("Aero Balance Front aero_bal", "aero_balance", "-", 0.509),
        ],
        "Powertrain & Transmission (Design Variables)": [
            ("Final Drive Ratio (Rapporto Riduzione)", "final_drive", "-", 4.0),
            ("Brake Bias Front", "brake_bias", "-", 0.60),
            ("Rear Torque Split", "t_split_rear", "-", 0.50),
        ],
        "Static Kinematics (Design Variables)": [
            ("Static Camber Front", "camber_f_static", "deg", -1.5),
            ("Static Camber Rear", "camber_r_static", "deg", -1.0),
            ("Static Toe Front", "toe_f_static", "deg", 0.0),
            ("Static Toe Rear", "toe_r_static", "deg", 0.0),
        ],
        "Tire Uncertainty (Fixed)": [
            ("Lat. Friction Scale LMUY", "LMUY", "-", 1.0),
            ("Long. Friction Scale LMUX", "LMUX", "-", 1.0),
        ],
    },
    "7-DOF": {
        "Geometry (Design Variables)": [
            ("Wheelbase L", "L", "m", 1.529),
            ("Front Track t_f", "t_f", "m", 1.252),
            ("Rear Track t_r", "t_r", "m", 1.136),
        ],
        "Mass Distribution (Design Variables)": [
            ("Weight Dist. Front wd_f", "wd_f", "-", 0.485),
        ],
        "Load Transfer (Design Variables)": [
            ("Lat. Load Transfer Dist. lltd", "lltd", "-", 0.525),
        ],
        "Roll Centers (Design Variables)": [
            ("Front Roll Center Height q1", "q1", "m", 0.0463),
            ("Rear Roll Center Height q2", "q2", "m", 0.07693),
        ],
        "Aerodynamics (Design Variables)": [
            ("Aero Balance Front aero_bal", "aero_balance", "-", 0.509),
            ("Drag Coefficient Cd", "Cd", "-", 1.2),
        ],
        "Powertrain & Transmission (Design Variables)": [
            ("Final Drive Ratio (Rapporto Riduzione)", "final_drive", "-", 4.0),
            ("Brake Bias Front", "brake_bias", "-", 0.60),
            ("Rear Torque Split", "t_split_rear", "-", 0.50),
            ("Max Power P_max", "P_max_kw", "kW", 80.0),
            ("Max Motor Torque T_max", "T_max_motor", "Nm", 100.0),
            ("Drivetrain Efficiency", "n_drivetrain", "-", 0.95),
        ],
        "Static Kinematics (Design Variables)": [
            ("Static Camber Front", "camber_f_static", "deg", -1.5),
            ("Static Camber Rear", "camber_r_static", "deg", -1.0),
            ("Static Toe Front", "toe_f_static", "deg", 0.0),
            ("Static Toe Rear", "toe_r_static", "deg", 0.0),
            ("Front Caster Angle", "caster_f", "deg", 3.0),
            ("Front KPI Angle", "kpi_f", "deg", 8.2),
            ("Front Scrub Radius", "scrub_radius_f", "m", 0.0368),
        ],
        "Tire Uncertainty (Fixed)": [
            ("Lat. Friction Scale LMUY", "LMUY", "-", 1.0),
            ("Long. Friction Scale LMUX", "LMUX", "-", 1.0),
        ],
    },
    "14-DOF": {
        "Geometry (Design Variables)": [
            ("Wheelbase L", "L", "m", 1.529),
            ("Front Track t_f", "t_f", "m", 1.252),
            ("Rear Track t_r", "t_r", "m", 1.136),
        ],
        "Mass Distribution (Design Variables)": [
            ("Weight Dist. Front wd_f", "wd_f", "-", 0.485),
        ],
        "Roll Centers (Design Variables)": [
            ("Front Roll Center Height q1", "q1", "m", 0.0463),
            ("Rear Roll Center Height q2", "q2", "m", 0.07693),
        ],
        "Aerodynamics (Design Variables)": [
            ("Aero Balance Front aero_bal", "aero_balance", "-", 0.509),
        ],
        "Powertrain & Transmission (Design Variables)": [
            ("Final Drive Ratio (Rapporto Riduzione)", "final_drive", "-", 4.0),
            ("Max Power P_max", "P_max_kw", "kW", 80.0),
            ("Max Motor Torque T_max", "T_max_motor", "Nm", 100.0),
            ("Drivetrain Efficiency", "n_drivetrain", "-", 0.95),
        ],
        "Static Kinematics (Design Variables)": [
            ("Front Static Camber", "camber_f_static", "deg", -1.5),
            ("Rear Static Camber", "camber_r_static", "deg", -1.0),
            ("Front Static Toe", "toe_f_static", "deg", 0.0),
            ("Rear Static Toe", "toe_r_static", "deg", 0.0),
        ],
        "Suspension — Roll (Design Variables)": [
            ("Front Roll Stiffness K_phi_f", "K_phi_f", "Nm/rad", 34377.0),
            ("Rear Roll Stiffness K_phi_r", "K_phi_r", "Nm/rad", 34377.0),
        ],
        "Suspension — Ride (Design Variables)": [
            ("Front Wheel Rate K_z_f", "K_z_f", "N/m", 37600.0),
            ("Rear Wheel Rate K_z_r", "K_z_r", "N/m", 38500.0),
        ],
        "Real Damper & Spring Setup (Fixed / Measured)": [
            ("Max Damper Bump Stroke", "damper_stroke_bump_max_m", "m", 0.040),
            ("Max Damper Rebound Stroke", "damper_stroke_rebound_max_m", "m", -0.030),
            ("Bumpstop Free Gap Front", "bumpstop_gap_f_m", "m", 0.025),
            ("Bumpstop Free Gap Rear", "bumpstop_gap_r_m", "m", 0.025),
            ("Bumpstop Stiffness Front", "bumpstop_k_f", "N/m", 80000.0),
            ("Bumpstop Stiffness Rear", "bumpstop_k_r", "N/m", 80000.0),
            ("Installed Spring Front", "spring_rate_installed_f_Nmm", "N/mm", 59.5),
            ("Installed Spring Rear", "spring_rate_installed_r_Nmm", "N/mm", 61.0),
            ("Motion Ratio Front MR_f", "motion_ratio_f", "-", 0.794),
            ("Motion Ratio Rear MR_r", "motion_ratio_r", "-", 0.800),
            ("Static Ride Height Front", "static_ride_height_f_m", "m", 0.050),
            ("Static Ride Height Rear", "static_ride_height_r_m", "m", 0.050),
        ],
        "Kinematic Maps (from Curve Digitizer — Fixed)": [
            ("Camber Gain Front [poly3]", "camber_gain_f_coeffs", "deg/m", None),
            ("Camber Gain Rear [poly3]", "camber_gain_r_coeffs", "deg/m", None),
            ("Bump Steer Front [poly3]", "toe_gain_f_coeffs", "deg/m", None),
            ("Bump Steer Rear [poly3]", "toe_gain_r_coeffs", "deg/m", None),
            ("Roll Center Front [poly3]", "rollcenter_f_coeffs", "m/m", None),
            ("Roll Center Rear [poly3]", "rollcenter_r_coeffs", "m/m", None),
            ("Anti-Dive [poly3]", "anti_dive_coeffs", "-", None),
            ("Anti-Squat [poly3]", "anti_squat_coeffs", "-", None),
            ("Ackermann Correction [poly3]", "ackermann_coeffs", "deg/deg", None),
            ("Caster Trail Front [poly3]", "caster_trail_f_coeffs", "m", None),
            ("Scrub Radius Front [poly3]", "scrub_radius_f_coeffs", "m", None),
            ("Camber Steer Delta [poly3]", "camber_steer_f_coeffs", "deg", None),
            ("Wheel Steer FL vs Pinion [poly3]", "wheel_steer_f_coeffs", "deg/deg", None),
        ],
        "Tire Uncertainty (Fixed)": [
            ("Lat. Friction Scale LMUY", "LMUY", "-", 1.0),
            ("Long. Friction Scale LMUX", "LMUX", "-", 1.0),
        ],
    },
    "ALL": {
        "Massa & Inerzie (Tutti)": [
            ("Massa Totale m", "m", "kg", 292.0),
            ("Inerzia Imbardata Iz", "Iz", "kg·m²", 120.0),
            ("Inerzia Ruota Iw", "Iw", "kg·m²", 1.2),
            ("Altezza Baricentro h_cg", "h_cg", "m", 0.28),
            ("Raggio Efficace Gomma R_w", "R_w", "m", 0.228),
        ],
        "Geometria & Telaio (Tutti)": [
            ("Passo L", "L", "m", 1.529),
            ("Carreggiata Ant. t_f", "t_f", "m", 1.252),
            ("Carreggiata Post. t_r", "t_r", "m", 1.136),
        ],
        "Distribuzione Masse & Carichi (MDO)": [
            ("Ripartizione Masse Ant. wd_f", "wd_f", "-", 0.485),
            ("Ripartizione Trasferimento LLTD", "lltd", "-", 0.525),
            ("Bilanciamento Aero Ant.", "aero_balance", "-", 0.509),
        ],
        "Aerodinamica (Tutti)": [
            ("Densità Aria rho", "rho", "kg/m³", 1.225),
            ("Area Frontale S", "S", "m²", 1.139),
            ("Coefficiente Cz (Downforce)", "Cz", "-", 3.421),
            ("Coefficiente Cd (Drag)", "Cd", "-", 1.200),
            ("Resistenza al Rotolamento Crr (Override QSY1)", "crr", "-", 0.015),
        ],
        "Propulsione, Frenata & Trasmissione (MDO)": [
            ("Tipo Veicolo (formula / b_segment_suv)", "vehicle_type", "-", "formula"),
            ("Tipo Trazione (rwd / fwd / awd)", "drivetrain_type", "-", "rwd"),
            ("Tipo Differenziale (open / lsd)", "diff_type", "-", "open"),
            ("Rapporto Trasmissione Finale (Final Drive)", "final_drive", "-", 4.0),
            ("Potenza Massima P_max", "P_max_kw", "kW", 80.0),
            ("Coppia Max Motore", "T_max_motor", "Nm", 100.0),
            ("Brake Bias Anteriore", "brake_bias", "-", 0.60),
            ("Ripartizione Coppia Post.", "t_split_rear", "-", 0.50),
            ("Capacità Batteria", "battery_capacity_kwh", "kWh", 6.8),
            ("Potenza Max Regen", "regen_max_kw", "kW", 40.0),
        ],
        "Cinematica Statica & Jacking": [
            ("Camber Statico Ant.", "camber_f_static", "deg", -1.5),
            ("Camber Statico Post.", "camber_r_static", "deg", -1.0),
            ("Toe Statico Ant.", "toe_f_static", "deg", 0.0),
            ("Toe Statico Post.", "toe_r_static", "deg", 0.0),
            ("Angolo Caster Ant.", "caster_f", "deg", 3.0),
            ("Angolo KPI Ant.", "kpi_f", "deg", 8.2),
            ("Raggio di Scrub Ant.", "scrub_radius_f", "m", 0.0368),
            ("Braccio a Terra Meccanico", "mechanical_trail_f", "m", 0.012),
            ("Centro di Rollio Ant. q1", "q1", "m", 0.0463),
            ("Centro di Rollio Post. q2", "q2", "m", 0.07693),
        ],
        "Rigidezze & Smorzamenti (14-DOF)": [
            ("Rigidezza Rollio Ant. K_phi_f", "K_phi_f", "Nm/rad", 34377.0),
            ("Rigidezza Rollio Post. K_phi_r", "K_phi_r", "Nm/rad", 34377.0),
            ("Smorzamento Rollio Ant. C_phi_f", "C_phi_f", "Nm·s/rad", 800.0),
            ("Smorzamento Rollio Post. C_phi_r", "C_phi_r", "Nm·s/rad", 600.0),
            ("Wheel Rate Ant. K_z_f", "K_z_f", "N/m", 37600.0),
            ("Wheel Rate Post. K_z_r", "K_z_r", "N/m", 38500.0),
            ("Smorzamento Corsa Ant. C_z_f", "C_z_f", "N·s/m", 1500.0),
            ("Smorzamento Corsa Post. C_z_r", "C_z_r", "N·s/m", 1200.0),
            ("Massa Non Sospesa Ant. m_us_f", "m_us_f", "kg", 18.0),
            ("Massa Non Sospesa Post. m_us_r", "m_us_r", "kg", 16.0),
        ],
        "Ammortizzatori Reali, Tamponi & Molle Installate": [
            ("Corsa Max Bump Ammortizzatore", "damper_stroke_bump_max_m", "m", 0.040),
            ("Corsa Max Rebound Ammortizzatore", "damper_stroke_rebound_max_m", "m", -0.030),
            ("Traferro Libero Tampone Ant.", "bumpstop_gap_f_m", "m", 0.025),
            ("Traferro Libero Tampone Post.", "bumpstop_gap_r_m", "m", 0.025),
            ("Rigidezza Tampone Ant.", "bumpstop_k_f", "N/m", 80000.0),
            ("Rigidezza Tampone Post.", "bumpstop_k_r", "N/m", 80000.0),
            ("Rigidezza Molla Installata Ant.", "spring_rate_installed_f_Nmm", "N/mm", 59.5),
            ("Rigidezza Molla Installata Post.", "spring_rate_installed_r_Nmm", "N/mm", 61.0),
            ("Motion Ratio Ant. MR_f", "motion_ratio_f", "-", 0.794),
            ("Motion Ratio Post. MR_r", "motion_ratio_r", "-", 0.800),
            ("Altezza da Terra Statica Ant.", "static_ride_height_f_m", "m", 0.050),
            ("Altezza da Terra Statica Post.", "static_ride_height_r_m", "m", 0.050),
        ],
        "Mappe Cinematiche Polinomiali (Curve Digitizer)": [
            ("Camber Gain Ant. [poly3]", "camber_gain_f_coeffs", "deg/m", None),
            ("Camber Gain Post. [poly3]", "camber_gain_r_coeffs", "deg/m", None),
            ("Bump Steer Ant. [poly3]", "toe_gain_f_coeffs", "deg/m", None),
            ("Bump Steer Post. [poly3]", "toe_gain_r_coeffs", "deg/m", None),
            ("Centro di Rollio Ant. [poly3]", "rollcenter_f_coeffs", "m/m", None),
            ("Centro di Rollio Post. [poly3]", "rollcenter_r_coeffs", "m/m", None),
            ("Anti-Dive [poly3]", "anti_dive_coeffs", "-", None),
            ("Anti-Squat [poly3]", "anti_squat_coeffs", "-", None),
            ("Correzione Ackermann [poly3]", "ackermann_coeffs", "deg/deg", None),
            ("Braccio a Terra Meccanico [poly3]", "caster_trail_f_coeffs", "m", None),
            ("Raggio di Scrub Ant. [poly3]", "scrub_radius_f_coeffs", "m", None),
            ("Delta Camber Steer Ant. [poly3]", "camber_steer_f_coeffs", "deg", None),
            ("Angolo Ruota vs Pignone Ant. [poly3]", "wheel_steer_f_coeffs", "deg/deg", None),
        ],
        "Incertezze Gomme (Scaling)": [
            ("Scala Aderenza Laterale LMUY", "LMUY", "-", 1.0),
            ("Scala Aderenza Longitudinale LMUX", "LMUX", "-", 1.0),
        ],
        "Pneumatici — Rilassamento Dinamico": [
            ("Lunghezza Rilassamento Long. sigma_x", "sigma_x", "m", 0.15),
            ("Lunghezza Rilassamento Lat. sigma_y", "sigma_y", "m", 0.22),
        ],
    }
}

MODEL_GUIDE_TEXT = {
    "3-DOF": """## 3-DOF Steady-State Model

**Gradi di Libertà**: Sideslip β, Yaw Rate r (β e r vengono risolti dal Newton Solver).

**Cosa ottimizza**: L'ottimizzatore SLSQP varia la geometria (L, t_f, t_r), la distribuzione dei pesi, il bilanciamento LLTD e del carico aerodinamico per **massimizzare l'accelerazione laterale limite** a velocità fissa.

**Parametri fissi durante MDO**: LMUX, LMUY (grip nominale dei pneumatici). Possono essere variati manualmente per analisi di sensitivity.

**Non modella**: Dinamica trasiente, rollio, beccheggio, camber cinematico, forze longitudinali.

**Ideale per**: Setup di bilanciamento di base, ottimizzazione rapida dei parametri chiave.
""",
    "7-DOF": """## 7-DOF Transient Trim Model

**Gradi di Libertà**: vx, vy, r (dinamica laterale) + ω_fl, ω_fr, ω_rl, ω_rr (spin ruote).

**Metodo**: Trim stazionario — l'ottimizzatore cerca il punto con ẋ=0 (equilibrio di forze e momenti) a sterzo e velocità fissati. I camber e toe statici modificano direttamente gli angoli di deriva effettivi delle ruote.

**Cosa ottimizza rispetto al 3-DOF**: In più include **camber statico** e **toe statico** come variabili di progetto. Queste modificano l'angolo effettivo di deriva (α_eff = α_geom + camber_contribution + toe_static) e quindi l'efficienza laterale di ogni ruota.

**LMUX/LMUY**: Parametri di incertezza — variarli simula pneumatici più o meno gommosi (es. T_amb bassa, usura, pressione).

**Non modella**: Rollio fisico, beccheggio, cinematica sospensione variabile.
""",
    "14-DOF": """## 14-DOF Full Vehicle Trim Model

**Gradi di Libertà**: vx, vy, r + ω×4 (ruote) + φ (roll), ż_heave, θ (pitch) + z_fl,fr,rl,rr (wheel travel) = 14 DOF.

**Mappe Cinematiche (Curve Digitizer)**:
Le funzioni che legano la corsa ruota agli angoli caratteristici sono **polinomi di grado 3** i cui coefficienti vengono caricati tramite il Curve Digitizer. Le mappe disponibili sono:
- **Camber vs Wheel Travel** (f/r): γ(z)
- **Bump Steer vs Wheel Travel** (f/r): δ_toe(z)
- **Roll Center Height vs Wheel Travel** (f/r): h_rc(z)
- **Anti-Dive / Anti-Squat**: frazione di forza anti-trasferimento
- **Ackermann Correction vs Steering**: correzione δ_ack(δ)

Per caricare una mappa: usa il **Curve Digitizer** (menu Tools), fai il fit, poi clicca **"Export to 14-DOF"** e seleziona il nome della curva.

**Variabili di Progetto Aggiuntive**: K_phi_f/r (rigidezza rollistica) e K_z_f/r (wheel rate) permettono di ottimizzare il setup sospensivo direttamente nel loop MDO.

**Roll Gradient KPI**: attivo solo con questo modello.
""",
    "ALL": """## ALL — Configurazione Globale Completa (Tutti i Parametri)

In questa modalità tutti i parametri del veicolo (**Massa, Inerzie, Geometria, Aerodinamica, Powertrain, Cinematica, Sospensioni, Ammortizzatori Reali e Incertezze**) sono visibili e modificabili liberamente.

**Funzionalità Profilo Setup**:
- **Salva su File Originale**: Salva le modifiche direttamente sul file di setup `.json` attivo.
- **Salva su Nuovo File**: Esporta la configurazione corrente in un nuovo file `.json` nominato dall'utente.
- **Selettore Setup Attivo**: Permette di passare istantaneamente tra setup salvati (es. Skidpad, AutoX, Endurance, Rain).
"""
}

nominal_fsae_params = VehicleParams.nominal_fsae_params
