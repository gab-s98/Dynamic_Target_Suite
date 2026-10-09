import os


class TireParams:
    """
    Parser for Pacejka .tir property files.
    
    Reads ALL sections (not only [PARAMETERS]) so that scaling coefficients
    like LMUY, LMUX (typically in [SCALING_COEFFICIENTS]) are correctly
    parsed and available via .get().
    
    Any key found across all sections is stored in self.params.
    Duplicate keys (same name in multiple sections) are overwritten by the
    last occurrence — consistent with how TNO/MSC ADAMS resolves them.
    """
    def __init__(self, tir_file_path=None):
        self.params = {}
        self.filepath = self._resolve_path(tir_file_path)
        if self.filepath:
            self.load_from_tir(self.filepath)

    @staticmethod
    def _resolve_path(path):
        """Robustly resolves .tir file path, falling back to nominal Hoosier tire if missing/empty."""
        if path and isinstance(path, str) and os.path.isfile(path):
            return os.path.abspath(path)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
        cwd = os.getcwd()
        
        # Build list of potential directory candidates
        cand_dirs = [
            os.path.join(repo_root, "data", "tire_tir"),
            os.path.join(repo_root, "tire_tir"),
            os.path.join(cwd, "data", "tire_tir"),
            os.path.join(cwd, "tire_tir"),
            os.path.join(base_dir, "tire_tir"),
            os.path.join(base_dir, "..", "..", "tire_tir"),
            repo_root,
            cwd
        ]
        
        # 1. If path was passed, try to match it directly or by filename
        if path and isinstance(path, str) and path.strip():
            raw_path = path.strip()
            fname = os.path.basename(raw_path)
            for d in cand_dirs:
                # Try matching by filename
                c1 = os.path.join(d, fname)
                if os.path.isfile(c1):
                    return os.path.abspath(c1)
                # Try matching relative to dir
                c2 = os.path.join(d, raw_path)
                if os.path.isfile(c2):
                    return os.path.abspath(c2)
                    
        # 2. Fallback to default tire file
        default_names = [
            "Simple_Load_Dependent.tir",
            "Hoosier_43075_16x7.5-10_R20_rim7_lm_2_3.tir",
            "Hoosier_43075_16x7.5-10_R20_Rim7_lm.tir",
            "SC24_Hoosier_16x7.5_10_R20_rim7_fixed_2_3.tir"
        ]
        for d in cand_dirs:
            for dname in default_names:
                c = os.path.join(d, dname)
                if os.path.isfile(c):
                    return os.path.abspath(c)
                    
        # 3. Check any .tir file in any candidate directory
        for d in cand_dirs:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith('.tir'):
                        c = os.path.join(d, f)
                        if os.path.isfile(c):
                            return os.path.abspath(c)
        return None

    def load_from_tir(self, filepath):
        """
        Parses the entire .tir file, capturing key=value pairs from
        ALL sections (not restricted to [PARAMETERS]).
        Comment characters: $ and !
        """
        if not filepath or not os.path.isfile(filepath):
            return
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('$') or line.startswith('!'):
                        continue
                    # Skip section headers
                    if line.startswith('['):
                        continue
                    if '=' in line:
                        key, val = line.split('=', 1)
                        key = key.strip()
                        # Strip inline comments
                        val = val.split('$')[0].split('!')[0].strip()
                        try:
                            if '/' in val:
                                parts = val.split('/')
                                if len(parts) == 2:
                                    self.params[key] = float(parts[0]) / float(parts[1])
                                else:
                                    self.params[key] = float(val)
                            else:
                                self.params[key] = float(val)
                        except (ValueError, ZeroDivisionError):
                            self.params[key] = val.strip("'\"")
        except (FileNotFoundError, PermissionError, OSError):
            # Allow graceful handling without crashing
            pass

    def get(self, key, default=0.0):
        """Safely retrieve a parameter with a fallback default."""
        return self.params.get(key, default)

    def get_relaxation_lengths(self, default_x: float = 0.15, default_y: float = 0.22) -> tuple:
        """
        Retrieves tire contact patch relaxation lengths (sigma_x, sigma_y) in meters.
        Reads explicit SIGMA_X and SIGMA_Y if present in .tir params, otherwise returns defaults.
        """
        sx = float(self.params.get('SIGMA_X', default_x))
        sy = float(self.params.get('SIGMA_Y', default_y))
        return sx, sy

    @property
    def sigma_x(self) -> float:
        """Tire longitudinal relaxation length [m]."""
        return self.get_relaxation_lengths()[0]

    @property
    def sigma_y(self) -> float:
        """Tire lateral relaxation length [m]."""
        return self.get_relaxation_lengths()[1]

    def get_tyreside(self, default: str = 'RIGHT') -> str:
        """
        Returns the calibrated tire mounting side ('RIGHT' or 'LEFT').
        Defaults to 'RIGHT' if not specified or unparseable.
        """
        raw = self.params.get('TYRESIDE', default)
        if isinstance(raw, str):
            clean = raw.replace("'", "").replace('"', '').strip().upper()
            if clean in ('RIGHT', 'LEFT'):
                return clean
        return default
