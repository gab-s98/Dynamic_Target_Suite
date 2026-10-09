import sys

try:
    import openmdao.api as om
    print("OpenMDAO imported successfully.")
except ImportError as e:
    print(f"Error importing openmdao: {e}")
    sys.exit(1)

try:
    import casadi as ca
    print("CasADi imported successfully.")
    
    # Instantiate a symbolic variable
    x = ca.SX.sym('x')
    print(f"CasADi symbolic variable instantiated: {x}")
except ImportError as e:
    print(f"Error importing casadi: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error instantiating CasADi symbolic variable: {e}")
    sys.exit(1)

print("Environment validation successful!")
