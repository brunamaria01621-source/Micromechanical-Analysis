import subprocess
import os
import numpy as np

def run_fast(fast_exe_path, dat_file_name_base):
    """
    Runs the FAST program, reads the output, and extracts the matrix [CM] and the stress matrix [SIGMA].
    
    Args:
        fast_exe_path (str): Full path to the FAST executable.
        dat_file_name_base (str): Input file name .dat, without the extension.
        
    Returns:
        tuple: (np.ndarray, np.ndarray) The extracted [CM] matrix and [SIGMA] stress matrix.
        
    Raises:
        RuntimeError: If there are problems running or reading the file.
    """
    working_dir = os.path.dirname(fast_exe_path)
    print(f"Running FAST on: {fast_exe_path}")
    print(f"With input file: {dat_file_name_base}")
    print(f"In the working directory: {working_dir}")

    try:
        # Starts process, redirecting stdin (Standart Input) and stdout (Standart Output)
        proc = subprocess.Popen(
            [fast_exe_path],
            cwd=working_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Send the file name (without .dat) + enter
        stdout, stderr = proc.communicate(dat_file_name_base + "\n", timeout=120)

        print("\n=========== FAST OUTPUT ===========\n")
        print(stdout)
        print("\n=========== END OF OUTPUT ============\n")

        # Split by lines
        lines = stdout.splitlines()

        # Search for line containing [CM]
        try:
            idx_cm = next(i for i, line in enumerate(lines) if line.strip().startswith("[CM]"))
        except StopIteration:
            # Try searching in .out file if standard output is incomplete
            out_file = os.path.join(working_dir, dat_file_name_base + ".out")
            if os.path.exists(out_file):
                with open(out_file, 'r') as f:
                    out_lines = f.readlines()
                try:
                    idx_cm = next(i for i, line in enumerate(out_lines) if line.strip().startswith("[CM]"))
                    lines = out_lines
                except StopIteration:
                    raise RuntimeError("Marker '[CM]' not found in either FAST output or .out file.")
            else:
                raise RuntimeError("Marker '[CM]' not found in either FAST output.")

        # Extracts the rows from the matrix [CM]
        cm_lines = lines[idx_cm+1:idx_cm+4]
        cm_values = []
        for line in cm_lines:
            cm_values.extend([float(v) for v in line.strip().split()])
        if len(cm_values) != 9:
            raise RuntimeError("Incomplete [CM] matrix in FAST output.")
        cm_matrix = np.array(cm_values).reshape(3,3)

        print("Matrix [CM] extracted successfully:")
        print(cm_matrix)

        # Looking for [SIGMA] or another stress marker
        sigma_matrix = None
        try:
            idx_sigma_start = next(i for i, line in enumerate(lines) if "SigmaXX" in line)
            sigma_lines = lines[idx_sigma_start:idx_sigma_start+3]
            sigma_values = []
            for line in sigma_lines:
                # Convert to float
                line = line.replace(',', '.')
                # Extracts numeric values ​​from the row
                values = [float(v) for v in line.strip().split() if v.replace('.', '', 1).isdigit()]
                if len(values) > 0:
                    sigma_values.append(values)
            if len(sigma_values) == 3:
                sigma_matrix = np.array(sigma_values).reshape(3, 1) # Assumindo 3x1
                print("\nMatrix [SIGMA] extracted successfully:")
                print(sigma_matrix)
        except StopIteration:
            print("\nStress marker (SigmaXX, etc.) not found in FAST output.")

        return cm_matrix, sigma_matrix

    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("FAST execution timeout.")
    except FileNotFoundError:
        raise RuntimeError(f"FAST executable not found: '{fast_exe_path}'")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

def CalcPropPlaneStress(C):
    """
    Calculates properties for Plane Stress.
    
    Args:
        C (np.ndarray): Stiffness matrix.
        
    Returns:
        tuple: (E, nu, G1, G2)
    """
    nu = C[0, 1] / C[0, 0]
    E = C[0, 0] * (1 - nu**2)
    G1 = C[2, 2]
    G2 = E / (2 * (1 + nu)) 
    return E, nu, G1, G2

def CalcPropPlaneStrain(C):
    """
    Calculates properties for Plane Strain.
    
    Args:
        C (np.ndarray): Stiffness matrix.
        
    Returns:
        tuple: (E, nu, G1, G2)
    """
    r = C[0, 1] / C[0, 0]
    nu = r / (1 + r)
    E = C[0, 0] * (1 + nu) * (1 - 2 * nu) / (1 - nu)
    G1 = C[2, 2]
    G2 = E / (2 * (1 + nu)) 
    return E, nu, G1, G2

def CalcPropPlaneStrain3x3(C):
    """
    Calculates properties for Plane Strain (for case 3x3).
    
    Args:
        C (np.ndarray): Stiffness matrix.
        
    Returns:
        tuple: (E11, E22, G12, nu12, nu21)
    """
    S = np.linalg.inv(C)
    E22 = 1 / S[0, 0] 
    E33 = 1 / S[1, 1] 
    G23 = 1 / S[2, 2] 
    nu23 = -S[0, 1] / S[0, 0] 
    nu32 = -S[1, 0] / S[1, 1]
    return E22, E33, G23, nu23, nu32

def calculate_average_stress_from_pos(pos_file_path):
    """
    Reads a .pos file from FAST and calculates the overall average stresses for each component.

    Args:
        pos_file_path (str): Full path to .pos file.

    Returns:
        tuple: (float, float, float, float) The average stresses (STRESS_XX, STRESS_YY, STRESS_ZZ, STRESS_XY).
    
    Raises:
        RuntimeError: If the file cannot be read or the format is incorrect.
    """
    print(f"Reading and processing .pos file: {pos_file_path}")
    
    try:
        with open(pos_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise RuntimeError(f".pos file not found: '{pos_file_path}'")

    all_stresses_xx = []
    all_stresses_yy = []
    all_stresses_zz = []
    all_stresses_xy = []
    
    in_data_block = False
    for line in lines:
        line = line.strip()
        # Finds the start header of the data block
        if line.startswith("%RESULT.CASE.STEP.ELEMENT.GAUSS.SCALAR.DATA"):
            in_data_block = True
            continue
        
        # If the line is not empty or a header in the data block
        if in_data_block and line and line[0] in ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            parts = line.split()
            # The first 4 values ​​after the element and gauss point identifiers are the stresses
            if len(parts) >= 4:
                try:
                    all_stresses_xx.append(float(parts[0]))
                    all_stresses_yy.append(float(parts[1]))
                    all_stresses_zz.append(float(parts[2]))
                    all_stresses_xy.append(float(parts[3]))
                except (ValueError, IndexError):
                    print(f"WARNING: Invalid data line, skipping: {line}")
                    continue
        
        # Exits the data block when the header of a new element is encountered
        if in_data_block and not line.startswith("%") and not line.startswith("+") and not line.startswith("-") and not line.startswith("0") and not line.startswith("1") and not line.startswith("2") and not line.startswith("3") and not line.startswith("4") and not line.startswith("5") and not line.startswith("6") and not line.startswith("7") and not line.startswith("8") and not line.startswith("9"):
            break

    if not all_stresses_xx:
        raise RuntimeError("No stress data found in .pos file. Check file format..")
        
    avg_xx = np.mean(all_stresses_xx)
    avg_yy = np.mean(all_stresses_yy)
    avg_zz = np.mean(all_stresses_zz)
    avg_xy = np.mean(all_stresses_xy)

    print("\n--- Global Average Stresses from .pos file ---")
    print(f"Average STRESS_XX: {avg_xx:.6e}")
    print(f"Average STRESS_YY: {avg_yy:.6e}")
    print(f"Average STRESS_ZZ: {avg_zz:.6e}")
    print(f"Average STRESS_XY: {avg_xy:.6e}")
    
    return avg_xx, avg_yy, avg_zz, avg_xy

def process_and_print_results(cm_matrix, sigma_matrix):
    """
    Processes CM and SIGMA matrices and prints elastic constants and stresses.
    
    Args:
        cm_matrix (np.ndarray): CM matrix extracted from FAST.
        sigma_matrix (np.ndarray): Stresses matrix extracted from FAST.
    """
    print("Starting calculation of elastic properties from the FAST CM matrix...")
    print("\nMatrix CM (FAST):")
    print(cm_matrix)
    
    # --- Properties for Plane Stress ---
    print("\n--- Properties for Plane Stress ---")
    E_ps, nu_ps, G1_ps, G2_ps = CalcPropPlaneStress(cm_matrix)
    print(f"E: {E_ps:.6e}")
    print(f"nu: {nu_ps:.6f}")
    print(f"G1: {G1_ps:.6e}")
    print(f"G2: {G2_ps:.6e}")

    # --- Properties for Plane Strain ---
    print("\n--- Properties for Plane Strain ---")
    E_pstrain, nu_pstrain, G1_pstrain, G2_pstrain = CalcPropPlaneStrain(cm_matrix)
    print(f"E: {E_pstrain:.6e}")
    print(f"nu: {nu_pstrain:.6f}")
    print(f"G1: {G1_pstrain:.6e}")
    print(f"G2: {G2_pstrain:.6e}")
    
    # --- Properties for Plane Strain (3x3) ---
    print("\n--- Properties for Plane Strain (3x3) ---")
    E11, E22, G12, nu12, nu21 = CalcPropPlaneStrain3x3(cm_matrix)
    print(f"E22: {E11:.6e}")
    print(f"E33: {E22:.6e}")
    print(f"G23: {G12:.6e}")
    print(f"nu23: {nu12:.6f}")
    print(f"nu32: {nu21:.6f}")
    
    # --- Stresses ---
    if sigma_matrix is not None:
        print("\n---Stresses extracted from the FAST output ---")
        print(f"SigmaXX: {sigma_matrix[0][0]:.6e}")
        print(f"SigmaYY: {sigma_matrix[1][0]:.6e}")
        print(f"SigmaXY: {sigma_matrix[2][0]:.6e}")
    else:
        print("\n--- Average Stresses (Sigma) ---")
        print("Stress values were not found at the FAST output.")

# Main Execution
if __name__ == "__main__":
    # Adapt the FAST executable path and .dat file name
    fast_exe_path = r"C:\Users\bruna\OneDrive\Abaqus_UFC\FASTv2.4.3 2\FASTv2.4.3\fast.exe"
    
    # Set the name of the .dat file you want to process
    dat_file_name_base = "SunVaidya_2D"
    #dat_file_name_base = "MieheKoch_2D"
    pos_file_path = os.path.join(os.path.dirname(fast_exe_path), dat_file_name_base + ".pos")

    try:
        # Try to run FAST and extract CM and SIGMA from standard output
        cm, sigma = run_fast(fast_exe_path, dat_file_name_base)
        process_and_print_results(cm, sigma)
    except RuntimeError as e:
        print(f"Error running FAST: {e}")
    
    print("\n" + "="*50)

    try:
        # Try calculate average stresses from .pos file
        calculate_average_stress_from_pos(pos_file_path)
    except RuntimeError as e:
        print(f"Error processing .pos file: {e}")