# --- Install python libraries in terminal - VS Code ---
# pip install pip (not necessary if already installed)
# pip install lmcv_tools

import numpy as np
import os
from pathlib import Path

# Import reorder from lmcv_tools:
from lmcv_tools.commands.reorder import start as lmcv_reorder_start

#######################################
# AUXILIARY FUNCTIONS
#######################################
def read_file_until_text(fd, text_to_find):
    """
    Reads a file line by line until a specific text is found.
    Args:
        fd (file object): The file descriptor.
        text_to_find (str): The text string to search for.
    Raises:
        EOFError: If the text is not found before the end of the file.
    """
    while True:
        line = fd.readline()
        if not line:  # End of the file
            raise EOFError(f"Text {text_to_find} not found in file.")
        if text_to_find in line:
            return

def read_materials(fd):
    """
    Reads the physical names (materials) from the $PhysicalNames block.
    Args:
        fd (file object): The file descriptor.
    Returns:
        tuple: (number of materials, list of materials, list of material names)
    """
    read_file_until_text(fd, '$PhysicalNames')
    line_with_n_materials = fd.readline().strip()

    try:
        n_materials = int(line_with_n_materials)
    except ValueError:
        print(f"ERROR: Could not convert line  '{line_with_n_materials}' to an integer.")
        return 0, [], []

    materials = []
    mat_names = []

    for i in range(n_materials):
        line = fd.readline().strip()
        parts = line.split()
        if len(parts) < 3:
            print(f"ERROR: Poorly formatted line in material {i+1}: '{line}'")
            continue
        try:
            dim = int(parts[0])
            phys_id = int(parts[1])
            name = parts[2].strip('"')
            materials.append((dim, phys_id, name))
            mat_names.append(name)
        except ValueError as e:
            print(f"Error converting material on line {i+1}: '{line}' -> {e}")
            continue

    return len(materials), materials, mat_names

def read_nodes(fd):
    """
    Reads the nodes from the $Nodes block (Gmsh format 4.1).
    Args:
        fd (file object): The file descriptor.
    Returns:
        np.array: A NumPy array containing node ID and coordinates (id, x, y, z).
    """
    read_file_until_text(fd, '$Nodes')
    # Line after $Nodes contains: numEntityBlocks numNodes minNodeTag maxNodeTag
    header_line = fd.readline().strip()
    parts = header_line.split()
    if len(parts) < 4:
        raise ValueError(f"Invalid format in $Nodes header: '{header_line}'")
    numEntityBlocks = int(parts[0])
    numNodes = int(parts[1])

    nodes = np.zeros((numNodes, 4))  # id, x, y, z
    node_index = 0
    for _ in range(numEntityBlocks):
        # Each block starts with: entityDim entityTag parametric numNodesInBlock
        block_header = fd.readline().strip().split()
        if len(block_header) < 4:
            raise ValueError(f"Invalid format in nodes block: '{block_header}'")
        numNodesInBlock = int(block_header[3])

        # Node IDs in this block
        node_ids = []
        for _ in range(numNodesInBlock):
            nid = int(fd.readline().strip())
            node_ids.append(nid)

        # Node coordinates (x, y, z)
        for nid in node_ids:
            coords_line = fd.readline().strip()
            coords = list(map(float, coords_line.split()))
            if len(coords) < 3:
                raise ValueError(f"Incomplete coordinates for the node {nid}: '{coords_line}'")
            
            nodes[node_index, 0] = nid
            nodes[node_index, 1] = coords[0]
            nodes[node_index, 2] = coords[1]
            nodes[node_index, 3] = coords[2]
            node_index += 1

    read_file_until_text(fd, '$EndNodes')
    return nodes

def read_elems(fd):
    """
    Reads the elements from the $Elements block.
    Args:
        fd (file object): The file descriptor.
    Returns:
        list: A list of lists, where each inner list contains element data
              (id, type, physical_group_id, node IDs...).
    """
    read_file_until_text(fd, '$Elements')
    # Line after $Elements contains: numEntityBlocks numElements minElemTag maxElemTag
    header_line = fd.readline().strip()
    parts = header_line.split()
    if len(parts) < 4:
        raise ValueError(f"Invalid Format in $Elements header: '{header_line}'")
    numEntityBlocks = int(parts[0])
    numElements = int(parts[1])

    valid_elem_types = {2, 3, 9, 16, 17}
    elements = []
    elem_count = 0

    for _ in range(numEntityBlocks):
        # Each block starts with: entityDim entityTag elemType numElementsInBlock
        block_header = fd.readline().strip().split()
        if len(block_header) < 4:
            raise ValueError(f"Invalid format in elements block: '{block_header}'")
        entity_tag = int(block_header[1])  # Physical group ID
        elemType = int(block_header[2])
        numElementsInBlock = int(block_header[3])

        if elemType not in valid_elem_types:
            # Ignore elements of unsupported types in this block
            for _ in range(numElementsInBlock):
                fd.readline()
            continue

        for _ in range(numElementsInBlock):
            line = fd.readline().strip()
            parts = line.split()
            if len(parts) < 2:
                print(f"WARNING: Line ignored in elements - less than 2 values: '{line}'")
                continue

            elem_id = int(parts[0])
            node_ids = list(map(int, parts[1:]))

            # Reorder the nodes according to element type 
            if elemType == 2:  # T3
                node_ids = [node_ids[0], node_ids[1], node_ids[2]]
            elif elemType == 9:  # T6
                node_ids = [node_ids[0], node_ids[3], node_ids[1], node_ids[4], node_ids[2], node_ids[5]]
            elif elemType == 3:  # Q4
                node_ids = [node_ids[0], node_ids[1], node_ids[2], node_ids[3]]
            elif elemType == 16:  # Q8
                node_ids = [node_ids[0], node_ids[4], node_ids[1], node_ids[5], node_ids[2], node_ids[6], node_ids[3], node_ids[7]]
            elif elemType == 17:  # Q9
                node_ids = [node_ids[0], node_ids[4], node_ids[1], node_ids[5], node_ids[2], node_ids[6], node_ids[3], node_ids[7], node_ids[8]]

            # Store element data with the physical group ID
            elements.append([elem_id, elemType, entity_tag] + node_ids)
            elem_count += 1

    read_file_until_text(fd, '$EndElements')
    print(f"Total valid elements read: {elem_count}")
    return elements

def micro_model_2D(epsM, nodes):
    """
    Identifies corner and edge nodes for 2D RVE and prints boundary conditions.
    Args:
        epsM (np.array): The macroscopic strain tensor.
        nodes (np.array): A NumPy array of nodes (id, x, y, z).
    Returns:
        tuple: (vertex nodes, bottom edge nodes, top edge nodes,
                left edge nodes, right edge nodes)
    """
    xmin, ymin = np.min(nodes[:, 1]), np.min(nodes[:, 2])
    xmax, ymax = np.max(nodes[:, 1]), np.max(nodes[:, 2])
    lx = xmax - xmin
    ly = ymax - ymin

    tol = 1e-6 * min(lx, ly) if min(lx, ly) > 0 else 1e-6

    nn = len(nodes)

    # Lists to append nodes, then convert to numpy arrays
    nodel_list, noder_list, nodeb_list, nodet_list = [], [], [], []
    vtx1, vtx2, vtx3, vtx4 = -1, -1, -1, -1

    for i in range(nn):
        # Index adjustment for node ID
        node_id = int(nodes[i, 0])  # Original node ID
        x = nodes[i, 1]
        y = nodes[i, 2]

        if np.isclose(x, 0, atol=tol):  # Left edge
            if np.isclose(y, 0, atol=tol):
                vtx1 = node_id
            elif np.isclose(y, ly, atol=tol):
                vtx4 = node_id
            else:
                nodel_list.append([node_id, x, y])
        elif np.isclose(x, lx, atol=tol):  # Right Edge
            if np.isclose(y, 0, atol=tol):
                vtx2 = node_id
            elif np.isclose(y, ly, atol=tol):
                vtx3 = node_id
            else:
                noder_list.append([node_id, x, y])
        elif np.isclose(y, 0, atol=tol):  # Bottom edge (excluding corners)
            nodeb_list.append([node_id, x, y])
        elif np.isclose(y, ly, atol=tol):  # Top edge (excluding corners)
            nodet_list.append([node_id, x, y])

    vertex = [vtx1, vtx2, vtx3, vtx4]

    # Convert lists to numpy arrays and sort
    nodel = np.array(nodel_list) if nodel_list else np.array([])
    noder = np.array(noder_list) if noder_list else np.array([])
    nodeb = np.array(nodeb_list) if nodeb_list else np.array([])
    nodet = np.array(nodet_list) if nodet_list else np.array([])

    # Sort based on y-cordinate for vertical edges, x-cordinate for horizontal edges
    if nodel.size > 0:
        nodel = nodel[nodel[:, 2].argsort()]
    if noder.size > 0:
        noder = noder[noder[:, 2].argsort()]
    if nodeb.size > 0:
        nodeb = nodeb[nodeb[:, 1].argsort()]
    if nodet.size > 0:
        nodet = nodet[nodet[:, 1].argsort()]

    # Check consistency of the mesh
    nl = len(nodel)
    nr = len(noder)
    nb = len(nodeb)
    nt = len(nodet)

    if nl != nr:
        print(f'Invalid mesh: nl = {nl}  nr = {nr}')
    if nb != nt:
        print(f'Invalid mesh: nb = {nb}  nt = {nt}')

    for i in range(nl):
        if not np.isclose(nodel[i, 2], noder[i, 2], atol=tol):
            print(f'Invalid mesh: y({nodel[i, 0]}) = {nodel[i, 2]:.6f} y({noder[i, 0]}) = {noder[i, 2]:.6f}')
    for i in range(nb):
        if not np.isclose(nodeb[i, 1], nodet[i, 1], atol=tol):
            print(f'Invalid mesh: x({nodeb[i, 0]}) = {nodeb[i, 1]:.6f} x({nodet[i, 0]}) = {nodet[i, 1]:.6f}.')

    print('%%MICRO.MODEL.DIMENSION')
    print('2\n')
    print('%%MICRO.MESH.VERTEX.NODES')
    print(f'{vtx1}   {vtx2}   {vtx3}   {vtx4}\n')
    print('%%MICRO.MESH.EDGE.NODES')
    print(f'{nl}')
    if nodel.size > 0:
        print(' '.join(map(str, nodel[:, 0].astype(int))))
    if noder.size > 0:
        print(' '.join(map(str, noder[:, 0].astype(int))))
    print(f'{nb}')
    if nodeb.size > 0:
        print(' '.join(map(str, nodeb[:, 0].astype(int))))
    if nodet.size > 0:
        print(' '.join(map(str, nodet[:, 0].astype(int))))
    print('\n')

    return vertex, nodeb, nodet, nodel, noder

def calc_H(x, y):
    """
    Calculates the H matrix for displacement field.
    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
    Returns:
        np.array: The H matrix.
    """
    H = np.array([
        [x, 0],
        [0, y],
        [y/2, x/2],
    ])
    return H

#######################################
# MAIN FUNCTION
#######################################
def rve_msh2dat():
    """
    Main function to read a Gmsh .msh file and generate a .dat file
    """
    # Strain field.
    epsM = np.array([0.001, 0.0, 0.0])
    #epsM = np.array([0.4, 0.0, 0.0])
    #epsM = np.array([0.4, 0.0, 0.0])
    print(f"epsM: {epsM}")

    # Define input file and related data.
    fname = 'SunVaidya_2D'
    #fname = 'MieheKoch_2D'
    E_modulus = np.array([68.3e9, 379.3e9])
    #E_modulus = np.array([2.08264e1])
    nu_poisson = np.array([0.3, 0.1])
    #nu_poisson = np.array([0.301653])
    kp = 1.0e4  # Spring stiffness
    #kp = 1.0e-6

    print(f"File name: {fname}")
    msh_name = f"{fname}.msh"

    # Gets the full path to the .msh file (needs to be fixed to user-specified directory)
    msh_path = Path(r"C:\Users\<username>\folder\subfolder") / msh_name
    try:
        fmsh = open(msh_name, 'r')
    except IOError:
        print(f"Error: file {msh_name} could not be opened!")
        return

    try:
        # Read the materials of the RVE.
        num_mat, materials_list, mat_name = read_materials(fmsh)
        n_E = len(E_modulus)

        # The physical entity's dimension tag is the first element of the tuple, `m[0]`
        if num_mat > 0 and materials_list[0][0] == 2 and num_mat == n_E:
            print(f"2D RVE with {num_mat} Materials.")
        else:
            print("Invalid file - 3D RVE or PhysicalNames not defined.")
            fmsh.close()
            return

        # Read nodes
        nodes = read_nodes(fmsh)
        nn = len(nodes)
        if nn == 0:
            print("Error: number of nodes = 0.")
            fmsh.close()
            return

        # Translate the origin to (0, 0).
        xmin = np.min(nodes[:, 1])
        ymin = np.min(nodes[:, 2])
        nodes[:, 1] -= xmin
        nodes[:, 2] -= ymin
        print(f"Nodes translated. xmin: {xmin}, ymin: {ymin}")

        # Read elements.
        elements = read_elems(fmsh)
        ne = len(elements)
        if ne == 0:
            print("Error: number of elements = 0.")
            fmsh.close()
            return

        # Get element type and related data.
        # Assuming all elements are of the same type
        elem_type = elements[0][1]  # Second column is element type
        elm_string = ''
        nne = 0  # Number of nodes per element
        ngr, ngs, ngt = 0, 0, 0  # Integration order

        if elem_type == 2:  # Element T3
            elm_string = '%ELEMENT.PLSTRAIN.T3'
            nne = 3
            ngr, ngs, ngt = 1, 1, 1
        elif elem_type == 3:  # Element Q4
            elm_string = '%ELEMENT.PLSTRAIN.Q4'
            nne = 4
            ngr, ngs, ngt = 2, 2, 1
        elif elem_type == 9:  # Element T6
            elm_string = '%ELEMENT.PLSTRAIN.T6'
            nne = 6
            ngr, ngs, ngt = 2, 1, 1
        elif elem_type == 16:  # Element Q8
            elm_string = '%ELEMENT.PLSTRAIN.Q8'
            nne = 8
            ngr, ngs, ngt = 2, 2, 1
        elif elem_type == 17:  # Element Q9
            elm_string = '%ELEMENT.PLSTRAIN.Q9'
            nne = 9
            ngr, ngs, ngt = 3, 3, 1
        else:
            print(f"Error: Unsupported element type: {elem_type}")
            fmsh.close()
            return

        # Open dat file and write header
        # Path needs to be fixed to user-specified FAST directory
        output_dir = r"C:\Users\<username>\some_folder\subfolder"
        dat_name = os.path.join(output_dir, f"{fname}.dat")
        try:
            fdat = open(dat_name, 'w')
        except IOError:
            print(f"Error: file {dat_name} could not be opened for writing!")
            fmsh.close()
            return

        fdat.write('%HEADER\n')
        fdat.write('File generated by Convert_msh2dat.\n\n')

        # Write nodes to the dat file.
        fdat.write('%NODE\n')
        fdat.write(f'{nn}\n\n')

        fdat.write('%NODE.COORD\n')
        fdat.write(f'{nn}\n')
        # nodes[:, 0] contains original IDs
        for i in range(nn):
            fdat.write(f'{int(nodes[i, 0]):-5d}  {nodes[i, 1]:17.12e}  {nodes[i, 2]:17.12e}  {nodes[i, 3]:17.12e}\n')

        # Get the vertices and edges of the RVE.
        vertex, nodeb, nodet, nodel, noder = micro_model_2D(epsM, nodes)
        print("Micro-model data obtained.")

        # Write node supports.
        fdat.write('\n%NODE.SUPPORT\n')
        fdat.write('1\n')  # Only 1 node supported
        fdat.write(f'{vertex[0]:-3d}  1   1   0   0   0   0\n')

        # Write springs.
        fdat.write('\n%SPRING.PROPERTY\n')
        fdat.write('1\n')
        fdat.write(f'1   linear   {kp:0.3e}\n')

        fdat.write('\n%SPRING.SUPPORT\n')
        fdat.write('6\n')
        fdat.write(f'1   {vertex[1]:-3d}    1     1\n')
        fdat.write(f'2   {vertex[1]:-3d}    2     1\n')
        fdat.write(f'3   {vertex[2]:-3d}    1     1\n')
        fdat.write(f'4   {vertex[2]:-3d}    2     1\n')
        fdat.write(f'5   {vertex[3]:-3d}    1     1\n')
        fdat.write(f'6   {vertex[3]:-3d}    2     1\n')

        # Write materials and sections.
        fdat.write('\n%MATERIAL\n')
        fdat.write(f'{num_mat}\n')

        fdat.write('\n%MATERIAL.ISOTROPIC\n')
        fdat.write(f'{num_mat}\n')
        for i, mat in enumerate(materials_list):
            fdat.write(f'{i+1}     {E_modulus[i]:0.5e}     {nu_poisson[i]:0.5f}\n')

        fdat.write('\n%SECTION\n')
        fdat.write(f'{num_mat}\n')

        fdat.write('\n%SECTION.HOMOGENEOUS.ISOTROPIC.2D\n')
        fdat.write(f'{num_mat}\n')
        for i, mat in enumerate(materials_list):
            fdat.write(f'{i+1}     {i+1}     1.0\n')

        # Write integration order - one per physical group
        fdat.write('\n%INTEGRATION.ORDER\n')
        fdat.write(f'{num_mat}\n')
        for i in range(num_mat):
            fdat.write(f'{i+1}     {ngr}     {ngs}     {ngt}     {ngr}     {ngs}     {ngt}\n')

        # Write elements.
        fdat.write('\n%ELEMENT\n')
        fdat.write(f'{ne}\n')

        fdat.write(f'\n{elm_string}\n')
        fdat.write(f'{ne}\n')

        # Loop for writing elements
        for elem_data in elements:
            elem_id = elem_data[0]
            phys_group = elem_data[2] 
            try:
                # Find the index of the physical group ID in the materials list
                # materials_list is [(dim, phys_id, name), ...]
                section_id = [m[1] for m in materials_list].index(phys_group) + 1
            except ValueError:
                section_id = 1  # If not found, assume 1
            
            node_ids = elem_data[3:]
            node_format_str = ' '.join([f' {nid:-4d}' for nid in node_ids])
            fdat.write(f'{elem_id:-4d}   {section_id}     {section_id} {node_format_str}\n')

        # Find row indices for vertex IDs for array access
        vtx1_row = np.where(nodes[:, 0] == vertex[0])[0][0]
        vtx2_row = np.where(nodes[:, 0] == vertex[1])[0][0]
        vtx3_row = np.where(nodes[:, 0] == vertex[2])[0][0]
        vtx4_row = np.where(nodes[:, 0] == vertex[3])[0][0]

        h1 = calc_H(nodes[vtx1_row, 1], nodes[vtx1_row, 2])
        h2 = calc_H(nodes[vtx2_row, 1], nodes[vtx2_row, 2])
        h3 = calc_H(nodes[vtx3_row, 1], nodes[vtx3_row, 2])
        h4 = calc_H(nodes[vtx4_row, 1], nodes[vtx4_row, 2])

        # Displacement on corners
        u1 = h1.T @ epsM
        u2 = h2.T @ epsM
        u3 = h3.T @ epsM
        u4 = h4.T @ epsM
        u3 = u2 + u4

        # Writes the constraint equations
        fdat.write('\n%REMARK\n')
        fdat.write(f'epsM = [{epsM[0]}, {epsM[1]}, {epsM[2]}]\n')

        nl_len = len(nodel)
        nb_len = len(nodeb)
        # 6 fixed vertices + 2 constraints per node for left-right and bottom-top edges
        nc = 6 + 2 * (nl_len + nb_len)

        fdat.write('\n%NODE.MULTI-POINT.CONSTRAINT\n')
        fdat.write(f'{nc}\n')

        # Fixed vertex constraints (Vertex 2, 3, 4)
        fdat.write(f'1 {vertex[1]:5d}   1   1.0   {u2[0]:e}\n')
        fdat.write(f'1 {vertex[1]:5d}   2   1.0   {u2[1]:e}\n')
        fdat.write(f'1 {vertex[2]:5d}   1   1.0   {u3[0]:e}\n')
        fdat.write(f'1 {vertex[2]:5d}   2   1.0   {u3[1]:e}\n')
        fdat.write(f'1 {vertex[3]:5d}   1   1.0   {u4[0]:e}\n')
        fdat.write(f'1 {vertex[3]:5d}   2   1.0   {u4[1]:e}\n')

        # Edge constraints (left-right)
        for i in range(nl_len):
            # Noder(i, 0) and Nodel(i, 0) are node IDs
            fdat.write(f'2 {int(noder[i, 0]):5d}   1   1.0   {int(nodel[i, 0]):5d}   1   -1.0   {u2[0]:e}\n')
            fdat.write(f'2 {int(noder[i, 0]):5d}   2   1.0   {int(nodel[i, 0]):5d}   2   -1.0   {u2[1]:e}\n')

        # Edge constraints (bottom-top)
        for i in range(nb_len):
            # Nodet(i, 0) and Nodeb(i, 0) are node IDs
            fdat.write(f'2 {int(nodet[i, 0]):5d}   1   1.0   {int(nodeb[i, 0]):5d}   1   -1.0   {u4[0]:e}\n')
            fdat.write(f'2 {int(nodet[i, 0]):5d}   2   1.0   {int(nodeb[i, 0]):5d}   2   -1.0   {u4[1]:e}\n')

        fdat.write('\n%END\n')
        print("\nSuccessfully generated .dat file.")

    except EOFError as e:
        print(f"Error reading .msh file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        fmsh.close()
        if 'fdat' in locals() and not fdat.closed:
            fdat.close()

    # Call reorder function
    print("\nStarting file reordering .dat...")
    reordering_method = 'sloan' # or rcm
    flags_to_use = {'-i': 'info'} 
    
    lmcv_reorder_start(reordering_method, str(dat_name), flags_to_use)
    print(f"Reordering complete. The %NODE.SOLVER.ORDER block has been added to '{dat_name}'.")

    try:
        with open(dat_name, 'r') as f:
            lines = f.readlines()
        
        # Find the positions of the blocks
        node_coord_start = -1
        solver_order_start = -1
        end_line = -1

        for i, line in enumerate(lines):
            if line.strip() == '%NODE.COORD':
                node_coord_start = i
            elif line.strip() == '%NODE.SOLVER.ORDER':
                solver_order_start = i
            elif line.strip() == '%END':
                end_line = i

        if node_coord_start == -1 or solver_order_start == -1 or end_line == -1:
            print("ERROR: Could not find required blocks in file. The file will not be overwritten..")
            return

        # Separates the reordering block
        solver_order_block = []
        for i in range(solver_order_start, len(lines)):
            line = lines[i]
            if i > solver_order_start and line.strip().startswith('%'):
                break  
            solver_order_block.append(line)

        f_solver_order_block = []
        for idx, line in enumerate(solver_order_block):
            if idx == 0:  
                f_solver_order_block.append(line)
                continue
            if line.strip().isdigit():
                f_solver_order_block.append(f"{int(line.strip()):d}\n")
            else:
                ids = line.split()
                f_line = "".join(f"{int(nid):6d}" for nid in ids) + "\n"
                f_solver_order_block.append(f_line)

        # Remove the end block and the %END line
        del lines[solver_order_start:solver_order_start + len(solver_order_block)]
        del lines[end_line:]

        # Inserts the reordering block after the coordinates
        node_coord_end = node_coord_start + 1 + nn + 1 
        new_lines = lines[:node_coord_end] + ["\n"] + f_solver_order_block + ["\n"] + lines[node_coord_end:]
        new_lines.append('%END\n')

        # Rewrites the file
        with open(dat_name, 'w') as f:
            f.writelines(new_lines)
        
        print(f"Positioning of blocko %NODE.SOLVER.ORDER successfully adjusted in file '{dat_name}'.")

    except Exception as e:
        print(f"Error rewriting file to move reorder block: {e}")

if __name__ == '__main__':
    rve_msh2dat()

