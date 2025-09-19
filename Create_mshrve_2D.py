# --- Install python libraries in terminal - VS Code ---
# pip install pip 
# pip install gmsh-sdk

import sys
import math
import gmsh
import os

print("Current directory:", os.getcwd())  # Change the current directory in terminal if necessary: cd "C:\Users\bruna\OneDrive\Abaqus_UFC" 

# --- Input parameters ---
r = 0.3868 
#r = 0.2  
volFrac = 47.0  
#volFrac = 12.5663706144
prefix = 'SunVaidya_2D'
#prefix = 'MieheKoch_2D'
nInc = 1  
element_type = 'T3'  # Choose between: 'T3', 'T6', 'Q4', 'Q8'

# --- Inicialization and Geometric Calculus ---
areaInc = math.pi * r**2  
a = (nInc * areaInc * 100.0 / volFrac) ** 0.5
areaTot = a ** 2

print('a       =', a)
print('areaTot =', areaTot)
print('areaInc =', areaInc)
print('volFrac =', 100.0 * nInc * areaInc / areaTot)

gmsh.initialize(sys.argv)  
gmsh.model.add("rve_model")

# --- Geometry ---
square_tag = gmsh.model.occ.addRectangle(0, 0, 0, a, a)   # Parameters (x, y, z, dx, dy), with z=0 for 2D
circle_tag = gmsh.model.occ.addDisk(a / 2, a / 2, 0, r, r) # Parameters (x_center, y_center, z_center, x_radius, y_radius)
#circle_tag = gmsh.model.occ.addDisk(a / 2 - r, a / 2 + r, 0, r, r)  # Eccentric Hole
gmsh.model.occ.synchronize()

# --- Fragmentation (separates matrix and inclusion) ---
gmsh.model.occ.fragment([(2, square_tag)], [(2, circle_tag)])   # this line should become a comment if there is a hole
#matrix_with_hole, _ = gmsh.model.occ.cut([(2, square_tag)], [(2, circle_tag)])   # line required if there is a hole instead of inclusion in the rve
gmsh.model.occ.synchronize()

# --- Surfaces ---
surfaces = gmsh.model.getEntities(2)
print("\n=== Surfaces ===")
def bbox_area(bbox):
    return (bbox[3]-bbox[0]) * (bbox[4]-bbox[1])

for s in surfaces:
    tag = s[1]
    bbox = gmsh.model.getBoundingBox(2, tag)
    size = [(bbox[3]-bbox[0]), (bbox[4]-bbox[1])]
    print(f"Surface tag={tag}, BBox={bbox}, Size={size}, Area≈{bbox_area(bbox)}")

# --- Automatically identify matrix and fiber ---
surfaces_sorted = sorted(surfaces, key=lambda s: bbox_area(gmsh.model.getBoundingBox(2, s[1])))
matrix_tag = surfaces_sorted[-1][1]  # largest area
inclusion_tag = surfaces_sorted[0][1] # smallest área

print(f"\nMatrix tag = {matrix_tag}")
print(f"Fiber tag  = {inclusion_tag}")

# --- Physical Groups ---
pg_matrix = gmsh.model.addPhysicalGroup(2, [matrix_tag])
gmsh.model.setPhysicalName(2, pg_matrix, "matrix")

pg_inclusion = gmsh.model.addPhysicalGroup(2, [inclusion_tag])   # this line should become a comment if there is a hole
gmsh.model.setPhysicalName(2, pg_inclusion, "fiber")             # this line should become a comment if there is a hole
gmsh.model.occ.synchronize()                                     # this line should become a comment if there is a hole

# --- Size Fields ---
size_matrix = 0.1
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_matrix)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_matrix)

# --- Element Type and Recombination ---
if element_type in ['T3', 'T6']:
    order = 1 if element_type == 'T3' else 2
    recombine = False
elif element_type in ['Q4', 'Q8']:
    order = 1 if element_type == 'Q4' else 2
    recombine = True
else:
    raise ValueError("Invalid element type.")

gmsh.option.setNumber("Mesh.ElementOrder", order)
gmsh.option.setNumber("Mesh.Algorithm", 8)   
if element_type == "Q8":                                  # Necessary to force the generation of Q8 elements (gmsh doesn't add the central node)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)   

if recombine:
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.setRecombine(2, matrix_tag)
    gmsh.model.mesh.setRecombine(2, inclusion_tag)

# --- Generate the mesh ---
gmsh.model.mesh.generate(2)
gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)  # Necessary to force the generation of Q8 elements (gmsh doesn't add the central node)

# --- Export to msh format ---
gmsh.write(prefix + ".msh")
gmsh.finalize()

