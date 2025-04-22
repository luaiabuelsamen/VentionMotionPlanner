import open3d as o3d

# Load mesh
mesh = o3d.io.read_triangle_mesh("assets/ur5e/mesh/vention/base_link_2.STL")

# Simplify mesh (target_number_of_triangles is your desired face count)
mesh_simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=20000)

# Compute normals - this is the missing step
mesh_simplified.compute_vertex_normals()
# Alternatively, you can use: mesh_simplified.compute_triangle_normals()

# Save the simplified mesh
o3d.io.write_triangle_mesh("simplified_base_link_2.stl", mesh_simplified)