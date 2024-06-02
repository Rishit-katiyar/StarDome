import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def subdivide_triangle(v1, v2, v3, max_edge_length):
    edge_lengths = [np.linalg.norm(v1 - v2), np.linalg.norm(v2 - v3), np.linalg.norm(v3 - v1)]
    if max(edge_lengths) <= max_edge_length:
        return [(v1, v2, v3)]
    
    v12 = normalize((v1 + v2) / 2)
    v23 = normalize((v2 + v3) / 2)
    v31 = normalize((v3 + v1) / 2)
    
    return (
        subdivide_triangle(v1, v12, v31, max_edge_length) +
        subdivide_triangle(v2, v23, v12, max_edge_length) +
        subdivide_triangle(v3, v31, v23, max_edge_length) +
        subdivide_triangle(v12, v23, v31, max_edge_length)
    )

def normalize(v):
    return v / np.linalg.norm(v)

def create_geodesic_dome(radius, max_edge_length):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])

    vertices = np.array([normalize(v) for v in vertices])
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    triangles = []
    for face in faces:
        v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        triangles.extend(subdivide_triangle(v1, v2, v3, max_edge_length))
    
    triangles = [(v1 * radius, v2 * radius, v3 * radius) for v1, v2, v3 in triangles]
    triangles = [tri for tri in triangles if all(v[2] >= 0 for v in tri)]
    
    return triangles

def plot_dome(triangles, ax):
    for tri in triangles:
        tri = np.array(tri)
        ax.add_collection3d(Poly3DCollection([tri], color='lightblue', alpha=0.5, edgecolor='k'))
        # Annotate edge lengths
        for i in range(3):
            ax.text(tri[i][0], tri[i][1], tri[i][2], s=f"{np.linalg.norm(tri[(i+1)%3] - tri[i]):.2f}", color='black')

def plot_roof(size, ax):
    corners = np.array([
        [-size/2, -size/2, 0], [size/2, -size/2, 0],
        [size/2, size/2, 0], [-size/2, size/2, 0]
    ])
    roof = Poly3DCollection([corners], color='tan', alpha=0.5, edgecolor='k')
    ax.add_collection3d(roof)
    ax.text(-size/2, 0, 0, s=f"Roof Size: {size} ft", color='black')

def plot_base(dome_radius, ax):
    # Creating a regular hexagon as the base
    theta = np.linspace(0, 2*np.pi, 7)
    x = dome_radius * np.cos(theta)
    y = dome_radius * np.sin(theta)
    z = np.zeros_like(x)
    base_vertices = np.vstack([x, y, z]).T
    base = Poly3DCollection([base_vertices], color='yellow', alpha=0.5, edgecolor='k')
    ax.add_collection3d(base)
    ax.text(0, 0, 0, s=f"Dome Base", color='black')

def connect_base_to_panel(base_vertices, panel_vertices, ax):
    for base_vertex in base_vertices:
        distances = [np.linalg.norm(base_vertex[:2] - panel_vertex[:2]) for panel_vertex in panel_vertices]
        nearest_panel_vertex = panel_vertices[np.argmin(distances)]
        metal_support = np.array([base_vertex, nearest_panel_vertex])
        ax.plot(metal_support[:,0], metal_support[:,1], metal_support[:,2], color='gray', linewidth=2)

def visualize_dome_with_support(dome_radius=5, roof_size=10, max_edge_length=3):
    triangles = create_geodesic_dome(dome_radius, max_edge_length)
    base_vertices = np.array([[dome_radius * np.cos(theta), dome_radius * np.sin(theta), 0] for theta in np.linspace(0, 2*np.pi, 7)])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    plot_roof(roof_size, ax)
    plot_dome(triangles, ax)
    plot_base(dome_radius, ax)

    # Plotting the metal support structure
    for tri in triangles:
        tri = np.array(tri)
        for i in range(3):
            ax.plot([tri[i][0], tri[(i+1)%3][0]], [tri[i][1], tri[(i+1)%3][1]], [tri[i][2], tri[(i+1)%3][2]], color='gray', linewidth=2)

    connect_base_to_panel(base_vertices, np.array([vertex for tri in triangles for vertex in tri]), ax)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_xlim([-roof_size/2, roof_size/2])
    ax.set_ylim([-roof_size/2, roof_size/2])
    ax.set_zlim([0, dome_radius+2])
    ax.set_title('Glass Dome with Metal Support Structure', fontsize=16, pad=20)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.zaxis.set_tick_params(width=2)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.view_init(elev=20, azim=30)  # Adjust the view angle for better visibility
    plt.tight_layout()
    plt.show()

visualize_dome_with_support()
