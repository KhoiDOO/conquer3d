import plotly.graph_objects as go
import numpy as np
import kaolin as kal

def visualize_mesh(
    mesh: kal.rep.SurfaceMesh, 
    plot_normals=False, 
    plot_wireframe=False,
    plot_vertex_colors=False,
    normal_length=0.05, 
    step=1
):
    fig = go.Figure()
    v = mesh.vertices.detach().cpu().numpy()
    f = mesh.faces.detach().cpu().numpy()
    n = mesh.vertex_normals.detach().cpu().numpy()
    
    # Prepare base properties for the Mesh3D trace
    mesh_kwargs = {
        'x': v[:, 0], 'y': v[:, 1], 'z': v[:, 2],
        'i': f[:, 0], 'j': f[:, 1], 'k': f[:, 2],
        'opacity': 0.8,
        'name': 'Mesh',
        'showlegend': True
    }

    if plot_vertex_colors:
        assert hasattr(mesh, 'vertex_colors') or mesh.has_attribute('vertex_colors'), "Mesh does not have vertex colors"
        vertex_colors = mesh.vertex_colors.detach().cpu().numpy()
        
        # Scale [0, 1] color values properly to [0, 255] format if necessary
        if np.issubdtype(vertex_colors.dtype, np.floating) and vertex_colors.max() <= 2.0:
            vertex_colors = np.clip(vertex_colors, 0.0, 1.0)
            vertex_colors = (vertex_colors * 255).astype(np.uint8)
        else:
            vertex_colors = np.clip(vertex_colors, 0, 255).astype(np.uint8)
            
        # Plotly expects an array/list of css rgb strings
        rgb_strings = [f'rgb({r},{g},{b})' for r, g, b in vertex_colors]
        mesh_kwargs['vertexcolor'] = rgb_strings
    else:
        mesh_kwargs['color'] = 'lightblue'
    
    # Plot the surface mesh with the kwargs constructed above
    fig.add_trace(go.Mesh3d(**mesh_kwargs))
    
    if plot_normals:
        x_lines, y_lines, z_lines = [], [], []

        for i in range(0, len(v), step):
            x_lines.extend([v[i, 0], v[i, 0] + n[i, 0] * normal_length, None])
            y_lines.extend([v[i, 1], v[i, 1] + n[i, 1] * normal_length, None])
            z_lines.extend([v[i, 2], v[i, 2] + n[i, 2] * normal_length, None])
            
        fig.add_trace(go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(color='red', width=2),
            name='Normals'
        ))
    
    if plot_wireframe:
        edges_x = []
        edges_y = []
        edges_z = []
        for i, j, k in f:
            # Edge 1: vertex i to j
            edges_x.extend([v[i, 0], v[j, 0], None])
            edges_y.extend([v[i, 1], v[j, 1], None])
            edges_z.extend([v[i, 2], v[j, 2], None])
            # Edge 2: vertex j to k
            edges_x.extend([v[j, 0], v[k, 0], None])
            edges_y.extend([v[j, 1], v[k, 1], None])
            edges_z.extend([v[j, 2], v[k, 2], None])
            # Edge 3: vertex k to i
            edges_x.extend([v[k, 0], v[i, 0], None])
            edges_y.extend([v[k, 1], v[i, 1], None])
            edges_z.extend([v[k, 2], v[i, 2], None])
        
        fig.add_trace(go.Scatter3d(
            x=edges_x,
            y=edges_y,
            z=edges_z,
            mode='lines',
            line=dict(color='darkblue', width=2),
            name='Mesh Edges',
            showlegend=True
        ))
        
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title='Mesh rendering',
        width=800,
        height=800
    )

    fig.show()