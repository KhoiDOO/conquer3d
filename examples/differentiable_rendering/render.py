import numpy as np
import math

import torch
import kaolin as kal
import nvdiffrast.torch as dr

def get_random_camera_batch(batch_size, fovy = np.deg2rad(45), iter_res=[512,512], cam_near_far=[0.1, 1000.0], cam_radius=3.0, device="cuda"):
    camera_pos = torch.stack(
        kal.ops.coords.spherical2cartesian(
            *kal.ops.random.sample_spherical_coords(
                (batch_size,), 
                azimuth_low=0., 
                azimuth_high=math.pi * 2,
                elevation_low=-math.pi / 2., 
                elevation_high=math.pi / 2., 
                device=device
            ),
            cam_radius
        ), dim=-1
    )
    return kal.render.camera.Camera.from_args(
        eye=camera_pos + torch.rand((batch_size, 1), device=device) * 0.5 - 0.25,
        at=torch.zeros(batch_size, 3),
        up=torch.tensor([[0., 1., 0.]]),
        fov=fovy,
        near=cam_near_far[0], far=cam_near_far[1],
        height=iter_res[0], width=iter_res[1],
        device=device
    )

def get_rotate_camera(itr, fovy = np.deg2rad(45), iter_res=[512,512], cam_near_far=[0.1, 1000.0], cam_radius=3.0, device="cuda"):
    ang = (itr / 10) * np.pi * 2
    camera_pos = torch.stack(kal.ops.coords.spherical2cartesian(torch.tensor(ang), torch.tensor(0.4), -torch.tensor(cam_radius)))
    return kal.render.camera.Camera.from_args(
        eye=camera_pos,
        at=torch.zeros(3),
        up=torch.tensor([0., 1., 0.]),
        fov=fovy,
        near=cam_near_far[0], far=cam_near_far[1],
        height=iter_res[0], width=iter_res[1],
        device=device
    )

def render_mesh(
    glctx,
    mesh, 
    camera, 
    iter_res, 
    return_types = ["mask", "depth"], 
    white_bg=False
):
    vertices_camera = camera.extrinsics.transform(mesh.vertices)
    face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(
        vertices_camera, mesh.faces
    )

    # Projection: nvdiffrast take clip coordinates as input to apply barycentric perspective correction.
    # Using `camera.intrinsics.transform(vertices_camera) would return the normalized device coordinates.
    proj = camera.projection_matrix().unsqueeze(1)
    proj[:, :, 1, 1] = -proj[:, :, 1, 1]
    homogeneous_vecs = kal.render.camera.up_to_homogeneous(
        vertices_camera
    )
    vertices_clip = (proj @ homogeneous_vecs.unsqueeze(-1)).squeeze(-1)
    faces_int = mesh.faces.int()

    rast, _ = dr.rasterize(glctx, vertices_clip, faces_int, iter_res)

    out_dict = {}
    for type in return_types:
        if type == "mask" :
            img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
        elif type == "depth":
            img = dr.interpolate(homogeneous_vecs, rast, faces_int)[0]
        elif type == "normals" :
            img = dr.interpolate(
                mesh.vertex_normals.reshape(len(mesh), -1, 3).contiguous(), rast,
                faces_int
            )[0]
        elif type == "colors" :
            if not mesh.has_attribute('vertex_colors'):
                raise ValueError("Mesh does not have vertex colors.")
            img = dr.interpolate(
                mesh.vertex_colors.reshape(len(mesh), -1, 3).contiguous(), rast,
                faces_int
            )[0]
        if white_bg:
            bg = torch.ones_like(img)
            alpha = (rast[..., -1:] > 0).float() 
            img = torch.lerp(bg, img, alpha)
        out_dict[type] = img

    return out_dict