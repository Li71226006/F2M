from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            return None
        return trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in scene_or_mesh.geometry.values())
        )
    return scene_or_mesh


def parse_origin(element):
    origin = element.find("origin")
    xyz = np.zeros(3)
    rotation = np.eye(3)
    if origin is not None:
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        rotation = R.from_euler("xyz", rpy).as_matrix()
    return xyz, rotation


def apply_transform(mesh, translation, rotation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    mesh.apply_transform(transform)
    return mesh


def create_primitive_mesh(geometry, translation, rotation):
    if geometry.tag.endswith("box"):
        mesh = trimesh.creation.box(extents=np.fromstring(geometry.attrib["size"], sep=" "))
    elif geometry.tag.endswith("sphere"):
        mesh = trimesh.creation.icosphere(radius=float(geometry.attrib["radius"]))
    elif geometry.tag.endswith("cylinder"):
        mesh = trimesh.creation.cylinder(radius=float(geometry.attrib["radius"]), height=float(geometry.attrib["length"]))
    else:
        raise ValueError(f"Unsupported geometry type: {geometry.tag}")
    return apply_transform(mesh, translation, rotation)


def load_link_geometries(urdf_path: str, link_names: list[str], collision: bool = False) -> dict[str, trimesh.Trimesh]:
    urdf_dir = os.path.dirname(urdf_path)
    root = ET.parse(urdf_path).getroot()
    link_geometries = {}

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        if link_name not in link_names:
            continue
        geom_index = "collision" if collision else "visual"
        link_meshes = []
        for visual in link.findall(".//" + geom_index):
            geometry = visual.find("geometry")
            if geometry is None or len(geometry) == 0:
                continue
            xyz, rotation = parse_origin(visual)
            try:
                if geometry[0].tag.endswith("mesh"):
                    mesh_filename = geometry[0].attrib["filename"]
                    full_mesh_path = os.path.join(urdf_dir, mesh_filename)
                    mesh = as_mesh(trimesh.load(full_mesh_path, force="mesh"))
                    scale = np.fromstring(geometry[0].attrib.get("scale", "1 1 1"), sep=" ")
                    mesh.apply_scale(scale)
                    link_meshes.append(apply_transform(mesh, xyz, rotation))
                else:
                    mesh = create_primitive_mesh(geometry[0], xyz, rotation)
                    scale = np.fromstring(geometry[0].attrib.get("scale", "1 1 1"), sep=" ")
                    mesh.apply_scale(scale)
                    link_meshes.append(mesh)
            except Exception as exc:
                print(f"Failed to load geometry for {link_name}: {exc}")
        if len(link_meshes) == 1:
            link_geometries[link_name] = link_meshes[0]
        elif len(link_meshes) > 1:
            link_geometries[link_name] = as_mesh(trimesh.Scene(link_meshes))
    return link_geometries
