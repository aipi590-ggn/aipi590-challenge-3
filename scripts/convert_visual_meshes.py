"""Convert Fetch robot visual meshes from DAE (Collada) to GLB for the web viewer.

Source: ZebraDevs/fetch_ros fetch_description package
https://github.com/ZebraDevs/fetch_ros

Requires: pip install trimesh pycollada
"""

import os
import subprocess
import tempfile
import trimesh

REPO_URL = "https://github.com/ZebraDevs/fetch_ros.git"

LINKS = [
    "base_link",
    "torso_lift_link",
    "head_pan_link",
    "head_tilt_link",
    "shoulder_pan_link",
    "shoulder_lift_link",
    "upperarm_roll_link",
    "elbow_flex_link",
    "forearm_roll_link",
    "wrist_flex_link",
    "wrist_roll_link",
    "gripper_link",
]


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "docs", "meshes", "fetch", "visual")
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Cloning {REPO_URL}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, tmp],
            check=True,
            capture_output=True,
        )

        mesh_dir = os.path.join(tmp, "fetch_description", "meshes")

        for name in LINKS:
            dae_path = os.path.join(mesh_dir, f"{name}.dae")
            glb_path = os.path.join(out_dir, f"{name}.glb")

            scene = trimesh.load(dae_path)
            if hasattr(scene, "geometry") and len(scene.geometry) > 0:
                combined = trimesh.util.concatenate(list(scene.geometry.values()))
            else:
                combined = scene

            combined.export(glb_path, file_type="glb")
            verts = len(combined.vertices)
            size_kb = os.path.getsize(glb_path) // 1024
            print(f"  {name}: {verts} verts, {size_kb}KB")

    print(f"\nWrote {len(LINKS)} GLB files to {out_dir}")


if __name__ == "__main__":
    main()
