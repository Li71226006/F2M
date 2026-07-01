from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
MODES = {
    "5f_full": [1, 1, 1, 1, 1],
    "4f_no_little": [1, 1, 1, 1, 0],
    "3f_thumb_index_middle": [1, 1, 1, 0, 0],
    "2f_thumb_index": [1, 1, 0, 0, 0],
}
MODE_TITLES = {
    "5f_full": "5F full hand",
    "4f_no_little": "4F no little",
    "3f_thumb_index_middle": "3F thumb/index/middle",
    "2f_thumb_index": "2F thumb/index",
}

LINK_COLORS = {
    # Muted, paper-friendly colors: distinct parts without a saturated demo look.
    "object": np.array([96, 128, 158, 180], dtype=np.uint8),
    "thumb": np.array([168, 92, 78, 255], dtype=np.uint8),
    "index": np.array([76, 118, 158, 255], dtype=np.uint8),
    "middle": np.array([86, 140, 128, 255], dtype=np.uint8),
    "ring": np.array([164, 126, 74, 255], dtype=np.uint8),
    "little": np.array([124, 104, 148, 255], dtype=np.uint8),
    "palm": np.array([184, 190, 198, 255], dtype=np.uint8),
    "forearm": np.array([145, 151, 160, 255], dtype=np.uint8),
    "inactive": np.array([132, 138, 146, 105], dtype=np.uint8),
}


@dataclass
class MujocoRenderer:
    """F2M local HTML/PLY/GLB visualization."""

    result_root: str
    title: str = "F2M Grasp Viewer"

    def make_index(self) -> None:
        root = self._root()
        cards = []
        for stats_path in sorted(root.glob("*/stats.json")):
            mode = stats_path.parent.name
            stats = self._load_json(stats_path)
            near = stats.get("after", {}).get("near_ratio")
            pen = stats.get("after", {}).get("max_penetration_mm")
            residual = stats.get("final_residual_mass")
            thumb_file = stats_path.parent / f"{mode}.png"
            if thumb_file.exists():
                thumb = f'<img src="{html.escape(mode)}/{html.escape(mode)}.png" alt="{html.escape(mode)} preview" />'
            else:
                thumb = '<div class="no-preview">no preview image</div>'
            cards.append(
                f"""<article>
  <a class="mode-card" href="interactive_viewer.html?mode={html.escape(mode)}">
    {thumb}
    <h2>{html.escape(MODE_TITLES.get(mode, mode))}</h2>
    <dl>
      <div><dt>near ratio</dt><dd>{self._fmt(near)}</dd></div>
      <div><dt>penetration</dt><dd>{self._fmt(pen)} mm</dd></div>
      <div><dt>residual</dt><dd>{self._fmt(residual)}</dd></div>
    </dl>
  </a>
</article>"""
            )
        body = "\n".join(cards) if cards else "<p>No result files yet.</p>"
        (root / "index.html").write_text(
            self._index_html(self.title, body),
            encoding="utf-8",
        )
        self._write_interactive_viewer(root)

    def export_scene_preview(self, hand, q, object_pc_normals, output_dir: str | Path, *, name: str) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scene = trimesh.Scene()
        hand_mesh = hand.get_trimesh_q(q)["visual"]
        if len(hand_mesh.vertices) > 0:
            scene.add_geometry(hand_mesh, geom_name="hand")
        object_pc = object_pc_normals[:, :3].detach().cpu().numpy()
        colors = np.tile(LINK_COLORS["object"][None, :], (object_pc.shape[0], 1))
        scene.add_geometry(trimesh.points.PointCloud(object_pc, colors=colors), geom_name="object_pc")
        path = output_dir / f"{name}.ply"
        scene.export(path)
        return path

    def export_comparison_html(
        self,
        hand,
        q_before,
        q_after,
        object_pc_normals,
        output_dir: str | Path,
        *,
        stats: dict,
    ) -> Path:
        """Write current mode assets and keep preview.html as a compatible entry."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = output_dir.name
        object_points = object_pc_normals[:, :3].detach().cpu().numpy()
        before_points = self._hand_points(hand, q_before)
        after_points = self._hand_points(hand, q_after)
        self._write_glb(hand, q_after, object_points, output_dir / f"{mode}.glb", mode)

        # Kept for backward compatibility with old reports. The main viewer no
        # longer uses this 2D projection as the primary visualization.
        image = self._render_projection(object_points, before_points, after_points)
        image.save(output_dir / f"{mode}.png")
        image.save(output_dir / f"{mode}.gif", save_all=True, append_images=[image], duration=700, loop=0)
        (output_dir / f"{mode}_stats.json").write_text(self._json(stats), encoding="utf-8")

        path = output_dir / "preview.html"
        path.write_text(
            f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(self.title)} - {html.escape(mode)}</title>
<meta http-equiv="refresh" content="0; url=../interactive_viewer.html?mode={html.escape(mode)}">
<a href="../interactive_viewer.html?mode={html.escape(mode)}">open interactive viewer</a>
""",
            encoding="utf-8",
        )
        return path

    def _root(self) -> Path:
        root = Path(self.result_root)
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[1] / root
        root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _load_json(path: Path) -> dict:
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _json(payload: dict) -> str:
        import json

        return json.dumps(payload, indent=2, ensure_ascii=False)

    @staticmethod
    def _fmt(value) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        return "n/a"

    @staticmethod
    def _hand_points(hand, q) -> np.ndarray:
        robot_pc_dict, _ = hand.get_transformed_links_pc(q)
        if not robot_pc_dict:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate([pc.detach().cpu().numpy() for pc in robot_pc_dict.values()], axis=0)

    def _write_glb(self, hand, q_after, object_points: np.ndarray, path: Path, mode: str) -> None:
        scene = trimesh.Scene()
        hand_meshes = hand.get_trimesh_q(q_after)
        for link_name, part_mesh in hand_meshes["parts"].items():
            if not isinstance(part_mesh, trimesh.Trimesh) or len(part_mesh.vertices) == 0:
                continue
            part_mesh = part_mesh.copy()
            color = self._link_color(link_name, mode)
            part_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=part_mesh,
                face_colors=np.tile(color[None, :], (len(part_mesh.faces), 1)),
            )
            scene.add_geometry(part_mesh, node_name=link_name, geom_name=link_name)

        object_cloud = trimesh.points.PointCloud(
            object_points,
            colors=np.tile(LINK_COLORS["object"][None, :], (object_points.shape[0], 1)),
        )
        try:
            object_mesh = object_cloud.convex_hull
            object_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=object_mesh,
                face_colors=np.tile(LINK_COLORS["object"][None, :], (len(object_mesh.faces), 1)),
            )
            scene.add_geometry(object_mesh, geom_name="object_hull")
        except Exception:
            scene.add_geometry(object_cloud, geom_name="object_pc")
        path.write_bytes(scene.export(file_type="glb"))

    @staticmethod
    def _render_projection(object_points: np.ndarray, before_points: np.ndarray, after_points: np.ndarray) -> Image.Image:
        width, height = 960, 640
        canvas = Image.new("RGB", (width, height), "#f8fafc")
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 18), "legacy preview only: steel object, gray before, muted after", fill="#1f2933")
        if not (len(object_points) and len(before_points) and len(after_points)):
            return canvas

        points = np.concatenate([object_points[:, :2], before_points[:, :2], after_points[:, :2]], axis=0)
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)

        def project(arr: np.ndarray) -> np.ndarray:
            xy = (arr[:, :2] - mins[None, :]) / span[None, :]
            out = np.empty_like(xy)
            out[:, 0] = 44 + xy[:, 0] * (width - 88)
            out[:, 1] = height - 44 - xy[:, 1] * (height - 108)
            return out

        obj_xy = project(object_points[:: max(1, len(object_points) // 500)])
        before_xy = project(before_points[:: max(1, len(before_points) // 900)])
        after_xy = project(after_points[:: max(1, len(after_points) // 900)])
        for x, y in obj_xy:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#607f9e")
        for x, y in before_xy:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="#8b949e")
        for x, y in after_xy:
            draw.ellipse((x - 1.4, y - 1.4, x + 1.4, y + 1.4), fill="#a85c4e")
        return canvas

    def _write_interactive_viewer(self, root: Path) -> None:
        modes = []
        buttons = []
        panels = []
        for stats_path in sorted(root.glob("*/stats.json")):
            mode = stats_path.parent.name
            if not (stats_path.parent / f"{mode}.glb").exists():
                continue
            modes.append(mode)
            stats = self._load_json(stats_path)
            active = len(modes) == 1
            buttons.append(
                f'<button class="{"active" if active else ""}" data-mode="{html.escape(mode)}">'
                f"{html.escape(MODE_TITLES.get(mode, mode))}</button>"
            )
            panels.append(self._viewer_panel(mode, stats, active))

        html_text = self._viewer_html(
            self.title,
            "".join(buttons) if buttons else "",
            "\n".join(panels) if panels else '<p class="empty">No rendered modes found.</p>',
        )
        (root / "interactive_viewer.html").write_text(html_text, encoding="utf-8")

    def _viewer_panel(self, mode: str, stats: dict, active: bool) -> str:
        after = stats.get("after", {}) or {}
        near = after.get("active_near_ratio", after.get("near_ratio"))
        disabled = after.get("disabled_near_ratio")
        pen = after.get("max_penetration_mm")
        residual = stats.get("final_residual_mass")
        stats_json = html.escape(self._json(stats))
        src = f"{html.escape(mode)}/{html.escape(mode)}"
        return f"""<section class="panel {'active' if active else ''}" data-mode="{html.escape(mode)}">
  <model-viewer src="{src}.glb" camera-controls auto-rotate shadow-intensity="0.65"
    exposure="1.05" tone-mapping="neutral" camera-orbit="115deg 72deg 0.28m"
    field-of-view="35deg" interaction-prompt="none"></model-viewer>
  <img class="viewer-fallback" src="{src}.png" alt="{html.escape(mode)} preview" />
  <aside>
    <img src="{src}.png" alt="{html.escape(mode)} still" />
    <dl>
      <div><dt>active near</dt><dd>{self._fmt(near)}</dd></div>
      <div><dt>disabled near</dt><dd>{self._fmt(disabled)}</dd></div>
      <div><dt>max penetration</dt><dd>{self._fmt(pen)} mm</dd></div>
      <div><dt>residual</dt><dd>{self._fmt(residual)}</dd></div>
    </dl>
    <a href="{src}.gif">GIF</a>
    <pre>{stats_json}</pre>
  </aside>
</section>"""

    def _viewer_payload(self, mode: str, stats: dict) -> dict:
        return {
            "object": stats.get("object"),
            "mode": mode,
            "title": MODE_TITLES.get(mode, mode),
            "short": self._short_mode_title(mode),
            "fingers": MODES.get(mode, []),
            "before": stats.get("before"),
            "after": stats.get("after"),
            "disabled_mass": stats.get("disabled_mass"),
            "start_residual_mass": stats.get("start_residual_mass"),
            "final_residual_mass": stats.get("final_residual_mass"),
            "sequence_summary": {
                "finger_order": stats.get("sequence", {}).get("finger_order"),
                "final_residual_mass": stats.get("sequence", {}).get("final_residual_mass"),
            },
        }

    @staticmethod
    def _short_mode_title(mode: str) -> str:
        if mode == "5f_full":
            return "5F full"
        if mode == "4f_no_little":
            return "4F no little"
        if mode.startswith("3f"):
            return "3F"
        if mode.startswith("2f"):
            return "2F"
        return mode

    @staticmethod
    def _link_color(link_name: str, mode: str) -> np.ndarray:
        stem = link_name.lower()
        compact = stem.replace("right_", "").replace("left_", "")
        compact = compact.replace("rh_", "").replace("lh_", "")
        compact = compact.replace("rh", "", 1) if compact.startswith("rh") else compact
        compact = compact.replace("lh", "", 1) if compact.startswith("lh") else compact
        mask = MODES.get(mode, [1, 1, 1, 1, 1])
        active = dict(zip(FINGER_NAMES, mask))
        if "forearm" in stem:
            return LINK_COLORS["forearm"]
        if compact.startswith("th"):
            return LINK_COLORS["thumb"] if active.get("thumb") else LINK_COLORS["inactive"]
        if compact.startswith("ff"):
            return LINK_COLORS["index"] if active.get("index") else LINK_COLORS["inactive"]
        if compact.startswith("mf"):
            return LINK_COLORS["middle"] if active.get("middle") else LINK_COLORS["inactive"]
        if compact.startswith("rf"):
            return LINK_COLORS["ring"] if active.get("ring") else LINK_COLORS["inactive"]
        if compact.startswith("lf"):
            return LINK_COLORS["little"] if active.get("little") else LINK_COLORS["inactive"]
        if "thumb" in stem:
            return LINK_COLORS["thumb"] if active.get("thumb") else LINK_COLORS["inactive"]
        if "index_finger" in stem or "index-finger" in stem:
            return LINK_COLORS["index"] if active.get("index") else LINK_COLORS["inactive"]
        if "middle_finger" in stem or "middle-finger" in stem:
            return LINK_COLORS["middle"] if active.get("middle") else LINK_COLORS["inactive"]
        if "ring_finger" in stem or "ring-finger" in stem:
            return LINK_COLORS["ring"] if active.get("ring") else LINK_COLORS["inactive"]
        if "little_finger" in stem or "little-finger" in stem:
            return LINK_COLORS["little"] if active.get("little") else LINK_COLORS["inactive"]
        if "palm" in stem or "wrist" in stem or "base" in stem:
            return LINK_COLORS["palm"]
        return LINK_COLORS["inactive"]

    @staticmethod
    def _finger_badges(mode: str) -> str:
        mask = MODES.get(mode, [1, 1, 1, 1, 1])
        fingers = ["thumb", "index", "middle", "ring", "little"]
        return "".join(
            f'<span class="finger-dot {finger} {"active" if mask[idx] else ""}"></span>'
            for idx, finger in enumerate(fingers)
        )

    @staticmethod
    def _index_html(title: str, body: str) -> str:
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ margin: 0; font-family: Inter, "Microsoft YaHei", Segoe UI, Arial, sans-serif; background: #edf2f7; color: #1f2933; }}
    header {{ min-height: 58px; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 0 18px; background: #f8fafc; border-bottom: 1px solid #d8e0e8; }}
    h1 {{ margin: 0; font-size: 18px; line-height: 1.2; }}
    .open {{ color: #1f6feb; font-weight: 650; text-decoration: none; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; padding: 18px; }}
    article {{ background: #ffffff; border: 1px solid #d8e0e8; overflow: hidden; }}
    .mode-card {{ display: block; color: inherit; text-decoration: none; }}
    img {{ display: block; width: 100%; height: 230px; object-fit: contain; background: linear-gradient(#e9f1f8, #ffffff); border-bottom: 1px solid #d8e0e8; }}
    .no-preview {{ display: grid; place-items: center; height: 230px; color: #667280; background: linear-gradient(#e9f1f8, #ffffff); border-bottom: 1px solid #d8e0e8; }}
    h2 {{ margin: 14px 14px 12px; font-size: 17px; }}
    dl {{ display: grid; gap: 8px; margin: 0 14px 16px; }}
    dt {{ font-size: 12px; color: #667280; }}
    dd {{ margin: 2px 0 0; font-size: 15px; font-weight: 650; }}
  </style>
</head>
<body>
  <header><h1>{html.escape(title)}</h1><a class="open" href="interactive_viewer.html">open viewer</a></header>
  <main>{body}</main>
</body>
</html>"""

    @staticmethod
    def _viewer_html(title: str, buttons: str, panels: str) -> str:
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    body {{ margin: 0; font-family: Inter, "Microsoft YaHei", Segoe UI, Arial, sans-serif; background: #edf2f7; color: #1f2933; }}
    header {{ min-height: 58px; display: flex; align-items: center; gap: 10px; padding: 0 18px; background: #f8fafc; border-bottom: 1px solid #d8e0e8; }}
    h1 {{ font-size: 18px; margin: 0 16px 0 0; white-space: nowrap; }}
    button {{ border: 1px solid #cbd5df; background: #fff; border-radius: 6px; padding: 8px 11px; cursor: pointer; }}
    button.active {{ background: #1f6feb; color: white; border-color: #1f6feb; }}
    main {{ height: calc(100vh - 58px); }}
    .panel {{ display: none; height: 100%; grid-template-columns: 1fr 330px; }}
    .panel.active {{ display: grid; }}
    model-viewer {{ width: 100%; height: 100%; background: linear-gradient(#e9f1f8, #ffffff); }}
    .viewer-fallback {{ display: none; width: 100%; height: 100%; object-fit: contain; background: linear-gradient(#e9f1f8, #ffffff); }}
    .no-model-viewer model-viewer {{ display: none; }}
    .no-model-viewer .viewer-fallback {{ display: block; }}
    aside {{ padding: 16px; background: #f8fafc; border-left: 1px solid #d8e0e8; overflow: auto; }}
    img {{ width: 100%; border: 1px solid #d8e0e8; background: white; }}
    dl {{ display: grid; gap: 10px; margin: 14px 0; }}
    dt {{ font-size: 12px; color: #667280; }}
    dd {{ margin: 2px 0 0; font-size: 16px; font-weight: 650; }}
    pre {{ font-size: 11px; white-space: pre-wrap; color: #55606d; }}
    a {{ color: #1f6feb; font-weight: 650; text-decoration: none; }}
    .empty {{ padding: 24px; }}
    @media (max-width: 820px) {{
      header {{ align-items: flex-start; flex-wrap: wrap; padding: 12px; }}
      main {{ height: auto; min-height: calc(100vh - 82px); }}
      .panel.active {{ display: block; }}
      model-viewer, .viewer-fallback {{ height: 62vh; min-height: 360px; }}
      aside {{ border-left: 0; border-top: 1px solid #d8e0e8; }}
    }}
  </style>
</head>
<body>
  <header><h1>{html.escape(title)}</h1>{buttons}<a href="index.html">index</a></header>
  <main>{panels}</main>
  <script>
    const buttons = [...document.querySelectorAll('button[data-mode]')];
    const panels = [...document.querySelectorAll('.panel')];
    const queryMode = new URLSearchParams(location.search).get('mode');

    function selectMode(mode) {{
      const selected = panels.some(panel => panel.dataset.mode === mode) ? mode : panels[0]?.dataset.mode;
      buttons.forEach(button => button.classList.toggle('active', button.dataset.mode === selected));
      panels.forEach(panel => panel.classList.toggle('active', panel.dataset.mode === selected));
    }}

    for (const button of buttons) {{
      button.addEventListener('click', () => selectMode(button.dataset.mode));
    }}
    for (const viewer of document.querySelectorAll('model-viewer')) {{
      viewer.addEventListener('error', () => {{
        document.documentElement.classList.add('no-model-viewer');
      }});
    }}
    setTimeout(() => {{
      if (!customElements.get('model-viewer')) {{
        document.documentElement.classList.add('no-model-viewer');
      }}
    }}, 1500);
    if (panels.length) selectMode(queryMode);
  </script>
</body>
</html>"""
