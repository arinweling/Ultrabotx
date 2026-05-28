import genesis as gs
from i4h_asset_helper.assets import get_i4h_local_asset_path
import os

gs.init(backend=gs.gpu, logging_level="info")

local_dir = get_i4h_local_asset_path()
probe_usd_path = os.path.join(local_dir, "Props", "ClariusUltrasoundProbe", "fixture_nomtl.usda")

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.8, 0.8, 0.5),
        camera_lookat=(0.0, 0.0, 0.2),
        camera_fov=40,
        max_FPS=60,
    ),
    show_viewer=True,
)

scene.add_entity(gs.morphs.Plane())

probe = scene.add_entity(
    material=gs.materials.Kinematic(),
    morph=gs.morphs.USD(file=probe_usd_path, scale=100),
    surface=gs.surfaces.Default(color=(0.753, 0.753, 1.0)),
)

scene.build()

while True:
    scene.step()
