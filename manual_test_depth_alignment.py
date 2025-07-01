import json
import tempfile
import time
import webbrowser
from pathlib import Path

import numpy as np

import depth_tools


def main():
    print("Loading the test data")
    depth = np.load("manual_test_depth_dilation/test_depth.npy")
    mask = np.load("manual_test_depth_dilation/test_mask.npy")
    with open("manual_test_depth_dilation/camera_params.json") as f:
        camera_params = json.load(f)
    intrinsics = depth_tools.CameraIntrinsics(
        f_x=camera_params["f_x"],
        f_y=camera_params["f_y"],
        c_x=camera_params["c_x"],
        c_y=camera_params["c_y"],
    )

    print("Applying a shift and scale on the original depth, then restoring.")
    depth[~mask] = 0
    transformed_depth = depth * 6 + 8
    restored_depth, _, _ = depth_tools.align_shift_scale(
        control_mask=None, gt_values=depth, mask=mask, pred_values=transformed_depth
    )

    print("Verify that the aligned depth and the original depth are the same.")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        html_path = td / "plot.html"
        fig = depth_tools.depths_2_plotly_fig(
            depth_maps=[
                {
                    "color": "blue",
                    "depth_map": depth,
                    "depth_mask": mask,
                    "name": "Original",
                    "size": 1,
                },
                {
                    "color": "red",
                    "depth_map": restored_depth,
                    "depth_mask": mask,
                    "name": f"Restored",
                    "size": 1,
                },
            ],
            coord_sys=depth_tools.CoordSys.LH_YUp,
            subsample={"max_num": 20_000},
            intrinsics=intrinsics,
        )
        fig.write_html(html_path)
        webbrowser.open(str(html_path))
        time.sleep(3)


if __name__ == "__main__":
    main()
