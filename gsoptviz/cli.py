import argparse
import json

from .sequence import Sequence


def main():
    p = argparse.ArgumentParser(description="Independent 3DGS + current/GT camera optimization player")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("inspect", "serve", "export", "video"):
        cmd = sub.add_parser(name)
        cmd.add_argument("manifest")
        if name != "inspect":
            cmd.add_argument("--max-gaussians", type=int, default=0, help="0 shows every Gaussian; positive value enables preview subsampling")
        if name == "serve":
            cmd.add_argument("--host", default="127.0.0.1")
            cmd.add_argument("--port", type=int, default=8080)
        elif name in ("export", "video"):
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--fps", type=float, default=3.)
            if name == "video":
                cmd.add_argument("--browser", help="Optional Chromium executable path")
                cmd.add_argument("--view", choices=("overview", "camera"), default="overview")
                cmd.add_argument("--camera-id", help="Camera to highlight; default is the one that moved most")
    a = p.parse_args()
    seq = Sequence(a.manifest)
    if a.command == "inspect":
        for index in range(len(seq.frames)):
            seq.splats(index)
        print(json.dumps(seq.summary(), indent=2))
    elif a.command == "serve":
        from .viewer import Player
        Player(seq, a.host, a.port, a.max_gaussians).run()
    elif a.command == "export":
        from .viewer import export_html
        print(json.dumps(export_html(seq, a.output, a.fps, a.max_gaussians), indent=2))
    else:
        from .video import export_video
        print(json.dumps(export_video(seq, a.output, a.fps, a.max_gaussians, a.browser, view=a.view, camera_id=a.camera_id), indent=2))


if __name__ == "__main__":
    main()
