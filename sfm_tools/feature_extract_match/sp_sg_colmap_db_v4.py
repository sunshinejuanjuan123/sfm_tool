"""Feature extraction & matching (v4 alias of sp_sg_colmap_db)."""

from sfm_tools.feature_extract_match.sp_sg_colmap_db import ImageMatchingDB

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="feature extraction and matching")
    parser.add_argument("--root_path", help="path to 3dgs format results")
    args = parser.parse_args()
    ImageMatchingDB(args.root_path).run()
