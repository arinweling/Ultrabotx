import os
from i4h_asset_helper.assets import retrieve_asset

def main():
    print("Downloading ABDPhantom assets...")
    local_dir = retrieve_asset(sub_path="Props/ABDPhantom", verbose=True)
    phantom_path = os.path.join(local_dir, "Props", "ABDPhantom", "phantom.usda")
    print(f"\nDownload complete.")
    print(f"Asset root: {local_dir}")
    print(f"Phantom USD: {phantom_path}")
    if not os.path.exists(phantom_path):
        print("WARNING: phantom.usda not found at expected path — check the asset layout.")

if __name__ == "__main__":
    main()
