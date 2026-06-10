import os
import sys
import shutil
import zipfile
import urllib.request
import ssl
import argparse
from pathlib import Path

# Direct download links for Mendeley datasets
MENDELEY_DATASETS = {
    "z6jp232g5j": {
        "url": "https://data.mendeley.com/public-api/zip/z6jp232g5j/download/1",
        "name": "multicrop_mendeley_1.zip"
    },
    "ptz377bwb8": {
        "url": "https://data.mendeley.com/public-api/zip/ptz377bwb8/download/1",
        "name": "potato_mendeley_2.zip"
    }
}

# Kaggle datasets
KAGGLE_DATASETS = [
    "shahadhossin567r7455/potato-leaf-disease-dataset",
    "dedeikhsandwisaputra/rice-leafs-disease-dataset",
    "smaranjitghose/corn-or-maize-leaf-disease-dataset"
]

# The 16 target classes
TARGET_CLASSES = [
    "Cashew__Healthy",
    "Cashew__Leaf_miner",
    "Cashew__Red_rust",
    "Corn__Healthy",
    "Corn__Leaf_blight",
    "Corn__Streak_virus",
    "Potato__Fungi",
    "Potato__Healthy",
    "Potato__Nematode",
    "Rice__Bacterial_leaf_blight",
    "Rice__Brown_spot",
    "Rice__Healthy",
    "Rice__Leaf_blast",
    "Tomato__Healthy",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Verticulium_wilt"
]

def get_target_class(parent_name, filepath):
    """
    Fuzzy match folder and file paths to the 16 target classes.
    """
    p_lower = parent_name.lower()
    path_str = str(filepath).lower()
    
    # ── CASHEW ──
    if "cashew" in p_lower or "cashew" in path_str:
        if "healthy" in p_lower:
            return "Cashew__Healthy"
        elif "miner" in p_lower or "leaf_miner" in p_lower:
            return "Cashew__Leaf_miner"
        elif "rust" in p_lower or "red_rust" in p_lower:
            return "Cashew__Red_rust"
            
    # ── CORN ──
    if "corn" in p_lower or "maize" in p_lower or "corn" in path_str or "maize" in path_str:
        if "healthy" in p_lower:
            return "Corn__Healthy"
        elif "blight" in p_lower or "leaf_blight" in p_lower or "gray" in p_lower:
            # Gray leaf spot and Leaf blight map to Leaf_blight
            return "Corn__Leaf_blight"
        elif "streak" in p_lower or "virus" in p_lower:
            return "Corn__Streak_virus"
        elif "common_rust" in p_lower or "common rust" in p_lower or "rust" in p_lower:
            # In Mendeley, Corn Red Rust is under Corn Streak or Red Rust, let's map appropriately.
            # But the 16 classes only have Corn__Healthy, Corn__Leaf_blight, Corn__Streak_virus.
            # So let's map rust to Corn__Leaf_blight or Streak_virus depending on the actual dataset.
            # Wait, common rust is a fungal disease, we can group it under Corn__Leaf_blight (or Streak_virus).
            # Let's map Corn rust/Common Rust to Corn__Leaf_blight.
            return "Corn__Leaf_blight"

    # ── POTATO ──
    if "potato" in p_lower or "potato" in path_str:
        if "healthy" in p_lower:
            return "Potato__Healthy"
        elif "nematode" in p_lower:
            return "Potato__Nematode"
        elif "fungi" in p_lower or "fungal" in p_lower or "blight" in p_lower or "virus" in p_lower or "pest" in p_lower:
            # Early blight, Late blight, Potato virus, Fungi, Pests all map to Potato__Fungi
            return "Potato__Fungi"
            
    # ── RICE ──
    if "rice" in p_lower or "rice" in path_str:
        if "healthy" in p_lower:
            return "Rice__Healthy"
        elif "brown" in p_lower or "brown_spot" in p_lower or "narrow" in p_lower:
            return "Rice__Brown_spot"
        elif "blast" in p_lower or "leaf_blast" in p_lower:
            return "Rice__Leaf_blast"
        elif "blight" in p_lower or "bacterial" in p_lower or "scald" in p_lower:
            return "Rice__Bacterial_leaf_blight"

    # ── TOMATO ──
    if "tomato" in p_lower or "tomato" in path_str:
        if "healthy" in p_lower:
            return "Tomato__Healthy"
        elif "septoria" in p_lower:
            return "Tomato__Septoria_leaf_spot"
        elif "wilt" in p_lower or "verticillium" in p_lower or "verticulium" in p_lower:
            return "Tomato__Verticulium_wilt"
            
    # Try exact match or direct sub-string match from target classes
    for tc in TARGET_CLASSES:
        if tc.lower() in p_lower or p_lower in tc.lower():
            return tc
            
    return None

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    )
    with urllib.request.urlopen(req, context=ctx) as response, open(output_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print("Download completed.")

def setup_kaggle():
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_config.exists():
        print("\n⚠️  Kaggle API key (kaggle.json) not found in ~/.kaggle/")
        print("Please upload your kaggle.json to the current directory or to Colab files, and it will be configured automatically.")
        if os.path.exists("kaggle.json"):
            os.makedirs(Path.home() / ".kaggle", exist_ok=True)
            shutil.move("kaggle.json", kaggle_config)
            os.chmod(kaggle_config, 0o600)
            print("✅ Configured kaggle.json successfully.")
            return True
        else:
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Download and prepare the leaf disease classification dataset.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output dataset directory path")
    parser.add_argument("--tmp-dir", type=str, default="tmp_download", help="Temporary directory for downloading and unzipping")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tmp_dir = Path(args.tmp_dir)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # 1. Download Mendeley Datasets
    print("\n=== Downloading Mendeley Datasets ===")
    for key, info in MENDELEY_DATASETS.items():
        zip_path = tmp_dir / info["name"]
        if not zip_path.exists():
            try:
                download_file(info["url"], zip_path)
            except Exception as e:
                print(f"❌ Failed to download {info['name']}: {e}")
        else:
            print(f"✓ {info['name']} already exists.")

    # 2. Download Kaggle Datasets
    print("\n=== Downloading Kaggle Datasets ===")
    kaggle_available = setup_kaggle()
    if kaggle_available:
        try:
            import kaggle
            for dataset in KAGGLE_DATASETS:
                dataset_name = dataset.split("/")[-1]
                zip_path = tmp_dir / f"{dataset_name}.zip"
                if not zip_path.exists():
                    print(f"Downloading Kaggle dataset {dataset}...")
                    kaggle.api.dataset_download_files(dataset, path=tmp_dir, unzip=False)
                    # Kaggle api downloads files as the slug name, check and rename if needed
                    downloaded_name = f"{dataset_name}.zip"
                    if not (tmp_dir / downloaded_name).exists():
                        # sometimes it downloads with another name
                        files = list(tmp_dir.glob("*.zip"))
                        # Find the newest zip
                        if files:
                            newest_file = max(files, key=os.path.getctime)
                            if newest_file.name != downloaded_name and "mendeley" not in newest_file.name:
                                shutil.move(newest_file, tmp_dir / downloaded_name)
                    print(f"✓ Downloaded {dataset_name}.zip")
                else:
                    print(f"✓ {dataset_name}.zip already exists.")
        except Exception as e:
            print(f"⚠️  Error downloading via Kaggle API: {e}")
            print("You can download the following Kaggle datasets manually and place them in the 'tmp_download' folder:")
            for dataset in KAGGLE_DATASETS:
                print(f"  - https://www.kaggle.com/datasets/{dataset}")
    else:
        print("⚠️  Skipping Kaggle API downloads. Please upload Kaggle zip files manually to the 'tmp_download' folder if needed.")

    # 3. Extract all zip files
    print("\n=== Extracting Datasets ===")
    extract_dir = tmp_dir / "extracted"
    os.makedirs(extract_dir, exist_ok=True)

    for zip_file in tmp_dir.glob("*.zip"):
        target_extract = extract_dir / zip_file.stem
        if not target_extract.exists():
            print(f"Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(target_extract)
        else:
            print(f"✓ {zip_file.name} already extracted.")

    # 4. Process and Merge into 16 Target Classes
    print("\n=== Processing and Organizing Images ===")
    
    # Initialize count dictionary
    class_counts = {tc: 0 for tc in TARGET_CLASSES}
    unmapped_counts = {}
    
    # Create target directories
    for tc in TARGET_CLASSES:
        os.makedirs(output_dir / tc, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Traverse extracted files
    print("Scanning extracted files...")
    all_files = list(extract_dir.rglob("*"))
    for file_path in all_files:
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # Exclude folders starting with "."
            if any(part.startswith(".") for part in file_path.parts):
                continue
                
            parent_name = file_path.parent.name
            target_class = get_target_class(parent_name, file_path)
            
            if target_class:
                # Target path in output_dir
                # Use unique name to prevent collisions across datasets
                unique_name = f"{file_path.parent.parent.name}_{file_path.parent.name}_{file_path.name}"
                # Clean name
                unique_name = "".join([c if c.isalnum() or c in (".", "_", "-") else "_" for c in unique_name])
                dest_path = output_dir / target_class / unique_name
                
                if not dest_path.exists():
                    shutil.copy2(file_path, dest_path)
                class_counts[target_class] += 1
            else:
                unmapped_counts[parent_name] = unmapped_counts.get(parent_name, 0) + 1

    print("\n=== Dataset Organization Summary ===")
    print(f"Dataset compiled at: {output_dir.resolve()}")
    total_mapped = 0
    for tc in TARGET_CLASSES:
        print(f"  {tc}: {class_counts[tc]} images")
        total_mapped += class_counts[tc]
    
    print(f"\nTotal successfully organized images: {total_mapped}")
    
    if unmapped_counts:
        print(f"\nIgnored folders / classes (not mapping to the 16 target classes):")
        for folder, count in sorted(unmapped_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {folder}: {count} images")

if __name__ == "__main__":
    main()
