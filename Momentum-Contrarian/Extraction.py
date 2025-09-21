import os
import shutil
import zipfile
import tarfile
import os
from pathlib import Path
import zstandard as zstd



def decompress_zst_folder(remove_suffix: bool = True):

    input_dir = Path(r"") #Change File name to DBN Folder
    output_dir = Path(r"") #Change File name to output csv
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in input_dir.glob("*.zst"):
        out_name = file.stem if remove_suffix else file.name
        out_path = output_dir / out_name

        try:
            with open(file, "rb") as f_in, open(out_path, "wb") as f_out:
                dctx = zstd.ZstdDecompressor()
                dctx.copy_stream(f_in, f_out)
            print(f"[OK] Decompressed {file.name} → {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to decompress {file.name}: {e}")

decompress_zst_folder()


