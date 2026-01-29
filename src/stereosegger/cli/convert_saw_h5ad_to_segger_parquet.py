import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
import scanpy as sc
import geopandas as gpd
from shapely.geometry import shape


def _read_tif(path: Path) -> np.ndarray:
    try:
        import tifffile
        return tifffile.imread(path)
    except ImportError:
        try:
            import imageio.v3 as iio
            return iio.imread(path)
        except ImportError as exc:
            msg = "Reading TIFF requires `tifffile` or `imageio`."
            raise ImportError(msg) from exc


def _vectorize_labels(labels: np.ndarray) -> gpd.GeoDataFrame:
    try:
        import rasterio.features
    except ImportError:
        raise ImportError("Vectorizing labels requires `rasterio`.") from None

    records = []
    for geom, value in rasterio.features.shapes(labels.astype(np.int32), mask=labels > 0):
        if value == 0:
            continue
        records.append({"boundary_id": int(value), "geometry": shape(geom)})
    gdf = gpd.GeoDataFrame(records, geometry="geometry")
    return gdf


def convert_saw_h5ad_to_parquet(
    h5ad_path: Path,
    out_dir: Path,
    bin_pitch: float = 1.0,
    min_count: int = 1,
    labels_tif: Path | None = None,
    tissue_mask_tif: Path | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    gene_name_source: str = "real_gene_name",
    top_genes: int | None = None,
    max_nnz: int | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)

    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
    elif {"x", "y"}.issubset(adata.obs.columns):
        coords = adata.obs[["x", "y"]].to_numpy()
    else:
        raise ValueError("Coordinates not found. Expected adata.obsm['spatial'] or adata.obs['x','y'].")

    if bbox is not None:
        xmin, xmax, ymin, ymax = bbox
        mask = (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax) & (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)
        adata = adata[mask].copy()
        coords = coords[mask]

    if tissue_mask_tif is not None:
        mask_img = _read_tif(tissue_mask_tif)
        coords_round = np.rint(coords).astype(int)
        valid = (
            (coords_round[:, 0] >= 0)
            & (coords_round[:, 0] < mask_img.shape[1])
            & (coords_round[:, 1] >= 0)
            & (coords_round[:, 1] < mask_img.shape[0])
        )
        mask_vals = np.zeros(coords_round.shape[0], dtype=bool)
        mask_vals[valid] = mask_img[coords_round[valid, 1], coords_round[valid, 0]] > 0
        adata = adata[mask_vals].copy()
        coords = coords[mask_vals]

    if top_genes is not None:
        gene_sums = np.asarray(adata.X.sum(axis=0)).ravel()
        top_idx = np.argsort(gene_sums)[::-1][:top_genes]
        adata = adata[:, top_idx].copy()

    if gene_name_source in adata.var:
        gene_names = adata.var[gene_name_source].astype(str).values
    else:
        gene_names = adata.var_names.astype(str).values

    X = adata.X
    if sp.issparse(X):
        X = X.tocoo()
    else:
        X = sp.coo_matrix(X)

    keep = X.data >= min_count
    rows = X.row[keep]
    cols = X.col[keep]
    vals = X.data[keep]

    if max_nnz is not None and len(vals) > max_nnz:
        rows = rows[:max_nnz]
        cols = cols[:max_nnz]
        vals = vals[:max_nnz]

    x = coords[rows, 0]
    y = coords[rows, 1]
    bx = np.rint(x / bin_pitch).astype(np.int32)
    by = np.rint(y / bin_pitch).astype(np.int32)

    transcripts_df = pd.DataFrame(
        {
            "transcript_id": np.arange(len(vals), dtype=np.int64),
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "bx": bx,
            "by": by,
            "gene_id": cols.astype(np.int32),
            "count": vals.astype(np.int32),
        }
    )

    if labels_tif is not None:
        print("Vectorizing labels and assigning transcripts...")
        labels = _read_tif(labels_tif)
        boundaries_gdf = _vectorize_labels(labels)
        boundaries_gdf.to_parquet(out_dir / "boundaries.parquet", index=False)

        x_idx = np.rint(transcripts_df.x.values).astype(int)
        y_idx = np.rint(transcripts_df.y.values).astype(int)
        valid = (x_idx >= 0) & (x_idx < labels.shape[1]) & (y_idx >= 0) & (y_idx < labels.shape[0])
        assigned_labels = np.zeros(len(transcripts_df), dtype=int)
        assigned_labels[valid] = labels[y_idx[valid], x_idx[valid]]
        
        transcripts_df["overlaps_nucleus"] = (assigned_labels > 0).astype(int)
        transcripts_df["cell_id"] = np.where(assigned_labels > 0, assigned_labels, -1)
    
    transcripts_df.to_parquet(out_dir / "transcripts.parquet", index=False)
    genes = pd.DataFrame({"gene_id": np.arange(len(gene_names), dtype=np.int32), "gene_name": gene_names.astype(str)})
    genes.to_parquet(out_dir / "genes.parquet", index=False)


def main():
    parser = argparse.ArgumentParser(description="Convert Stereo-seq SAW H5AD to Parquet")
    parser.add_argument("--h5ad", type=Path, required=True, help="Path to SAW bin1 h5ad file.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for Segger parquet files.")
    parser.add_argument("--bin_pitch", type=float, default=1.0, help="Bin pitch for rounding to grid coordinates.")
    parser.add_argument("--min_count", type=int, default=1, help="Minimum count to keep a bin-gene entry.")
    parser.add_argument("--labels_tif", type=Path, default=None, help="Optional label TIFF for boundary polygons.")
    parser.add_argument("--tissue_mask_tif", type=Path, default=None, help="Optional tissue mask TIFF.")
    parser.add_argument("--bbox", type=float, nargs=4, default=None, help="Bounding box xmin xmax ymin ymax.")
    parser.add_argument("--gene_name_source", type=str, default="real_gene_name", help="Column in adata.var for gene names.")
    parser.add_argument("--top_genes", type=int, default=None, help="Keep only top K genes by total counts.")
    parser.add_argument("--max_nnz", type=int, default=None, help="Debug: cap number of non-zero entries.")

    args = parser.parse_args()
    convert_saw_h5ad_to_parquet(
        h5ad_path=args.h5ad,
        out_dir=args.out_dir,
        bin_pitch=args.bin_pitch,
        min_count=args.min_count,
        labels_tif=args.labels_tif,
        tissue_mask_tif=args.tissue_mask_tif,
        bbox=args.bbox,
        gene_name_source=args.gene_name_source,
        top_genes=args.top_genes,
        max_nnz=args.max_nnz,
    )


if __name__ == "__main__":
    main()