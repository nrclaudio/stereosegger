import click
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

    transcripts = pd.DataFrame(
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
    transcripts.to_parquet(out_dir / "transcripts.parquet", index=False)

    genes = pd.DataFrame(
        {"gene_id": np.arange(len(gene_names), dtype=np.int32), "gene_name": gene_names.astype(str)}
    )
    genes.to_parquet(out_dir / "genes.parquet", index=False)

    if labels_tif is not None:
        labels = _read_tif(labels_tif)
        boundaries = _vectorize_labels(labels)
        boundaries.to_parquet(out_dir / "boundaries.parquet", index=False)


@click.command(name="convert_saw_h5ad_to_segger_parquet")
@click.option("--h5ad", "h5ad_path", type=Path, required=True, help="Path to SAW bin1 h5ad file.")
@click.option("--out_dir", type=Path, required=True, help="Output directory for Segger parquet files.")
@click.option("--bin_pitch", type=float, default=1.0, help="Bin pitch for rounding to grid coordinates.")
@click.option("--min_count", type=int, default=1, help="Minimum count to keep a bin-gene entry.")
@click.option("--labels_tif", type=Path, default=None, help="Optional label TIFF for boundary polygons.")
@click.option("--tissue_mask_tif", type=Path, default=None, help="Optional tissue mask TIFF.")
@click.option("--bbox", type=float, nargs=4, default=None, help="Bounding box xmin xmax ymin ymax.")
@click.option("--gene_name_source", type=str, default="real_gene_name", help="Column in adata.var for gene names.")
@click.option("--top_genes", type=int, default=None, help="Keep only top K genes by total counts.")
@click.option("--max_nnz", type=int, default=None, help="Debug: cap number of non-zero entries.")
def main(
    h5ad_path: Path,
    out_dir: Path,
    bin_pitch: float,
    min_count: int,
    labels_tif: Path | None,
    tissue_mask_tif: Path | None,
    bbox: tuple[float, float, float, float] | None,
    gene_name_source: str,
    top_genes: int | None,
    max_nnz: int | None,
):
    convert_saw_h5ad_to_parquet(
        h5ad_path=h5ad_path,
        out_dir=out_dir,
        bin_pitch=bin_pitch,
        min_count=min_count,
        labels_tif=labels_tif,
        tissue_mask_tif=tissue_mask_tif,
        bbox=bbox,
        gene_name_source=gene_name_source,
        top_genes=top_genes,
        max_nnz=max_nnz,
    )


if __name__ == "__main__":
    main()
