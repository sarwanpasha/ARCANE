"""Download the HLCA core dataset (Sikkema et al. 2023) from CELLxGENE Discover."""
import cellxgene_census
DID = "066943a2-fdac-4b29-b348-40cede398e4e"  # An integrated cell atlas of the human lung in health and disease (core)
cellxgene_census.download_source_h5ad(DID, to_path="data/raw/hlca_core.h5ad", census_version="2025-11-08")
print("HLCA_OK", DID)
