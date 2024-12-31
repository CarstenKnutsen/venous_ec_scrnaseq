" Goal: Cleanup source data for dendritic cell scRNAseq
Date:24120
Author:Carsten Knutsen
conda_env:soupxR
"
#install.packages('SoupX')
#install.packages('installr')
#library(installr)
#install.Rtools()
#install.packages('Matrix')
#BiocManager::install("DropletUtils")
library(DropletUtils)
library(Matrix)
library(SoupX)
library(Seurat)
library(ggplot2)
graphics.off() 
par("mar") 
par(mar=c(1,1,1,1))
getwd()
data_dir <- 'data/single_cell_files/cellranger_output'
sub_fols <- list.dirs(path = data_dir, full.names = FALSE, recursive = FALSE)
output_dir <-'data/single_cell_files/soupx'
dir.create(output_dir, recursive = TRUE)
figs_dir <- 'data/figures/soupx'
dir.create(figs_dir, recursive = TRUE)
for (fol in sub_fols)
{
  print(fol)
  subdir <- sprintf('%s/%s/outs',data_dir,fol)
  subdir_out <- sprintf('%s/%s',output_dir,fol)
  cellnames <- read.csv(sprintf('%s/filtered_feature_bc_matrix/barcodes.tsv.gz', subdir),header =FALSE)
  filt.matrix <- Read10X_h5(sprintf("%s/filtered_feature_bc_matrix.h5",subdir),use.names = F)
  raw.matrix <- Read10X_h5(sprintf("%s/raw_feature_bc_matrix.h5",subdir),use.names = F)
  soup.channel = SoupChannel(raw.matrix, filt.matrix)
  srat <- CreateSeuratObject(counts = filt.matrix)
  srat <-RenameCells(srat, new.names = cellnames$V1)
  srat    <- SCTransform(srat, verbose = F)
  srat    <- RunPCA(srat, verbose = F)
  srat    <- RunUMAP(srat, dims = 1:30, verbose = F)
  srat    <- FindNeighbors(srat, dims = 1:30, verbose = F)
  srat    <- FindClusters(srat, verbose = T)
  meta    <- srat@meta.data
  umap    <- srat@reductions$umap@cell.embeddings
  png(sprintf("%s/%s.png",figs_dir,fol),width = 5, height = 4, units = 'in',res=300)
  soup.channel  <- setClusters(soup.channel, setNames(meta$seurat_clusters, rownames(meta)))
  soup.channel  <- setDR(soup.channel, umap)
  soup.channel  <- autoEstCont(soup.channel,forceAccept=TRUE)
  gg <- plotMarkerDistribution(soup.channel)
  ggsave(sprintf("%s/%s_marker_distribution.png",figs_dir,fol),
         gg)
  adj.matrix  <- adjustCounts(soup.channel, roundToInt = T)
  DropletUtils:::write10xCounts(subdir_out, adj.matrix)
  dev.off()
}

