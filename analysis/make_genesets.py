"""Builds genesets.json (name -> HGNC symbols). Aliases for renamed genes are included
(e.g. WARS1/WARS, CAVIN2/SDPR) so intersection with var_names works for either GENCODE version.
Sources are recorded in genesets_sources.json. Run: python make_genesets.py"""
import json, os

G = {}
S = {}

def add(name, genes, src):
    seen = []
    for g in genes:
        if g not in seen:
            seen.append(g)
    G[name] = seen
    S[name] = src

# ---- interferon ----
add("ISG_typeI", ["IFI27", "IFI44L", "IFI44", "IFIT1", "IFIT2", "IFIT3", "IFI6", "ISG15", "MX1", "MX2",
                  "OAS1", "OAS2", "OAS3", "OASL", "RSAD2", "HERC5", "HERC6", "EPSTI1", "LY6E", "XAF1",
                  "CMPK2", "USP18", "SIGLEC1", "SPATS2L", "EIF2AK2", "LAMP3", "SAMD9L", "PLSCR1", "IFITM3"],
    "Chiche 2014 Arthritis Rheumatol 10.1002/art.38628 (modules M1.2/M3.4/M5.12); Banchereau 2016 Cell "
    "10.1016/j.cell.2016.03.008; Perez 2022 Science 10.1126/science.abf1970")
add("ISG_typeI_core_specific", ["IFI27", "IFI44L", "RSAD2", "IFIT1", "IFIT3", "ISG15", "MX1", "OAS1",
                                "OAS3", "HERC5", "SIGLEC1", "LY6E", "CMPK2", "USP18"],
    "Genes strongly induced by IFN-alpha/beta and weakly by IFN-gamma (Chiche 2014 M1.2; Rose 2013 ARD "
    "10.1136/annrheumdis-2012-201586 for SIGLEC1). Use as the ISG covariate in 'beyond-ISG' models.")
add("IFN_gamma_specific", ["CXCL9", "IDO1", "GBP1", "GBP4", "GBP5", "CIITA", "WARS1", "WARS", "IRF1",
                           "UBD", "SOCS1", "CD274", "FCGR1A", "APOL6", "SECTM1", "IL32", "BATF2", "STAT1"],
    "IFN-gamma-dominant genes (GBP cluster, CIITA, IDO1, CXCL9, WARS1) per Hallmark IFNG response "
    "(Liberzon 2015 Cell Syst 10.1016/j.cels.2015.12.004) minus Hallmark IFNA genes; SLE type II IFN: "
    "Munroe 2016 ARD 10.1136/annrheumdis-2015-208140; Chiche 2014 module M5.12")

# ---- treatment ----
add("glucocorticoid_core", ["TSC22D3", "FKBP5", "ZBTB16", "DDIT4", "PER1", "KLF9", "IRS2", "TXNIP",
                            "SESN1", "CXCR4", "SLA", "DUSP1", "CD163", "IL1R2", "VSIG4", "ADORA3"],
    "Canonical GR targets (GILZ=TSC22D3, FKBP5, ZBTB16, DDIT4, PER1, KLF9) used as in vivo GC "
    "pharmacodynamic markers in blood; monocyte-specific GC genes from Ehrchen 2007 Blood "
    "10.1182/blood-2006-04-019620")
add("glucocorticoid_lymphoid_noIEG", ["TSC22D3", "FKBP5", "ZBTB16", "DDIT4", "PER1", "KLF9", "IRS2",
                                       "TXNIP", "SESN1", "CXCR4", "SLA"],
    "GC core without DUSP1 (shared with dissociation IEG list) - use this when testing GC vs artifact")
add("GC_induced_monocyte", ["CD163", "VSIG4", "ADORA3", "IL1R2", "MAOA", "MERTK", "FKBP5", "TSC22D3",
                            "ZBTB16", "DDIT4", "FPR1", "CD36", "SLC1A3", "AMPH"],
    "Ehrchen 2007 Blood 10.1182/blood-2006-04-019620 (GC-induced anti-inflammatory monocyte subtype); "
    "check overlap with MS1 set")
add("MS1_immunosuppressive_monocyte", ["S100A8", "S100A9", "S100A12", "RETN", "ALOX5AP", "IL1R2",
                                       "CD163", "VCAN", "PLAC8", "CYP1B1", "SELL", "MCEMP1", "CLU"],
    "Reyes 2020 Nat Med 10.1038/s41591-020-0752-4 (sepsis MS1 monocytes, HLA-DR low)")
add("HLA_classII", ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1", "HLA-DMA",
                    "HLA-DMB", "CD74"], "MHC-II; low in MS1/GC monocytes, up with IFN-gamma")

# ---- artifacts ----
add("dissociation_IEG", ["FOS", "FOSB", "JUN", "JUNB", "JUND", "ATF3", "EGR1", "EGR2", "EGR3", "IER2",
                         "IER3", "IER5", "NR4A1", "NR4A2", "ZFP36", "ZFP36L1", "ZFP36L2", "KLF2", "KLF4",
                         "KLF6", "DUSP1", "PPP1R15A", "BTG2", "GADD45B", "CEBPD", "SOCS3", "MYC", "RHOB",
                         "CCN1", "CYR61", "SRF", "UBC", "CCNL1", "NFKBIA"],
    "van den Brink 2017 Nat Methods 10.1038/nmeth.4437 (dissociation-induced genes); "
    "O'Flanagan 2019 Genome Biol 10.1186/s13059-019-1830-0; Denisenko 2020 Genome Biol "
    "10.1186/s13059-020-02048-6")
add("heat_shock", ["HSPA1A", "HSPA1B", "HSPA6", "HSPA8", "HSPH1", "HSP90AA1", "HSP90AB1", "HSPB1",
                   "HSPD1", "HSPE1", "DNAJB1", "DNAJA1", "BAG3", "CHORDC1", "HSPA4L"],
    "van den Brink 2017; Denisenko 2020 (37C enzymatic dissociation / handling stress)")
add("IEG_core_specific", ["FOS", "FOSB", "JUN", "EGR1", "ATF3", "IER2", "NR4A1", "ZFP36", "HSPA1A",
                          "HSPA1B", "DNAJB1"],
    "Minimal artifact core with no GC/NF-kB overlap; used as artifact covariate")

# ---- B lineage ----
add("DN2_ABC_up", ["ITGAX", "TBX21", "FCRL5", "FCRL3", "FCRL2", "FCRL4", "ZEB2", "SOX5", "CD86",
                   "CXCR3", "FGR", "HCK", "LILRB1", "TOX", "ENC1", "MPP6", "EMP3", "SIGLEC10", "CD19",
                   "MS4A1", "AIM2", "TNFRSF13B"],
    "Jenks 2018 Immunity 10.1016/j.immuni.2018.08.015; Wang 2018 Nat Commun 10.1038/s41467-018-03750-7; "
    "Dai 2024 Science 10.1126/science.adf8531 (ZEB2)")
add("DN2_ABC_down", ["CR2", "CXCR5", "CD27", "CD24", "CD38", "FCER2", "IL4R", "TCL1A"],
    "Low in DN2/aN (CD21lo CXCR5- CD27-); Jenks 2018")
add("activated_naive_B", ["IGHD", "TCL1A", "ITGAX", "TBX21", "FCRL5", "CD86", "CD69", "CXCR3",
                          "SOX5", "MPP6", "ZEB2"],
    "aN (IgD+CD27-CD11c+ T-bet+) precursors of DN2: Tipton 2015 Nat Immunol 10.1038/ni.3152; Jenks 2018")
add("plasmablast", ["MZB1", "JCHAIN", "XBP1", "PRDM1", "IRF4", "TNFRSF17", "SDC1", "TXNDC5", "SSR4",
                    "DERL3", "FKBP11", "SEC11C", "CD38", "CD27", "PRDX4", "ELL2", "IGHG1", "IGKC"],
    "Banchereau 2016 Cell (plasmablast module, IFN-independent); Arazi 2019 Nat Immunol "
    "10.1038/s41590-019-0398-x")

# ---- T lineage ----
add("Tph", ["PDCD1", "CXCL13", "MAF", "ICOS", "IL21", "PRDM1", "CCR2", "CXCR6", "TOX", "TOX2",
            "CD200", "TIGIT", "BHLHE40", "SLAMF6"],
    "Rao 2017 Nature 10.1038/nature21411; Bocharnikov 2019 JCI Insight 10.1172/jci.insight.130062; "
    "Law 2024 Nature 10.1038/s41586-024-07627-2")
add("Tfh_circulating", ["CXCR5", "BCL6", "IL21", "ICOS", "PDCD1", "CXCL13", "MAF", "SH2D1A", "TOX2",
                        "ASCL2", "CD200", "POU2AF1", "BTLA"],
    "Crotty 2019 Immunity 10.1016/j.immuni.2019.04.011 (Tfh review)")
add("AHR_JUN_Th22_axis", ["AHR", "AHRR", "CYP1A1", "CYP1B1", "JUN", "IL22", "CCR6", "TIPARP"],
    "Law 2024 Nature 10.1038/s41586-024-07627-2: AHR/JUN restrain CXCL13+ T cells; IFN-I suppresses this "
    "axis (predicts LOW JUN in IFN-high CD4 memory). Contains JUN -> interpret jointly with IEG core")
add("cytotoxic_GZMB_GZMH", ["GZMB", "GZMH", "GZMA", "PRF1", "GNLY", "NKG7", "FGFBP2", "CX3CR1", "KLRD1",
                            "CST7", "CCL5", "ZEB2", "TBX21", "ASSIGN_PLACEHOLDER"],
    "Perez 2022 (CD8 GZMH expansion in SLE); Szabo 2019 Nat Commun 10.1038/s41467-019-12464-3")
add("CD8_GZMK_inflammaging", ["GZMK", "CCL4", "CCL5", "CST7", "EOMES", "CXCR3", "CCR5", "CD74",
                              "HLA-DRA", "DUSP2", "CD27"],
    "Mogilenko 2021 Immunity 10.1016/j.immuni.2020.11.005")
add("exhaustion", ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "CTLA4", "ENTPD1", "CD160", "CD244",
                   "EOMES", "CXCL13", "LAYN", "CD38", "KLRG1"],
    "McKinney 2015 Nature 10.1038/nature14468; Buang 2021 Nat Commun 10.1038/s41467-021-22312-y")
add("Treg_activated", ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "TNFRSF4", "TNFRSF9", "LAYN",
                       "HLA-DRA", "CCR8", "MAGEH1"], "Miyara 2009 Immunity 10.1016/j.immuni.2009.03.019")

# ---- myeloid / other ----
add("NFkB_TNF", ["NFKBIA", "TNFAIP3", "NFKB1", "NFKB2", "RELB", "REL", "TNF", "IL1B", "CCL3", "CCL4",
                 "CXCL8", "NFKBIZ", "PTGS2", "ICAM1", "BIRC3", "TNFAIP2", "CD83", "SOD2", "IER3", "CCL20"],
    "Hallmark TNFA_SIGNALING_VIA_NFKB subset (Liberzon 2015); overlaps IEG (IER3, NFKBIA): use "
    "NFkB_TNF_noIEG for artifact-robust tests")
add("NFkB_TNF_noIEG", ["TNFAIP3", "NFKB1", "NFKB2", "RELB", "TNF", "IL1B", "CCL3", "NFKBIZ", "PTGS2",
                       "ICAM1", "BIRC3", "TNFAIP2", "CD83", "SOD2", "CCL20"], "As above minus IEG overlap")
add("SIGLEC1_IFN_monocyte", ["SIGLEC1", "IFI27", "IFI44L", "ISG15", "RSAD2", "MX1", "LY6E", "IFIT3",
                             "OAS1", "CCL2", "CCL8", "LAMP3", "IFITM3", "TNFSF10", "CD38"],
    "York 2007 Arthritis Rheum 10.1002/art.22482; Rose 2013 ARD 10.1136/annrheumdis-2012-201586; "
    "Perez 2022 (cM ISG state)")
add("DC3_CD14pos_cDC2", ["CD1C", "FCER1A", "CLEC10A", "CD14", "CD163", "S100A8", "S100A9", "VCAN",
                         "LYZ", "CD1D", "ANXA1", "HLA-DQA1"],
    "Villani 2017 Science 10.1126/science.aah4573; Dutertre 2019 Immunity 10.1016/j.immuni.2019.08.008 "
    "(DC3 expanded in SLE blood, correlate with activity)")
add("LDG_immature_neutrophil", ["ELANE", "MPO", "AZU1", "PRTN3", "CTSG", "DEFA4", "DEFA1", "DEFA1B",
                                "DEFA3", "LTF", "LCN2", "CAMP", "MMP8", "MMP9", "CEACAM8", "CEACAM6",
                                "BPI", "OLFM4", "CD177", "ARG1", "MS4A3", "CRISP3"],
    "Denny 2010 J Immunol 10.4049/jimmunol.0902904; Mistry 2019 PNAS 10.1073/pnas.1908576116; "
    "Bennett 2003 J Exp Med 10.1084/jem.20021553")
add("neutrophil_mature", ["FCGR3B", "CXCR2", "CXCR1", "CSF3R", "FPR1", "CMTM2", "PROK2", "MME",
                          "S100A12", "MNDA", "SELL", "AQP9", "G0S2", "NAMPT"],
    "Mature neutrophil markers (Xie 2020 Nat Immunol 10.1038/s41590-020-0736-z)")
add("platelet", ["PPBP", "PF4", "GP9", "ITGA2B", "GNG11", "TUBB1", "CAVIN2", "SDPR", "CLU", "SPARC",
                 "NRGN", "TREML1", "GP1BB", "MYL9", "PTCRA", "CMTM5"],
    "Platelet contamination / platelet-leukocyte aggregates; Hao 2021 Cell 10.1016/j.cell.2021.04.048 "
    "(PBMC 'Platelet' cluster markers)")
add("erythrocyte", ["HBB", "HBA1", "HBA2", "HBD", "ALAS2", "CA1", "SLC4A1"], "RBC/ambient contamination")
add("hypoxia", ["VEGFA", "SLC2A1", "SLC2A3", "PGK1", "LDHA", "BNIP3", "BNIP3L", "ENO1", "ALDOA", "P4HA1",
                "ADM", "NDRG1", "PDK1", "EGLN3", "ANKRD37", "MXI1", "PFKFB3", "PFKFB4", "ERO1A", "ERO1L",
                "HK2", "KDM3A", "DDIT4", "BHLHE40"],
    "Hallmark HYPOXIA subset (Liberzon 2015); ex vivo blood handling (Massoni-Badosa 2020 Genome Biol "
    "10.1186/s13059-020-02032-0)")
add("adaptive_NK", ["KLRC2", "B3GAT1", "IL32", "CD3E", "GZMH", "LILRB1", "KLRC1_neg_marker_absent"],
    "CMV-driven adaptive NK (NKG2C+, FCER1G-low, CD57+): Schlums 2015 Immunity "
    "10.1016/j.immuni.2015.02.008")
add("adaptive_NK_down", ["FCER1G", "KLRC1", "ZBTB16", "SH2D1B", "IL2RB"], "Schlums 2015 (lost in adaptive NK)")

# cell cycle: Tirosh 2016 Science 10.1126/science.aad0501, Seurat cc.genes.updated.2019
add("cell_cycle_S", ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6",
                     "CDCA7", "DTL", "PRIM1", "UHRF1", "CENPU", "MLF1IP", "HELLS", "RFC2", "RPA2", "NASP",
                     "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2",
                     "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1",
                     "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"],
    "Tirosh 2016 Science 10.1126/science.aad0501 (Seurat cc.genes.updated.2019)")
add("cell_cycle_G2M", ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2",
                       "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "PIMREG", "FAM64A", "SMC4",
                       "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1",
                       "KIF20B", "HJURP", "CDCA3", "JPT1", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C",
                       "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA",
                       "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5",
                       "CENPA"], "Tirosh 2016")

rpl = ["RPL3", "RPL4", "RPL5", "RPL6", "RPL7", "RPL7A", "RPL8", "RPL9", "RPL10", "RPL10A", "RPL11",
       "RPL12", "RPL13", "RPL13A", "RPL14", "RPL15", "RPL17", "RPL18", "RPL18A", "RPL19", "RPL21", "RPL22",
       "RPL23", "RPL23A", "RPL24", "RPL26", "RPL27", "RPL27A", "RPL28", "RPL29", "RPL30", "RPL31", "RPL32",
       "RPL34", "RPL35", "RPL35A", "RPL36", "RPL36A", "RPL37", "RPL37A", "RPL38", "RPL39", "RPL41",
       "RPLP0", "RPLP1", "RPLP2"]
rps = ["RPS2", "RPS3", "RPS3A", "RPS4X", "RPS4Y1", "RPS5", "RPS6", "RPS7", "RPS8", "RPS9", "RPS10",
       "RPS11", "RPS12", "RPS13", "RPS14", "RPS15", "RPS15A", "RPS16", "RPS17", "RPS18", "RPS19", "RPS20",
       "RPS21", "RPS23", "RPS24", "RPS25", "RPS26", "RPS27", "RPS27A", "RPS28", "RPS29", "RPSA", "FAU",
       "UBA52"]
add("ribosomal", rpl + rps, "HGNC cytosolic ribosomal protein gene group (RPL*/RPS*); compute as fraction")
add("mitochondrial", ["MT-ND1", "MT-ND2", "MT-ND3", "MT-ND4", "MT-ND4L", "MT-ND5", "MT-ND6", "MT-CO1",
                      "MT-CO2", "MT-CO3", "MT-ATP6", "MT-ATP8", "MT-CYB"],
    "13 mtDNA protein-coding genes; compute percent counts. Osorio & Cai 2021 Bioinformatics "
    "10.1093/bioinformatics/btaa751 (thresholds)")
add("naive_quiescence_T", ["CCR7", "SELL", "LEF1", "TCF7", "IL7R", "SATB1", "BACH2", "KLF2", "S1PR1",
                           "SOX4", "PECAM1", "TXK"],
    "Naive/RTE quiescence; naive CD4 depletion in SLE (Perez 2022)")

# clean placeholders
G["cytotoxic_GZMB_GZMH"] = [g for g in G["cytotoxic_GZMB_GZMH"] if g != "ASSIGN_PLACEHOLDER"]
G["adaptive_NK"] = ["KLRC2", "B3GAT1", "IL32", "GZMH", "LILRB1", "CD52", "CCL5"]

here = os.path.dirname(os.path.abspath(__file__))
json.dump(G, open(os.path.join(here, "genesets.json"), "w"), indent=1)
json.dump(S, open(os.path.join(here, "genesets_sources.json"), "w"), indent=1)
print(len(G), "sets")
