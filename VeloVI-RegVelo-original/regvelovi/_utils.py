from pathlib import Path
from typing import Optional, Union
from urllib.request import urlretrieve
from typing import Iterable, Literal

import anndata
import numpy as np
import pandas as pd
import scvelo as scv
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix

def get_permutation_scores(save_path: Union[str, Path] = Path("data/")) -> pd.DataFrame:
    """Get the reference permutation scores on positive and negative controls.

    Parameters
    ----------
    save_path
        path to save the csv file

    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if not (save_path / "permutation_scores.csv").is_file():
        URL = "https://figshare.com/ndownloader/files/36658185"
        urlretrieve(url=URL, filename=save_path / "permutation_scores.csv")

    return pd.read_csv(save_path / "permutation_scores.csv")


def preprocess_data(
    adata: AnnData,
    spliced_layer: Optional[str] = "Ms",
    unspliced_layer: Optional[str] = "Mu",
    min_max_scale: bool = True,
    filter_on_r2: bool = True,
) -> AnnData:
    """Preprocess data.

    This function removes poorly detected genes and minmax scales the data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    spliced_layer
        Name of the spliced layer.
    unspliced_layer
        Name of the unspliced layer.
    min_max_scale
        Min-max scale spliced and unspliced
    filter_on_r2
        Filter out genes according to linear regression fit

    Returns
    -------
    Preprocessed adata.
    """
    if min_max_scale:
        scaler = MinMaxScaler()
        adata.layers[spliced_layer] = scaler.fit_transform(adata.layers[spliced_layer])

        scaler = MinMaxScaler()
        adata.layers[unspliced_layer] = scaler.fit_transform(
            adata.layers[unspliced_layer]
        )

    if filter_on_r2:
        scv.tl.velocity(adata, mode="deterministic")

        adata = adata[
            :, np.logical_and(adata.var.velocity_r2 > 0, adata.var.velocity_gamma > 0)
        ].copy()
        adata = adata[:, adata.var.velocity_genes].copy()

    return adata

def organize_multiview_anndata(
    rna_anndata,
    atac_anndata: Optional[anndata.AnnData] = None,
    multiview_key: str = "readout",
) -> anndata.AnnData:
    """Concatenate spliced, unspliced and ATAC readout anndata objects.
    mimick orgnize_multiome_anndata function in MultiVI

    Parameters
    ----------

    """
    
    ## create X layers
    spliced_anndata = rna_anndata.copy()
    unspliced_anndata = rna_anndata.copy()

    spliced_anndata.X = rna_anndata.layers["Ms"]
    unspliced_anndata.X = rna_anndata.layers["Mu"]
    spliced_anndata.var['readout'] = ["spliced"] * spliced_anndata.var.shape[0]
    unspliced_anndata.var['readout'] = ["unspliced"] * unspliced_anndata.var.shape[0]

    rna_anndata_merge = anndata.concat([spliced_anndata,unspliced_anndata], axis=1,join='outer')

    ## process side information
    # layers
    for i in list(rna_anndata_merge.layers.keys()): 
        del rna_anndata_merge.layers[i]
    
    # obs
    for key, value in spliced_anndata.obs.items():
        rna_anndata_merge.obs[key] = value

    # uns
    for key, value in spliced_anndata.uns.items():
        rna_anndata_merge.uns[key] = value

    rna_anndata_merge.layers["readout"] = rna_anndata_merge.X.copy()
    rna_anndata_merge.var['id'] = rna_anndata_merge.var.index
    ### add ATAC readout if have
    if atac_anndata is not None:
        var_rna = rna_anndata_merge.var.copy()

        if (atac_anndata.obs.shape[0] == rna_anndata.obs.shape[0]) is not True:
            raise ValueError('ATAC anndata needs to have the same number of cells with rna anndata')
        
        atac_anndata.var['readout'] = ["accessibility"] * atac_anndata.var.shape[0]
        atac_anndata.var['id'] = atac_anndata.var.index
        var_atac = atac_anndata.var
        multi_anndata_merge = anndata.concat([rna_anndata_merge,atac_anndata], axis=1)

        ## process side information
        # var
        var = pd.concat([var_rna[["id","readout"]],var_atac[["id","readout"]]],axis = 0)
        multi_anndata_merge.var = var.copy()

        # layers
        for i in list(multi_anndata_merge.layers.keys()): 
            del multi_anndata_merge.layers[i]
        
        # obs
        for key, value in rna_anndata_merge.obs.items():
            multi_anndata_merge.obs[key+'_rna'] = value
        for key, value in atac_anndata.obs.items():
            multi_anndata_merge.obs[key+'_atac'] = value

        # uns
        for key, value in rna_anndata_merge.uns.items():
            multi_anndata_merge.uns[key] = value
        for key, value in atac_anndata.uns.items():
            multi_anndata_merge.uns[key+'_atac'] = value

        multi_anndata_merge.layers["readout"] = multi_anndata_merge.X.copy()   
        multi_anndata_merge.X = csr_matrix(multi_anndata_merge.X)

        return multi_anndata_merge
    else:
        rna_anndata_merge.X = csr_matrix(rna_anndata_merge.X)

        return rna_anndata_merge

def sanity_check(
       adata,
       network_mode: Literal["GENIE3","full_ODE"] = "GENIE3",
    ) -> anndata.AnnData:

    if network_mode == "GENIE3":
        reg_index = [i in adata.var.index.values for i in adata.uns["regulators"]]
        tar_index = [i in adata.var.index.values for i in adata.uns["targets"]]
        adata.uns["regulators"] = adata.uns["regulators"][reg_index]
        adata.uns["targets"] = adata.uns["targets"][tar_index]
        W = adata.uns["skeleton"]
        W = W[reg_index,:]
        W = W[:,tar_index]
        adata.uns["skeleton"] = W
        W = adata.uns["network"]
        W = W[reg_index,:]
        W = W[:,tar_index]
        adata.uns["network"] = W
        
        regulators = adata.uns["regulators"][adata.uns["skeleton"].sum(1) > 0]
        targets = adata.uns["targets"][adata.uns["skeleton"].sum(0) > 0]
        
        adata = adata[:,np.unique(regulators.tolist()+targets.tolist())].copy()
        
        ## to make sure consistency
        regulator_index = [i in regulators for i in adata.var.index.values]
        target_index = [i in targets for i in adata.var.index.values]
        regulators = adata.var.index.values[regulator_index]
        targets = adata.var.index.values[target_index]
        print("num regulators: "+str(len(regulators)))
        print("num targets: "+str(len(targets)))
        
        W = pd.DataFrame(adata.uns["skeleton"],index = adata.uns["regulators"],columns = adata.uns["targets"])
        W = W.loc[regulators,targets]
        adata.uns["skeleton"] = W
        W = pd.DataFrame(adata.uns["network"],index = adata.uns["regulators"],columns = adata.uns["targets"])
        W = W.loc[regulators,targets]
        adata.uns["network"] = W
        
        adata.uns["regulators"] = regulators
        adata.uns["targets"] = targets

    if network_mode == "full_ODE":
        ## filter the gene first
        gene_name = adata.var.index.tolist()
        full_name = adata.uns["regulators"]
        index = [i in gene_name for i in full_name]
        full_name = full_name[index]
        adata = adata[:,full_name].copy()

        W = adata.uns["skeleton"]
        W = W[index,:]
        W = W[:,index]
        adata.uns["skeleton"] = W 
        W = adata.uns["network"]
        W = W[index,:]
        W = W[:,index]
        adata.uns["network"] = W
        
        ###
        W = adata.uns["skeleton"]
        gene_name = adata.var.index.tolist()
    
        indicator = W.sum(0) > 0 ## every gene would need to have a upstream regulators
        regulators = [gene for gene, boolean in zip(gene_name, indicator) if boolean]
        targets = [gene for gene, boolean in zip(gene_name, indicator) if boolean]
        print("num regulators: "+str(len(regulators)))
        print("num targets: "+str(len(targets)))
        W = adata.uns["skeleton"]
        W = W[indicator,:]
        W = W[:,indicator]
        adata.uns["skeleton"] = W
        W = adata.uns["network"]
        W = W[indicator,:]
        W = W[:,indicator]
        adata.uns["network"] = W
        adata.uns["regulators"] = regulators
        adata.uns["targets"] = targets

        W = pd.DataFrame(adata.uns["skeleton"],index = adata.uns["regulators"],columns = adata.uns["targets"])
        W = W.loc[regulators,targets]
        adata.uns["skeleton"] = W
        W = pd.DataFrame(adata.uns["network"],index = adata.uns["regulators"],columns = adata.uns["targets"])
        W = W.loc[regulators,targets]
        adata.uns["network"] = W

        adata = adata[:,indicator].copy()
    
    return adata

    
    