

## residue-residue interactions available
|interactions|     Foldx | graphein | RIP-MD | Kamil |
|-------|-----------|----------|--------|-------|
|**aromatic**|        | [x]        | [x] | [x]   |
|**hydrogen**| [x]     |     [x]    | [x] | |
|**hydrophobic**|      |   [x]     | [x]   |  |
|**volumetric**|  [x]  |          | | | 
|**vdW**|              | [x]      | [x] | | 
|**charge**| [x]       |           | [x] | [x] |
|contact| [x]      |           | [x] | [x] |
|**disulfide**| [x]    |  [x]       |  [x] | [x] |
|partcov| [x]      |          | |
|water| [x]        |           | |
|distance|      |   [x]         |  [x] | [x] | 
|**cation pi**|      |    [x]        | [x] | [x] |
|**ionic**|      |        [x]    | |
|**salt bridges**|   |    [x]         | [x] | [x] |
|stacking pi |    |     [x]         | | 
|delaunay     |    |        [x]     | |
|**arg - arg**|       |            |  [x] | [x] 
--------------------------------------


* partcov -  Interactions made by protein atoms and metals


## information source
* Foldx print networks:  http://foldxsuite.crg.eu/command/PrintNetworks
* Graphein: https://graphein.ai/modules/graphein.protein.html#edges
* RIP-MD paper

## selected criteria
**disulfde**   $$dist(Sulfur, Sulfur) < 2.2 A$$
**hydrophobic** $$dist(\text{any R-group atom}, \text{any R-group atom} ) < 5 A$$ for ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO, TYR
**ionic** $$dist(C_{\alpha i}, C_{\alpha j} ) < 6 A$$  where i,j in  ARG, LYS, HIS, ASP, GLU
**salt bridges** $$dist(NH/NZ, OE/OD)$$ for ARG/LYS and ARG/GLU respectively
**arg-arg** $$dist(\text{guanidine}_i, \text{guanidine}_j) < 5 A$$
**cation $$\pi$$** $$dist(\text{aromatic ring}_i, \text{aromatic ring}_j)$$ i in Trp, Tyr, Phe (His?)
