import os

import pandas as pd
import seaborn as sns
import shutil
import re


def pymol_adjust(fname_png):
    formatting = \
    'set cartoon_fancy_helices, 1\n' + \
    'set cartoon_fancy_sheets, 1\n' + \
    'set spec_reflect, 0\n' + \
    'set ray_trace_mode, 1\n' + \
    'set ray_shadow, 0\n' + \
    'set ray_trace_color, black\n' + \
    'set bg_rgb=[1,1,1]\n' + \
    'set cartoon_highlight_color, grey90\n' + \
    f'png {fname_png}.png, width=1000, height=1000, ray=1\n'
    return formatting

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


class NbToCol:
    NUM_COLORS = 16
    def __init__(self):
        self.cmap = sns.diverging_palette(240, 125, s=80, l=55, n=self.NUM_COLORS, as_cmap=True)
        
    def map_color(self, val):
        rgb_tuple = self.cmap(val)
        rgb_list = [round(col, 4) for col in rgb_tuple[:3]]
        return str(rgb_list)
    


class PyMolColor:
    template  = None
    def __init__(self, custom_cmap=None):
        '''
        initialize class instance
        params:
            custom_cmap (plt.cmap)
        
        '''
        self.cmap = NbToCol()
        if custom_cmap is not None:
            self.cmap.cmap = custom_cmap
    
    def init_script(self, fname, pdb_chain, pdb_list, embeddings): 
        '''
        prefix (str): dictionary to store script
        fname (str): output .pml script name
        chain (str): pdb chain file name
        cof (int): 0 - NAD, 1 - NADP, 2 - SAM, 3 - FAD
        embeddings (np.ndarray): scores for each residue as integers [0-255]
        '''
        pdb, chain = pdb_chain.split('_')

        ### get residues
        residues = [str(i) if i is not None else None for i in pdb_list ]
        #print(embeddings.shape, len(residues))
        #print(residues)
        name_cof = 'xtz'
        if not fname.endswith('.pml'):
            fname = fname + '.pml'
        with open(fname, 'wt') as f:
            f.write(f'fetch {pdb}, async=0\n')
            for i, (ri, ei) in enumerate(zip(residues, embeddings)):
                if ri is None:
                    continue
                coli = self.cmap.map_color(ei)
                name_coli = f'color{i}{name_cof}'
                name_resi = f'res {ri}'
                f.write(f'set_color {name_coli}, {coli}\n')
                f.write(f'color {name_coli}, resi {ri}\n')
            f.write(pymol_adjust(fname))
        
    def draw(self, directory, chain, chain_name, pdb_list, embeddings,**kw_args):
        '''
        creates pymol scripts with coloring from GNN embeddings
        `directory` stores separate file for each cofactor and copied pdb strucutre
        to run them
        params:
            directory (str) path to store results, must exists
            chain (str) path to .pdb core
            chain_name (str) name of directory
            sequence (str) Rossmann core sequence used to find graph for it
        '''
        #print(scores_scaled.shape)
        #create scripts
        path_preffix = os.path.join(directory, chain_name)
        self.create_dir(directory)
        self.create_dir(path_preffix)
        pdb_file_name = os.path.basename(chain)
        shutil.copy(chain, path_preffix + '/' + pdb_file_name)
                
        self.init_script(path_preffix, chain, pdb_list, embeddings)
  
    @classmethod
    def create_dir(cls, directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)