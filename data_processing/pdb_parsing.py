import numpy as np
import random

# Converts the protein-ligand complexes into 4D tensor. 
class FeatureExtractorPDB():
    def __init__(self):
        self.atom_codes = {}
        # 'others' includs metal atoms and B atom. There are no B atoms on training and test sets. 
        others = ([3,4,5,11,12,13]+list(range(19,32))+list(range(37,51))+list(range(55,84)))
        # C and N atoms can be hybridized in three ways and S atom can be hybridized in two ways here. 
        # Hydrogen atom is also considered for feature extraction.
        
        # 1 is H, 6 is C, 7 is N, 8 is O, 9 is F, 15 is P, 16 is S, 17 is Cl, 19-32 are Arsenic to Selenium, 37-51 are Rubidium to Indium, 55-84 are Cesium to Polonium
        atom_types = [1,(6,1),(6,2),(6,3),(7,1),(7,2),(7,3),8,15,(16,2),(16,3),34,[9,17,35,53],others] 
        # (6,1): C atom with sp hybridization, (6,2) C sp2, (6,3) C sp3
        # (16,2): S sp2, (16,3) S sp3 sp3d
      
        for i, j in enumerate(atom_types):
            if type(j) is list:
                for k in j:
                    self.atom_codes[k] = i
                
            else:
                self.atom_codes[j] = i              
        
        self.sum_atom_types = len(atom_types)
        
    # Onehot encoding of each atom. The atoms in protein or ligand are treated separately.
    def encode(self, atomic_num, molprotein):
        encoding = np.zeros(self.sum_atom_types*2)
        if molprotein == 1: # protein
            encoding[self.atom_codes[atomic_num]] = 1.0 
        else: # ligand
            encoding[self.sum_atom_types+self.atom_codes[atomic_num]] = 1.0
        
        return encoding

    # Get atom coords and atom features from the complexes.   
    def get_features(self, molecule, molprotein):
        coords = []
        features = []
            
        for atom in molecule:
            coords.append(atom.coords)
            if atom.atomicnum in [6,7,16]:
                atomicnum = (atom.atomicnum,atom.hyb)
                features.append(self.encode(atomicnum,molprotein))
            else:
                features.append(self.encode(atom.atomicnum,molprotein))
        
        coords = np.array(coords, dtype=np.float32) # shape [num_atoms, 3]
        features = np.array(features, dtype=np.float32) # shape [num_atoms, 28]
        
        return coords, features
     
    # Define the rotation matrixs of 3D stuctures.
    def rotation_matrix(self, t, roller):
        if roller==0:
            return np.array([[1,0,0],[0,np.cos(t),np.sin(t)],[0,-np.sin(t),np.cos(t)]])
        elif roller==1:
            return np.array([[np.cos(t),0,-np.sin(t)],[0,1,0],[np.sin(t),0,np.cos(t)]])
        elif roller==2:
            return np.array([[np.cos(t),np.sin(t),0],[-np.sin(t),np.cos(t),0],[0,0,1]])

    # Generate 3d grid or 4d tensor. Each grid represents a voxel. Each voxel represents the atom in it by onehot encoding of atomic type.
    # Each complex in train set is rotated 9 times for data amplification.
    # The complexes in core set are not rotated. 
    # The default resolution is 20*20*20.
    def grid(self, coords, features, resolution=1.0, max_dist=10.0, n_amplification=9):
        """
        Generate a grid representation of the protein-ligand complex.

        n_amplification: int = number of rotations to apply to the complex.
        if n_amplification=0, the complex is not rotated, only the original complex is used.
        (return shape [1, 20, 20, 20, features.shape[1]])
        otherwise, the complex is rotated n_amplification times,
        and return shape [n_amplification + 1, 20, 20, 20, features.shape[1]].
        """
        assert coords.shape[1] == 3
        assert coords.shape[0] == features.shape[0]  

        grid=np.zeros((n_amplification + 1, 20, 20, 20, features.shape[1]),dtype=np.float32)
        x=y=z=np.array(range(-10, 10),dtype=np.float32)+0.5
        for i in range(len(coords)):
            coord=coords[i]
            tmpx=abs(coord[0]-x)
            tmpy=abs(coord[1]-y)
            tmpz=abs(coord[2]-z) 
            if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                grid[0,np.argmin(tmpx),np.argmin(tmpy),np.argmin(tmpz)] += features[i]
                
                
        # for testing, the complex is not rotated. n_rotations=1
        
        for j in range(n_amplification):
            theta = random.uniform(np.pi/18,np.pi/2)
            roller = random.randrange(3)
            coords = np.dot(coords, self.rotation_matrix(theta,roller))
            for i in range(len(coords)):
                coord=coords[i]
                tmpx=abs(coord[0]-x)
                tmpy=abs(coord[1]-y)
                tmpz=abs(coord[2]-z)
                
                # because the grid is 20x20x20, and tmp(x,y,z) out of 19.5 means that the atom is out of the grid, then eliminate it
                if np.max(tmpx)<=19.5 and np.max(tmpy)<=19.5 and np.max(tmpz) <=19.5:
                    grid[j+1,np.argmin(tmpx),np.argmin(tmpy),np.argmin(tmpz)] += features[i]
                
        return grid

def test_get_grid(protein, ligand):
    Feature = FeatureExtractorPDB()
    coords1, features1 = Feature.get_features(protein,1)
    coords2, features2 = Feature.get_features(ligand,0)
    
    center=(np.max(coords2,axis=0)+np.min(coords2,axis=0))/2
    coords=np.concatenate([coords1,coords2],axis = 0)
    features=np.concatenate([features1,features2],axis = 0)
    assert len(coords) == len(features)
    coords = coords-center
    grid=Feature.grid(coords,features)
    
    return grid