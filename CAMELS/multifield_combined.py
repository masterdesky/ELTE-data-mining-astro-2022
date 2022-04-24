import torch
import numpy as np

class MultifieldDataset():
    '''
    
    '''
    def __init__(self,
                 mode, fmaps, fmaps_norm, fparams, splits, *,
                 norm_params=True, 
                 memmap=False, seed=None, verbose=False):
        
        # Private parameters
        self.memmap = memmap
        
        # get the total number of sims and maps
        y_raw = np.loadtxt(fparams)
        n_sims, n_maps, n_params = \
                y_raw.shape[0], y_raw.shape[0]*splits, y_raw.shape[1]
        y = np.zeros((n_maps, n_params), dtype=np.float32)
        for i in range(n_sims):
            for j in range(splits):
                y[i*splits + j] = y_raw[i]

        # Normalize label values
        # > These values are the preset limits of simulations in the
        # > CAMELS dataset: Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, A_AGN2
        # > respectively
        if norm_params:
            minimum = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
            maximum = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
            y = (y - minimum) / (maximum - minimum)

        # get the size and offset depending on the type of dataset
        if   mode=='train':
            offset, size_sims = int(0.00*n_sims), int(0.90*n_sims)
        elif mode=='valid':
            offset, size_sims = int(0.90*n_sims), int(0.05*n_sims)
        elif mode=='test':
            offset, size_sims = int(0.95*n_sims), int(0.05*n_sims)
        elif mode=='all':
            offset, size_sims = int(0.00*n_sims), int(1.00*n_sims)
        else:
            raise Exception("Wrong name! Should be `train`, `valid`, `test` or `all`")
        size_maps = size_sims*splits

        # Shuffle the simulations (indeces from 0 to 999 in case of CAMELS)
        np.random.seed(seed)
        sim_numbers = np.arange(n_sims) #shuffle sims not maps
        np.random.shuffle(sim_numbers)
        sim_numbers = sim_numbers[offset:offset+size_sims] #select indexes of mode

        # get the corresponding indexes of the maps associated to the sims
        indexes = np.zeros(size_maps, dtype=np.int32)
        count = 0
        for i in sim_numbers:
            for j in range(splits):
                indexes[count] = i*splits + j
                count += 1

        # keep only the value of the parameters of the considered maps
        y = y[indexes]

        # define the matrix containing the maps with rotations and flipings
        channels = len(fmaps)
        dumb     = np.load(fmaps[0])    #[number of maps, height, width]
        height, width = dumb.shape[1], dumb.shape[2];  del dumb
        if not memmap:
            data = np.zeros((size_maps, channels, height, width),
                            dtype=np.float32)
        else:
            y_maps = y
            y = np.zeros((size_maps*8, n_params), dtype=np.float32)
            data = np.zeros((size_maps*8, channels, height, width),
                            dtype=np.float32)

        # read the data
        print(f"Found {channels} channels\nReading data...")
        for channel, (fmap, fnorm) in enumerate(zip(fmaps, fmaps_norm)):

            # read maps in the considered channel
            data_c = np.load(fmap)
            if data_c.shape[0] != n_maps:
                raise Exception("Sizes do not match")
            if verbose:
                print(f"{np.min(data_c):.3e} < F(all|orig) < {np.max(data_c):.3e}")

            # rescale maps
            if fmap.find('Mstar') != -1:
                data_c = np.log10(data_c + 1.0)
            else:
                data_c = np.log10(data_c)
            if verbose:
                print(f"{np.min(data_c):.3e} < F(all|resc) < {np.max(data_c):.3e}")

            # normalize maps
            if fnorm is None:  
                mean, std = np.mean(data_c), np.std(data_c)
            else:
                # read data
                data_norm = np.load(fnorm)

                # rescale data
                if fnorm.find('Mstar') != -1:
                    data_norm = np.log10(data_norm + 1.0)
                else:
                    data_norm = np.log10(data_norm)

                # compute mean and std
                mean, std = np.mean(data_norm), np.std(data_norm)
                del data_norm

            data_c = (data_c - mean) / std
            if verbose:
                print(f"{np.min(data_c):.3e} < F(all|norm) < {np.max(data_c):.3e}") 


            if not memmap:
                # keep only the data of the chosen set
                data[:,channel,:,:] = data_c[indexes]
            else:
                data_c = data_c[indexes]

                # Loop over all rotation angles (0: 0, 1: 90, 2: 180, 3: 270)
                c_maps = 0
                for rot in range(4):
                    data_rot = np.rot90(data_c, k=rot, axes=(1, 2))

                    data[c_maps:c_maps+size_maps, channel, :, :] = data_rot
                    y[c_maps:c_maps+size_maps]              = y_maps
                    c_maps += size_maps

                    data[c_maps:c_maps+size_maps, channel, :, :] = \
                                                np.flip(data_rot, axis=1)
                    y[c_maps:c_maps+size_maps]              = y_maps
                    c_maps += size_maps

                if verbose:
                    print(f"Channel {channel} contains {c_maps} maps")
                    print(f"{np.min(data_c):.3e} < F < {np.max(data_c):.3e}")
        
        self.size = data.shape[0]
        self.X    = torch.tensor(data,   dtype=torch.float32)
        self.y    = torch.tensor(y, dtype=torch.float32)
        del data, data_c

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.memmap:
            # Choose a rotation angle (0: 0, 1: 90, 2: 180, 3: 270)
            # and whether do flipping or not
            rot  = np.random.randint(0, 4)
            flip = np.random.randint(0, 1)

            # rotate and flip the maps
            maps = torch.rot90(self.X[idx], k=rot, dims=[1, 2])
            if flip == 1:
                maps = torch.flip(maps, dims=[1])

            return maps, self.y[idx]
        else:
            return self.X[idx], self.y[idx]