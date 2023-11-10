import numpy as np
import sys
sys.path.append('..')
from Files_aux.files_01_detection.const_mod import Model

def Train_Data(Mod, total_num_symbols, M, channel_type, Es, code_rate, snr_range, local=None):
    symbs, indices, channel_output, channel_alph, bits = Model(Mod, total_num_symbols, M, channel_type, Es, code_rate, snr_range)

    shape_output = np.shape(channel_output)
    x = np.array([])
     
    x = np.append(x, np.stack([np.real(channel_output[:]),
                    np.imag(channel_output[:])], axis=1))
    
    y = np.float_(indices).reshape(-1)
    
    if local is not None:
        x.tofile(local + 'x_rand.dat')
        channel_alph.tofile(local + 'alph.dat')
        y.tofile(local + 'y_rand.dat')
        symbs.tofile(local + 'symb.dat')
    return x, channel_alph, y, symbs, bits