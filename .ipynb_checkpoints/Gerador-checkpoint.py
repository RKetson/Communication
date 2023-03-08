import numpy as np
from files_01_detection.const_mod import Model

def Train_Data(Mod, total_num_symbols, M, channel_type, Es, code_rate, min, max, local):

    symbs, indices, channel_output = Model(Mod, total_num_symbols, M, channel_type, Es, code_rate, [min, max])
    x = np.stack([np.real(channel_output[0][:]),
                    np.imag(channel_output[0][:])], axis=1)
    x = np.concatenate((x, np.array([np.real(channel_output[1])]).T), axis=1)
    
    y = np.float_(indices[0])
    
    x.tofile(local + 'x_rand.dat')
    y.tofile(local + 'y_rand.dat')
    symbs.tofile(local + 'symb.dat')
    
    return 0