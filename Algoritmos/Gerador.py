import numpy as np
from files_01_detection.const_mod import Model

def Train_Data(Mod, total_num_symbols, M, channel_type, Es, code_rate, min, max, local=None):
    symbs, indices, channel_output, channel_alph = Model(Mod, total_num_symbols, M, channel_type, Es, code_rate, [min, max])

    shape_output = np.shape(channel_output)
    x = np.array([])
    
    for i in range(shape_output[0]):       
        x = np.append(x, np.stack([np.real(channel_output[i][:]),
                        np.imag(channel_output[i][:])], axis=1))
    x = x.reshape((-1, shape_output[1], 2))
    
    y = np.float_(indices[0]).reshape((-1, shape_output[1]))
    
    if local is not None:
        x.tofile(local + 'x_rand.dat')
        channel_alph.tofile(local + 'alph.dat')
        y.tofile(local + 'y_rand.dat')
        symbs.tofile(local + 'symb.dat')
    
    return x, channel_alph.reshape((-1, 1)), y, symbs