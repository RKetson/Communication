import numpy as np
import commpy.modulation as cm
import commpy.utilities as cu
import matplotlib.pyplot as plt
import commpy.channels
from libs.commpy_mod import SISOFlatChannel

def mod_constellation(M, unitAvgPower=True, mod='PSK'):
    bits_per_symbol = int(np.log2(M))
    bitarrays = [cu.dec2bitarray(obj, bits_per_symbol)
                 for obj
                 in np.arange(0, M)]
    sig_mod = cm.PSKModem(M) if mod == 'PSK' else cm.QAMModem(M)
    const  = np.array([complex(sig_mod.modulate(bits)) for bits in bitarrays])

    if unitAvgPower and mod == 'QAM':
        const = const / np.sqrt((M - 1) * (2 ** 2) / 6)

    return const

def mod_demod(mod, x, M, unitAvgPower=True):
    const = mod_constellation(M, unitAvgPower=unitAvgPower, mod=mod)

    const = const.reshape(const.shape[0], 1)
    return abs(x - const).argmin(0)

def generate_symbols(mod, transmissions=100, M=16):
    """
    Parameters
    ----------
    transmissions: int
        Number of transmissions. Default is 100.
    M: int
        Number of symbols in the constellation. Default is 16.
    Returns
    -------
    """
    constellation = mod_constellation(M, unitAvgPower=True, mod=mod)

    ind = np.random.randint(M, size=transmissions)

    # PSK symbols for each antenna
    x   = constellation[ind]

    return x, ind

def Model(Mod, num_symbols, M, type, Es, code_rate, SNR_dB):
    
    symbs, indices = generate_symbols(Mod, num_symbols, M)
    
    if type == 'awgn':
        channel = SISOFlatChannel(None, (1 + 0j, 0j))
        channel.set_SNR_dB(SNR_dB, float(code_rate), Es)
        output = channel.propagate(symbs)
        
    elif type == 'rayleigh':
        channel = SISOFlatChannel(None, (0j, 1 + 0j))
        channel.set_SNR_dB(SNR_dB, float(code_rate), Es)
        output = channel.propagate(symbs)
    
    elif type == 'crazy':
        output = crazy_channel_propagate(symbs, SNR_dB)
        
    else:
        raise ValueError(f'Channel type {type} not found')
    
    return symbs.reshape(1,-1), indices.reshape(1,-1), output

def main():
    num_of_symbols = 3000
    symbs, indices=generate_symbols(transmissions=num_of_symbols, M=16)
    channel = SISOFlatChannel(None, (1 + 0j, 0j))

    SNR_dB=15  #AK: this is not working!!! SNR does not change
    code_rate=1 #Rate of the used code
    Es=1 #Average symbol energy
    channel.set_SNR_dB(SNR_dB, float(code_rate), Es)

    #transmit over the channel
    channel_output = channel.propagate(symbs)

    plt.plot(np.real(channel_output),np.imag(channel_output),'bo')
    plt.ylabel('Quadrature')
    plt.xlabel('In-phase')

    #print(indices.shape)
    #if want to plot original Tx symbols
    #for i in range(num_of_symbols):
    #    print(np.real(symbs[i]), ',',np.imag(symbs[i]),',',indices[i])

    plt.plot(np.real(symbs),np.imag(symbs),'rx')

    for i in range(num_of_symbols):
        print(np.real(channel_output[i]), ',',np.imag(channel_output[i]),',',indices[i])

    plt.show()

if __name__ == '__main__':
    main()

