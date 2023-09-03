import numpy as np
import commpy.modulation as cm
import commpy.utilities as cu
import matplotlib.pyplot as plt
import commpy.channels
from libs.commpy_mod import SISOFlatChannel
import commpy.channelcoding.convcode as cc

def mod_constellation(M, unitAvgPower=True, mod='PSK', trellis=None):
    bits_per_symbol = int(np.log2(M))
    bitarrays = []
    sig_mod = cm.PSKModem(M) if mod == 'PSK' else cm.QAMModem(M)
    const = []
    if trellis is None:
        bitarrays = [cu.dec2bitarray(obj, bits_per_symbol)
                     for obj
                     in np.arange(0, M)]
        const  = np.array([complex(sig_mod.modulate(bits)) for bits in bitarrays])
    else:
        bitarrays = [cc.conv_encode(cu.dec2bitarray(obj, bits_per_symbol), trellis, 'cont')
                     for obj
                     in np.arange(0, M)]
        const  = np.array([[complex(sig_mod.modulate(bits[i:i + bits_per_symbol])) for i in range(0, len(bits), bits_per_symbol)] for             bits in bitarrays])

    if unitAvgPower and mod == 'QAM':
        if trellis is None:
            const = const / np.sqrt((M - 1) * (2 ** 2) / 6)
        else:
            k = trellis.k
            n = trellis.n
            rate = float(k) / n
            n_out_bits = (bits_per_symbol / rate)
            
            const = (const / np.sqrt((M - 1) * (2 ** 2) / 6)) / n_out_bits

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
    # Parâmetros do código convolucional
    bits_per_symbol = int(np.log2(M))
    constraint_length = np.array(bits_per_symbol, ndmin=1)  # Comprimento de restrição do código (3 neste exemplo)
    code_generator = np.array((5, 7), ndmin=2)  # Polinômio gerador em octal

    # Criando o objeto do código convolucional
    trellis = cc.Trellis(memory=constraint_length, g_matrix=code_generator)
    
    x = np.array([])
    ind = np.array([])
    for i in range(transmissions):
        constellation = mod_constellation(M, unitAvgPower=True, mod=mod, trellis=trellis)

        ind = np.append(ind, np.random.randint(M))

        # PSK symbols for each antenna
        x   = np.append(x, constellation[int(ind[-1])])

    return x, ind

def Model(Mod, num_symbols, M, type, Es, code_rate, SNR_dB, vel_alph=20):
    
    symbs, indices = generate_symbols(Mod, num_symbols, M)
    
    def Propagate(channel, len_faixa, SNR_dB, code_rate, Es, vel_alph=20):
        output = np.array([])
        alph = np.array([])
        
        if len_faixa == 2:
            snr_rand = np.random.uniform(SNR_dB[0], SNR_dB[1], num_symbols)
            for i in range(0, len(symbs), vel_alph):                
                channel.set_SNR_dB(snr_rand[int(i/vel_alph)], float(code_rate), Es)
                out, al = channel.propagate(symbs[i:i+vel_alph], True)
                alph = np.append(alph, np.array(al))
                output = np.append(output, np.array(out))
            output = np.array(output).reshape((-1, vel_alph))
        elif len_faixa == 1:
            channel.set_SNR_dB(SNR_dB[0], float(code_rate), Es)
            for i in range(0, num_symbols, vel_alph):
                out, alph = channel.propagate(symbs[i:i+vel_alph], True)
                alph = np.append(alph, np.array(al))
                output = np.append(output, np.array(out))
            output = np.array(output).reshape((-1, vel_alph))
        else:
            raise ValueError(f'Faixa de SNR mal especificada')
        
        return output, alph
    
    if type == 'awgn':
        channel = SISOFlatChannel(None, (1 + 0j, 0j))
        output = Propagate(channel, len(SNR_dB), SNR_dB, code_rate, Es)
        
    elif type == 'rayleigh':
        channel = SISOFlatChannel(None, (0j, 1 + 0j))
        output = Propagate(channel, len(SNR_dB), SNR_dB, code_rate, Es, vel_alph)
    
    elif type == 'crazy':
        output = crazy_channel_propagate(symbs, SNR_dB)
        
    else:
        raise ValueError(f'Channel type {type} not found')
    
    return symbs.reshape(1,-1), indices.reshape(1,-1), output[0].reshape(-1), output[1]

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

