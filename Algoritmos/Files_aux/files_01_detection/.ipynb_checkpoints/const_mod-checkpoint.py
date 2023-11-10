import numpy as np
import commpy.modulation as cm
import commpy.utilities as cu
import matplotlib.pyplot as plt
import commpy.channels
from libs.commpy_mod import SISOFlatChannel
import commpy.channelcoding.convcode as cc

def bit_generation(M, indices):
    bits_per_symbol = int(np.log2(M))
    bitarrays = [cu.dec2bitarray(obj, bits_per_symbol) for obj in indices]
    
    return bitarrays

#def codec_symbs(bits_array, bits_per_symbol, trellis):
#    codec_array = np.array([cc.conv_encode(bits, trellis, 'cont') for bits in bits_array])
    
#    return codec_array.reshape((-1, bits_per_symbol))
def codec_conv(msg, matriz_g):

    n = matriz_g.shape[1]
    k = matriz_g.shape[0]
    rate = k / n

    g_poly = []
    for j in range(k):
        y = [list(bin(matriz_g[j, i]).split('b')[1]) for i in range(n)]
        g_poly.append(y)
    g_poly = np.array(g_poly)

    # Inicialização dos registradores de deslocamento
    register = [0] * (g_poly.shape[-1] - 1)

    # Inicialização da sequência codificada
    encoded_data = []

    # Implementação apenas com 1 bit de informação
    # Codificação convolucional
    for bit in msg:
        # Atualiza os registradores de deslocamento
        register.insert(0, bit)

        # Calcula os bits de saída
        output = [sum([bit * int(coef) for coef, bit in zip(g_poly[0, i], register)]) % 2
                for i in range(n)]

        # Adiciona os bits de saída à sequência codificada
        encoded_data.extend(output)

        # Remove os bits mais antigos dos registradores de deslocamento
        register.pop()

    # A sequência codificada é a saída
    return np.array(encoded_data), rate

def mod_constellation(bits_array, M, unitAvgPower=True, mod='PSK', rate=1):
    sig_mod = cm.PSKModem(M) if mod == 'PSK' else cm.QAMModem(M)
    const  = np.array([complex(sig_mod.modulate(bits)) for bits in bits_array])
    
    if unitAvgPower and mod == 'QAM':
            const = (const / np.sqrt((M - 1) * (2 ** 2) / 6))

    return (const / rate).reshape(-1) 
   
"""
def mod_constellation(M, indices, unitAvgPower=True, mod='PSK', trellis=None):
    bits_per_symbol = int(np.log2(M))
    bitarrays = []
    sig_mod = cm.PSKModem(M) if mod == 'PSK' else cm.QAMModem(M)
    const = []
    if trellis is None:
        bitarrays = [cu.dec2bitarray(obj, bits_per_symbol)
                     for obj
                     in indices]
        const  = np.array([complex(sig_mod.modulate(bits)) for bits in bitarrays])
    else:
        bitarrays = [cc.conv_encode(cu.dec2bitarray(obj, bits_per_symbol), trellis, 'cont')
                     for obj
                     in indices]
        const  = np.array([[complex(sig_mod.modulate(bits[i:i + bits_per_symbol])) for i in range(0, len(bits), bits_per_symbol)] for             bits in bitarrays])

    if unitAvgPower and mod == 'QAM':
        if trellis is None:
            const = const / np.sqrt((M - 1) * (2 ** 2) / 6)
        else:
            k = trellis.k
            n = trellis.n
            rate = float(k) / n
            n_out_bits = (bits_per_symbol * rate)
            
            const = (const / np.sqrt((M - 1) * (2 ** 2) / 6)) / n_out_bits

    return const.reshape(-1)
"""

def mod_demod(mod, x, M, unitAvgPower=True):
    const = mod_constellation(M, unitAvgPower=unitAvgPower, mod=mod)

    const = const.reshape(const.shape[0], 1)
    return abs(x - const).argmin(0)

def generate_symbols(mod, transmissions=100, M=16, codec=False):
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
    ind = np.random.randint(0, M, transmissions)
    bits_per_symbol = int(np.log2(M))
    bits_gerados = bit_generation(M, ind)
    bits = np.array(bits_gerados[:])
    x = []
    rate = 1
    
    if codec:
        # Parâmetros do código convolucional
        #constraint_length = np.array(3, ndmin=1)  # Comprimento de restrição do código (3 neste exemplo)
        #code_generator = np.array((5, 7), ndmin=2)  # Polinômio gerador em octal

        # Criando o objeto do código convolucional
        #trellis = cc.Trellis(memory=constraint_length, g_matrix=code_generator)
        #rate = float(trellis.k) / trellis.n
        g = np.array([[5, 7]])
        bits, rate = codec_conv(bits.reshape(-1), g)
        bits = bits.reshape((-1, bits_per_symbol))
        #bits = codec_symbs(bits, bits_per_symbol, trellis) 
    
    x = mod_constellation(bits, M, unitAvgPower=True, mod=mod, rate=rate)
#    x = np.array([])
#    ind = np.array([])
#    for i in range(transmissions):
#        constellation = mod_constellation(M, unitAvgPower=True, mod=mod, trellis=trellis)
#
#        ind = np.append(ind, np.random.randint(M))

        # PSK symbols for each antenna
#        x   = np.append(x, constellation[int(ind[-1])])

    return x, ind, bits_gerados

def Model(Mod, num_symbols, M, type, Es, code_rate, SNR_dB, vel_alph=0):
    codec = True if code_rate != 1 else False
    symbs, indices, bits = generate_symbols(Mod, num_symbols, M, codec)
    def Propagate(channel, len_faixa):
        output = np.array([])
        alph = np.array([])
        
        if len_faixa == 2:
            snr_rand = np.random.uniform(SNR_dB[0], SNR_dB[1], num_symbols)
            step = 1 if type == "awgn" else vel_alph
            for i in range(0, len(symbs), step):                
                channel.set_SNR_dB(snr_rand[int(i/step)], float(code_rate), Es)
                out, al = channel.propagate(symbs[i:i+step], True)
                alph = np.append(alph, np.array(al))
                output = np.append(output, np.array(out))
            output = np.array(output).reshape(-1) if type == "awgn" else np.array(output).reshape((-1, vel_alph))
        elif len_faixa == 1:
            channel.set_SNR_dB(SNR_dB[0], float(code_rate), Es)
            step = 1 if type == "awgn" else vel_alph
            for i in range(0, len(symbs), step):
                out, al = channel.propagate(symbs[i:i+step], True)
                alph = np.append(alph, np.array(al))
                output = np.append(output, np.array(out))
            output = np.array(output).reshape(-1) if type == "awgn" else np.array(output).reshape((-1, vel_alph))
        else:
            raise ValueError(f'Faixa de SNR mal especificada')
        
        return output, alph
    
    if type == 'awgn':
        channel = SISOFlatChannel(None, (1 + 0j, 0j))
        output = Propagate(channel, len(SNR_dB))
        
    elif type == 'rayleigh':
        channel = SISOFlatChannel(None, (0j, 1 + 0j))
        output = Propagate(channel, len(SNR_dB))
    
    elif type == 'crazy':
        output = crazy_channel_propagate(symbs, SNR_dB)
        
    else:
        raise ValueError(f'Channel type {type} not found')
        
    return symbs.reshape(1,-1), indices.reshape(1,-1), output[0].reshape(-1), output[1], bits

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

