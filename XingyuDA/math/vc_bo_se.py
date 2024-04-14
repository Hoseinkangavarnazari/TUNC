import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from random import randint

q = 10**(-3)                            # security parameter
e = np.exp(1)                           # Euler's number
p = 2**8                                # finite field size

m = 10          # genration size
n = 1000        # message size (symbols)

delta = 0.2
gamma = 10
N = 6                                  # Network size


def bits_per_symbol(p) -> int:
    return int(np.ceil(np.log2(p)))


def cff(w) -> int:
    keypool_size = e * (w+1) * np.log(1/q)
    return int(np.ceil(keypool_size))


def drkd(w) -> (int, int):
    keypool_size = e * (w+1)**2 * (2 / (delta**2)) * (gamma + 1) * np.log(N)
    ell = (1 / (1 - delta)) * e * (w+1) * np.log(1/q)
    return int(np.ceil(keypool_size)), int(np.ceil(ell))


def vc_hommac(w):
    l_mac = cff(w)
    return l_mac * (2*m + n)


def vc_macsig(w):
    _, l_mac = drkd(w)
    return (m+n+1) * l_mac + (3/2) * bits_per_symbol(p) * (m + l_mac + 1)


def vc_ns_hmac(w):
    l_mac = cff(w)
    l_dmac = 2
    return (m+n+1) * l_mac + (l_mac + l_dmac)


def vc_dual_hmac(w):
    l_mac = cff(w)
    l_dmac = l_mac
    return (m+n+1) * l_mac + (l_mac + 1) * l_dmac


def vc_ehmac(w):
    _, ell = drkd(w)
    l_mac = ell / 2
    l_dmac = l_mac
    return (m+n+1) * l_mac + (l_mac + 1) * l_dmac + (3/2) * bits_per_symbol(p) * (l_dmac + 1)


def bo_hommac(w):
    l_mac = cff(w)
    return l_mac / (m+n)


def bo_macsig(w):
    keypool, l_mac = drkd(w)
    return (l_mac + 1) / (m+n) + (bits_per_symbol(keypool) * l_mac) / (bits_per_symbol(p) * (m+n))


def bo_ns_hmac(w):
    l_mac = cff(w)
    l_dmac = 2
    return (l_mac + l_dmac) / (m+n)


def bo_dual_hmac(w):
    l_mac = cff(w)
    l_dmac = l_mac
    return (l_mac + l_dmac) / (m+n)


def bo_ehmac(w):
    _, ell = drkd(w)
    l_mac = ell / 2
    l_dmac = l_mac
    return (l_mac + l_dmac + 1) / (m+n)


w_values = np.array([1, 2, 3, 4])
bar_width = 0.15  # Width of the bars
indices = np.arange(len(w_values))  # Calculating positions for each group on the x-axis


def plot_vcs():
    # Verification Complexity
    vc_hommac_values = [vc_hommac(w) for w in w_values]
    vc_macsig_values = [vc_macsig(w) for w in w_values]
    vc_ns_hmac_values = [vc_ns_hmac(w) for w in w_values]
    vc_dual_hmac_values = [vc_dual_hmac(w) for w in w_values]
    vc_ehmac_values = [vc_ehmac(w) for w in w_values]

    plt.figure(figsize=(10, 6))
    plt.bar(indices - bar_width*2, vc_hommac_values, width=bar_width, label='HomMac')
    plt.bar(indices - bar_width, vc_macsig_values, width=bar_width, label='MacSig')
    plt.bar(indices, vc_ns_hmac_values, width=bar_width, label='NS-HMAC')
    plt.bar(indices + bar_width, vc_dual_hmac_values, width=bar_width, label='Dual-HMAC')
    plt.bar(indices + bar_width*2, vc_ehmac_values, width=bar_width, label='E-HMAC')

    plt.xticks(indices, w_values)  # Set x-ticks to be the values of w
    plt.xlabel('Number of Compromised Nodes', fontsize=14)
    plt.ylabel(r'Number of Multiplications ($\times 10^4$)', fontsize=14)
    
    def y_fmt(y, pos):
        return f'{int(y/10000)}'

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(y_fmt))
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=12)
    plt.grid(which='both', linestyle='-.', linewidth=0.5)
    plt.savefig('/Users/xingyuzhou/NoteOnGithub/Diplomarbeit/DA_Tex/figs/3_vcs.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.close()


def plot_bos():
    # Online Bandwidth Overhead
    bo_hommac_values = [bo_hommac(w) for w in w_values]
    bo_macsig_values = [bo_macsig(w) for w in w_values]
    bo_ns_hmac_values = [bo_ns_hmac(w) for w in w_values]
    bo_dual_hmac_values = [bo_dual_hmac(w) for w in w_values]
    bo_ehmac_values = [bo_ehmac(w) for w in w_values]

    plt.figure(figsize=(10, 6))
    plt.bar(indices - bar_width*2, bo_hommac_values, width=bar_width, label='HomMac')
    plt.bar(indices - bar_width, bo_macsig_values, width=bar_width, label='MacSig')
    plt.bar(indices, bo_ns_hmac_values, width=bar_width, label='NS-HMAC')
    plt.bar(indices + bar_width, bo_dual_hmac_values, width=bar_width, label='Dual-HMAC')
    plt.bar(indices + bar_width*2, bo_ehmac_values, width=bar_width, label='E-HMAC')

    plt.xticks(indices, w_values)
    plt.xlabel('Number of Compromised Nodes', fontsize=14)
    plt.ylabel('Online Bandwidth Overhead', fontsize=14)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(which='both', linestyle='-.', linewidth=0.5)
    plt.savefig('/Users/xingyuzhou/NoteOnGithub/Diplomarbeit/DA_Tex/figs/3_bos.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.close()


def plot_ses():
    # Security

    def generate_ones(length: int) -> int:
        return (1 << length) - 1

    def generate_random_ones(length: int, pr_denominator) -> int:
        result = 0
        for i in range(length):
            if randint(1, pr_denominator) == 1:
                result |= (1 << i)
        return result

    def format_binary(binary_int, length) -> str:
        return format(binary_int, 'b').zfill(int(length))

    def calculate_d(binary1_int, binary2_int) -> (int, int):    # count safe keys in binary1_int
        result = binary1_int & (~binary2_int)
        d = bin(result).count('1')
        return result, d

    def se_one_run(w, scheme):
        if scheme == 'dual_hmac':
            pr_denominator = 2 * (w+1)
        else:
            pr_denominator = w+1

        union_compromised = 0
        if scheme == 'macsig':
            K, l_mac = drkd(w)
            keypool = generate_ones(K)

            keys_selected_index = random.sample(range(K), l_mac)
            keypool_used = 0
            for j in keys_selected_index:
                keypool_used |= (1 << j)

            for _ in range(w):
                keyset_compromised = generate_random_ones(K, pr_denominator) & keypool_used
                union_compromised = union_compromised | keyset_compromised

            keyset_healthy = generate_random_ones(K, pr_denominator) & keypool_used
            keys_safe, d = calculate_d(keyset_healthy, union_compromised)

        elif scheme == 'ehmac':
            l_mac = int(np.ceil(drkd(w)[1] / 2))
            keypool = generate_ones(l_mac)

            for _ in range(w):
                keyset_compromised = generate_random_ones(l_mac, pr_denominator)
                union_compromised = union_compromised | keyset_compromised

            keyset_healthy = generate_random_ones(l_mac, pr_denominator)
            keys_safe, d = calculate_d(keyset_healthy, union_compromised)

        else:
            l_mac = cff(w)
            keypool = generate_ones(l_mac)

            for _ in range(w):
                keyset_compromised = generate_random_ones(l_mac, pr_denominator)
                union_compromised = union_compromised | keyset_compromised
                # print(f'{"Keyset Compromised:":<20} {format_binary(keyset_compromised, np.ceil(cff(w))):<40}, {bin(keyset_compromised).count("1")}')

            keyset_healthy = generate_random_ones(l_mac, pr_denominator)
            keys_safe, d = calculate_d(keyset_healthy, union_compromised)

        # print(f'{"Keypool:":<20} {format_binary(keypool, len(bin(keypool))-2):<40} {bin(keypool).count("1")}')
        # print(f'{"Union Compromised:":<20} {format_binary(union_compromised, len(bin(keypool))-2):<40} {bin(union_compromised).count("1")}')
        # print(f'{"Keyset Healthy:":<20} {format_binary(keyset_healthy, len(bin(keypool))-2):<40} {bin(keyset_healthy).count("1")}')
        # print(f'{"Keys Safe:":<20} {format_binary(keys_safe, len(bin(keypool))-2):<40} {bin(keys_safe).count("1")}')
        # print(f'{"D:":<20} {d}')

        return d

    def avg_d(w, scheme, runs):
        d_sum = 0
        for i in range(runs):
            d_sum += se_one_run(w, scheme)
        return d_sum / runs

    runs = 10**4
    se_hommac_values = [avg_d(w, 'hommac', runs) for w in w_values]
    se_macsig_values = [avg_d(w, 'macsig', runs) for w in w_values]
    se_ns_hmac_values = [avg_d(w, 'ns_hmac', runs) for w in w_values]
    se_dual_hmac_values = [avg_d(w, 'dual_hmac', runs) for w in w_values]
    se_ehmac_values = [avg_d(w, 'ehmac', runs) for w in w_values]

    plt.figure(figsize=(10, 6))
    plt.bar(indices - bar_width*2, se_hommac_values, width=bar_width, label='HomMac')
    plt.bar(indices - bar_width, se_macsig_values, width=bar_width, label='MacSig')
    plt.bar(indices, se_ns_hmac_values, width=bar_width, label='NS-HMAC')
    plt.bar(indices + bar_width, se_dual_hmac_values, width=bar_width, label='Dual-HMAC')
    plt.bar(indices + bar_width*2, se_ehmac_values, width=bar_width, label='E-HMAC')

    plt.xticks(indices, w_values)
    plt.xlabel('Number of Compromised Nodes', fontsize=14)
    plt.ylabel('Average Number of Safe Keys', fontsize=14)
    plt.ylim(0, 15)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(which='both', linestyle='-.', linewidth=0.5)
    plt.savefig('/Users/xingyuzhou/NoteOnGithub/Diplomarbeit/DA_Tex/figs/3_ses.pdf', dpi=300, format='pdf', bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    plot_vcs()
    plot_bos()
    plot_ses()
