"""
Named channel conditions shared by plot_llr.py and evaluate.py.

Grammar (tokens joined by '_'):
    awgn | fade<N>            N = symbols per fading block (fade1 = fast fading, fade8 = one C=16 token
                              / half a C=32 token, fade16 = one C=32 token)
    over<X> | under<X>        receiver SNR estimate error of +X / -X dB  (imperfect CSI, AWGN or fading)
    h<Y>                      channel-estimate error variance Y, h_hat = h + CN(0, Y)  (fading only)
    rnd<X>                    receiver SNR estimate error drawn ~ U(-X, X) dB per transmission

Examples: awgn, awgn_over3, awgn_under3, awgn_rnd3, fade8, fade8_h0.1, fade16_h0.2_under2
"""
import re


def parse_condition(name):
    c = dict(name=name, channel="awgn", fade_block=0, snr_err_db=0.0, h_err_var=0.0, snr_err_rand=0.0)
    for tok in name.split("_"):
        if tok == "awgn":
            c["channel"] = "awgn"
        elif tok.startswith("fade"):
            c["channel"], c["fade_block"] = "blockfading", int(tok[4:])
        elif tok.startswith("over"):
            c["snr_err_db"] = float(tok[4:])
        elif tok.startswith("under"):
            c["snr_err_db"] = -float(tok[5:])
        elif tok.startswith("rnd"):
            c["snr_err_rand"] = float(tok[3:])
        elif tok.startswith("h") and re.match(r"^h[0-9.]+$", tok):
            c["h_err_var"] = float(tok[1:])
        else:
            raise ValueError("unknown token %r in condition %r" % (tok, name))
    if c["h_err_var"] > 0 and c["channel"] == "awgn":
        raise ValueError("h<Y> only applies to fading conditions (%s)" % name)
    return c


def describe(c):
    s = "AWGN" if c["channel"] == "awgn" else ("fast Rayleigh" if c["fade_block"] == 1 else "block Rayleigh (%d sym)" % c["fade_block"])
    if c["snr_err_db"]:
        s += ", SNR est. %+g dB" % c["snr_err_db"]
    if c["snr_err_rand"]:
        s += ", SNR est. +-%g dB" % c["snr_err_rand"]
    if c["h_err_var"]:
        s += ", h est. err var %g" % c["h_err_var"]
    if not (c["snr_err_db"] or c["snr_err_rand"] or c["h_err_var"]):
        s += ", perfect CSI"
    return s
