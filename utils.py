import os
import re
import time
import random
import hashlib
from multiprocessing import cpu_count
from tandems import _tandems

try:
    from urllib.parse import unquote
except ImportError:
    from urlparse import unquote

UNK_ID = 0

SQL_RESERVED_WORDS = []
with open("./SQL_reserved_words.txt", "rb") as f:
    for line in f:
        if line[0] != bytes(b"#")[0]:
            SQL_RESERVED_WORDS.append(line.replace(b"\n", b""))


def oracle_q_encoder(row):
    def dashrepl(m):
        if m.group(0):
            return " q"+re.sub(r"[ |!|\'|\"|\;|\:|\(|\)|\{|\}]", "", m.group(0))+"q "
        else:
            return ""
    row = re.sub(r"nq\'.*\'", dashrepl, row)
    row = re.sub(r"nQ\'.*\'", dashrepl, row)
    row = re.sub(r"Nq\'.*\'", dashrepl, row)
    row = re.sub(r"NQ\'.*\'", dashrepl, row)
    row = re.sub(r"q\'.*\'", dashrepl, row)
    row = re.sub(r"Q\'.*\'", dashrepl, row)
    return row


def mysql_n_encoder(row):
    def dashrepl(m):
        if m.group(0):
            return " N"+re.sub(r"[ |!|\'|\"|\;|\:|\(|\)|\{|\}]", "", m.group(0))+"N "
        else:
            return ""
    row = re.sub(r"[N|n]\'.*\'", dashrepl, row)
    return row


def postgres_q_encoder(row):
    def dashrepl(m):
        if m.group(0):
            return " "+re.sub(r"[ |!|\'|\"|\;|\:|\(|\)|\{|\}]", "", m.group(0))+" "
        else:
            return ""
    row = re.sub(r"\$.*\$", dashrepl, row)
    return row      


def escape_byte(b):
    if b in [
        bytearray(b"-"), 
        bytearray(b"+"), 
        bytearray(b"."), 
        bytearray(b"?"), 
        bytearray(b"*"), 
        bytearray(b"}"), 
        bytearray(b"{"), 
        bytearray(b"["), 
        bytearray(b"]"),
        bytearray(b")"),
        bytearray(b"("),
        bytearray(b"\\"),
        bytearray(b"^"),
        bytearray(b"$")]:
        return b'\\'+b
    else:
        return b


"""
    br_sub - byte range sub
    Args:
        ba - bytearray
        repl - bytearray for replay
        start_b - start byte like 0x3c
        end_b - end byte like 0x3e. Have to be greather than start_b
        multy_replave - True or False. If True all duplicate tokens will be replaced to one token. 
    return:
        bytearray with repl insted byte from range from start_b to end_b
"""
def br_sub(ba, repl, start_b, end_b=None, multy_replace=True):
    if end_b == None:
        end_b = start_b

    reg_tokens = []
    for i in range(start_b, end_b+1):
        reg_tokens.append(escape_byte(bytearray([i])))
    
    if len(reg_tokens) > 1:
        regex = b"|".join(reg_tokens)
        if multy_replace:
            regex = b"["+regex+b"]+"
    else:
        regex = bytes(reg_tokens[0])
        if multy_replace:
            regex = regex+b"+"
    regex = re.compile(regex)
    ba = re.sub(regex, b" "+repl+b" ", ba)
    return ba


def replace_tokens(row, multy_replace=True, replace_unk=True, strict_unquote=False):
    row = row.lower()
    row = re.sub(re.compile(rb"\_"), b" UNDRSCR ", row)
    row = br_sub(row, b"BT_CLNS", 0x3a, 0x3b, multy_replace=multy_replace)
    row = br_sub(row, b"BT_CMPRSN", 0x3c, 0x3e, multy_replace=multy_replace)
    row = br_sub(row, b"BT_SIGNS", 0x7f, 0x9f, multy_replace=multy_replace)
    row = br_sub(row, b"BT_SIGNS", 0xa1, 0xff, multy_replace=multy_replace)
    row = br_sub(row, b"BT_SPCIAL", 0x22, 0x27, multy_replace=multy_replace)
    # row = br_sub(row, b"BTOKEN_FOUR", 0x41, 0x5a, multy_replace=multy_replace) # A-Z
    # row = br_sub(row, b"BTOKEN_FOUR", 0x61, 0x7a, multy_replace=multy_replace) # a- z
    row = br_sub(row, b"BT_BAR_BRCS", 0x7b, 0x7d, multy_replace=multy_replace) 
    row = br_sub(row, b"BT_PRNTHS", 0x28, 0x29, multy_replace=multy_replace)
    row = re.sub(re.compile(rb"(--|#)+"), b" CMMNT_LINE ", row)
    row = br_sub(row, b"BT_MATH_SIGNS", 0x2b, 0x2f, multy_replace=multy_replace)
    row = br_sub(row, b"BT_BSLAH_BRCTS", 0x5b, 0x5d, multy_replace=multy_replace)
    row = br_sub(row, b" ", 0x07, 0x0d, multy_replace=multy_replace)
    row = br_sub(row, b" ", 0xa0, multy_replace=multy_replace)
    # row = br_sub(row, b"BTOKEN_TEN", 0x20, multy_replace=multy_replace) # ' '
    row = br_sub(row, b"BT_TRSH", 0x01, 0x06, multy_replace=multy_replace)
    row = br_sub(row, b"BT_TRSH", 0x0e, 0x1f, multy_replace=multy_replace)
    #row = br_sub(row, b"BTOKEN_ANOTHER_SEPARATORS", 0x5f, multy_replace=multy_replace)#_

    row = re.sub(rb'\s{2,}', b' ', row)
    row = row.decode()
    if strict_unquote:
        while row != unquote(row):
            row = unquote(row)
    row = oracle_q_encoder(row)
    row = mysql_n_encoder(row)
    row = postgres_q_encoder(row)
    def ord_256(c):
        c = ord(c)
        if c > 256:
            return 255
        else:
            return c
    row = bytearray(map(ord_256, row))

    tokens = [] #we are using lists to follow the order during parsing  
    tokens.append([re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"), b"UUID"])
    tokens.append([re.compile(rb"/\*\!"), b'X'])
    tokens.append([re.compile(rb"\?+"), b'QUSTIN'])
    tokens.append([re.compile(rb"/\*.*\*/"), b'CMMNT'])
    tokens.append([re.compile(rb"/\*.*"), b'CMMNT_PART'])
    tokens.append([re.compile(rb"@@?[a-zA-Z0-9]+"), b'VAR'])
    tokens.append([re.compile(rb"(^|(?<=[(|)| |:|\"|\'|\[|\]|\{|\}|,|.|\?|\/|=|-]))([a-z]{1})([(|)| |:|\"|\'|\[|\]|\{|\}|,|.|\?|\/|=|-]|$)"), rb"SINGLE_CHAR \3"])
    tokens.append([re.compile(rb"[0-9]+"), b'NUM'])
    tokens.append([re.compile(rb"[0-9a-z]{8,}"), b'STRING'])

    for token in tokens:
        row = re.sub(token[0], b" "+token[1]+b" ", row)

    row = br_sub(row, b"BT_NUMS", 0x30, 0x39, multy_replace=multy_replace)
    
    if replace_unk:
        row = row.split()
        new_row = []
        for w in row:
            if (w.upper() == w) or (w.upper() in SQL_RESERVED_WORDS):
                new_row.append(w)
            else:
                new_row.append(b'unk')
        row = b" ".join(new_row)

    return row


def replace_tandems(row):
    return _tandems.replace_tandems(row.decode()).encode()


def norm_len(seq, max_len, UNK_ID=UNK_ID):
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        return seq+[UNK_ID]*(max_len-len(seq))


def prepare_sen(row, multy_replace, max_seq_len):
    sen = replace_tokens(row)
    if multy_replace:
        if len(sen.split()) > max_seq_len:
            sen = b" ".join(sen.split()[:max_seq_len])
        sen = replace_tandems(sen)
    return sen


def file_as_bytes(file):
    with file:
        return file.read()


def calculate_model_hash(checkpoint_file):
    return hashlib.md5(file_as_bytes(open(checkpoint_file+".index", 'rb'))).hexdigest()


def batch(data, n=cpu_count()):
    l = len(data)
    for ndx in range(0, l, n):
        yield data[ndx:min(ndx + n, l)]

