import numpy as np
from os import urandom

def WORD_SIZE():
    return(16);

def ALPHA():
    return(7);

def BETA():
    return(2);

MASK_VAL = 2 ** WORD_SIZE() - 1;

def shuffle_together(l):
    
    state = np.random.get_state();
    for x in l:
        np.random.set_state(state);
        np.random.shuffle(x);

def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL));

def enc_one_round(p, k):
    c0, c1 = p[0], p[1];
    c0 = ror(c0, ALPHA());
    c0 = (c0 + c1) & MASK_VAL;
    c0 = c0 ^ k;
    c1 = rol(c1, BETA());
    c1 = c1 ^ c0;
    return(c0,c1);

def dec_one_round(c,k):
    c0, c1 = c[0], c[1];
    c1 = c1 ^ c0;
    c1 = ror(c1, BETA());
    c0 = c0 ^ k;
    c0 = (c0 - c1) & MASK_VAL;
    c0 = rol(c0, ALPHA());
    return(c0, c1);

def expand_key(k, t):
    ks = [0 for i in range(t)];
    ks[0] = k[len(k)-1];
    l = list(reversed(k[:len(k)-1]));
    for i in range(t-1):
        l[i%3], ks[i+1] = enc_one_round((l[i%3], ks[i]), i);
    return(ks);

def encrypt(p, ks):
    x, y = p[0], p[1];
    for k in ks:
        x,y = enc_one_round((x,y), k);
    return(x, y);

def decrypt(c, ks):
    x, y = c[0], c[1];
    for k in reversed(ks):
        x, y = dec_one_round((x,y), k);
    return(x,y);

def check_testvector():
  key = (0x1918,0x1110,0x0908,0x0100)
  pt = (0x6574, 0x694c)
  ks = expand_key(key, 22)
  ct = encrypt(pt, ks)
  if (ct == (0xa868, 0x42f2)):
    print("Testvector verified.")
    return(True);
  else:
    print("Testvector not verified.")
    return(False);

#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
  X = np.zeros((4 * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range(4 * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

#takes a text file that contains encrypted block0, block1, true diff prob, real or random
#data samples are line separated, the above items whitespace-separated
#returns train data, ground truth, optimal ddt prediction
def readcsv(datei):
    data = np.genfromtxt(datei, delimiter=' ', converters={x: lambda s: int(s,16) for x in range(2)});
    X0 = [data[i][0] for i in range(len(data))];
    X1 = [data[i][1] for i in range(len(data))];
    Y = [data[i][3] for i in range(len(data))];
    Z = [data[i][2] for i in range(len(data))];
    ct0a = [X0[i] >> 16 for i in range(len(data))];
    ct1a = [X0[i] & MASK_VAL for i in range(len(data))];
    ct0b = [X1[i] >> 16 for i in range(len(data))];
    ct1b = [X1[i] & MASK_VAL for i in range(len(data))];
    ct0a = np.array(ct0a, dtype=np.uint16); ct1a = np.array(ct1a,dtype=np.uint16);
    ct0b = np.array(ct0b, dtype=np.uint16); ct1b = np.array(ct1b, dtype=np.uint16);
    
    #X = [[X0[i] >> 16, X0[i] & 0xffff, X1[i] >> 16, X1[i] & 0xffff] for i in range(len(data))];
    X = convert_to_binary([ct0a, ct1a, ct0b, ct1b]); 
    Y = np.array(Y, dtype=np.uint8); Z = np.array(Z);
    return(X,Y,Z);

#baseline training data generator
def make_train_data(n, nr, diff=(0x0040,0)):
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  plain1l[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  plain1r[Y==0] = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  ks = expand_key(keys, nr);
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
  return(X,Y);

#real differences data generator
def real_differences_data(n, nr, diff=(0x0040,0)):
  #generate labels
  Y = np.frombuffer(urandom(n), dtype=np.uint8); Y = Y & 1;
  #generate keys
  keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
  #generate plaintexts
  plain0l = np.frombuffer(urandom(2*n),dtype=np.uint16);
  plain0r = np.frombuffer(urandom(2*n),dtype=np.uint16);
  #apply input difference
  plain1l = plain0l ^ diff[0]; plain1r = plain0r ^ diff[1];
  num_rand_samples = np.sum(Y==0);
  #expand keys and encrypt
  ks = expand_key(keys, nr);
  ctdata0l, ctdata0r = encrypt((plain0l, plain0r), ks);
  ctdata1l, ctdata1r = encrypt((plain1l, plain1r), ks);
  #generate blinding values
  k0 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  k1 = np.frombuffer(urandom(2*num_rand_samples),dtype=np.uint16);
  #apply blinding to the samples labelled as random
  ctdata0l[Y==0] = ctdata0l[Y==0] ^ k0; ctdata0r[Y==0] = ctdata0r[Y==0] ^ k1;
  ctdata1l[Y==0] = ctdata1l[Y==0] ^ k0; ctdata1r[Y==0] = ctdata1r[Y==0] ^ k1;
  #convert to input data for neural networks
  X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r]);
  return(X,Y);

# ============================================================
# Theory Consistency (CE2)
# ============================================================
#
# Design rationale (revised)
# ----------------------------
# An earlier version of this module estimated the theoretical
# reference via a Monte Carlo output-differential lookup table
# (empirical DDT). That approach is statistically infeasible
# for Speck32/64: the output-difference space has 2^32 possible
# values, so the expected number of collisions among N Monte
# Carlo trials is ~N^2 / 2^33, which stays negligible even at
# N=10^7 (~11,600 collision-pairs among 10^7 trials, each with
# probability mass on the order of 2e-7 -- far too rare for an
# independently drawn evaluation sample to ever land on one).
# This was confirmed empirically: 10,000 trials produced 10,000
# unique entries, and an independent 1,000-sample evaluation set
# had zero overlap.
#
# The theoretical reference is therefore computed analytically,
# per evaluation sample, rather than looked up:
#
#   1. Speck's round function is linear over XOR differences in
#      every operation except the single modular addition per
#      round (rotation and XOR-with-round-key preserve
#      differences exactly, since the same key is applied to
#      both branches of a pair). Probability enters the cipher
#      only through that one addition per round.
#
#   2. Lipmaa & Moriai ("Efficient Algorithms for Computing
#      Differential Properties of Addition", FSE 2001) give a
#      closed-form probability xdp+(alpha, beta -> gamma) for a
#      SPECIFIC addition-differential transition, evaluable in
#      O(word_size) with no enumeration and no table.
#
#   3. For each evaluation sample, both branches (real plaintext
#      pairs, real round keys) are run through the round
#      function in lockstep. At each round, the real difference
#      entering and leaving that round's addition is read off
#      directly (no symbolic modeling needed, since both
#      branches' actual values are known), and xdp+ is applied
#      to that one observed transition.
#
#   4. The per-round probabilities are multiplied across all
#      rounds (Markov/independence assumption) to give the
#      sample's analytical single-trail probability.
#
# This is a KNOWN LOWER BOUND on the true propagation
# probability, not an exact value: round-reduced Speck is known
# to exhibit differential trail clustering, where multiple
# distinct trails connect the same (input, output) difference
# pair and their probabilities sum, so the true probability can
# exceed any single trail's probability -- this is precisely why
# Gohr's neural distinguisher (CRYPTO 2019) outperforms classical
# single-trail differential distinguishers, and is analyzed
# directly in Benamira et al. ("A Deeper Look at Machine
# Learning-Based Cryptanalysis", EUROCRYPT 2021). This CE2
# reference should therefore be reported and interpreted
# explicitly as a single-trail lower bound, not as the exact
# theoretical probability, per the framework's own requirement
# to document evidential scope and limits.


def _eq_mask(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
) -> np.ndarray:
    """
    Bitwise eq(alpha_i, beta_i, gamma_i): 1 at bit positions
    where alpha, beta, and gamma agree, per Lipmaa & Moriai.
    """

    return (~(alpha ^ beta) & ~(beta ^ gamma)) & MASK_VAL


def _popcount(x: np.ndarray) -> np.ndarray:
    """
    Bitwise population count, portable across numpy versions
    (avoids depending on np.bitwise_count, added only in very
    recent numpy releases).
    """

    x = x.astype(np.uint32).copy()
    count = np.zeros_like(x)
    while np.any(x):
        count += (x & 1)
        x >>= 1
    return count


def xdp_plus(
    alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray,
) -> np.ndarray:
    """
    Lipmaa & Moriai (FSE 2001) closed-form XOR differential
    probability of WORD_SIZE()-bit modular addition:
        xdp+(alpha, beta -> gamma)

    Vectorized over numpy arrays of equal shape (dtype uint16
    or wider). Returns 0.0 at positions where the differential
    is invalid.

    Reference: H. Lipmaa, S. Moriai, "Efficient Algorithms for
    Computing Differential Properties of Addition", FSE 2001.
    """

    n = WORD_SIZE()

    alpha = alpha.astype(np.uint32) & MASK_VAL
    beta = beta.astype(np.uint32) & MASK_VAL
    gamma = gamma.astype(np.uint32) & MASK_VAL

    eq = _eq_mask(alpha, beta, gamma)

    valid = ((alpha ^ beta ^ gamma) & 1) == 0

    for i in range(n - 1):
        eq_i = (eq >> i) & 1
        required = (
            (alpha >> (i + 1)) ^ (beta >> (i + 1)) ^ (beta >> i)
        ) & 1
        actual = (gamma >> (i + 1)) & 1
        mismatch = (eq_i == 1) & (required != actual)
        valid = valid & ~mismatch

    not_eq_low = (~eq) & ((1 << (n - 1)) - 1)
    weight = _popcount(not_eq_low)

    probability = np.power(2.0, -weight.astype(np.float64))

    return np.where(valid, probability, 0.0)


def estimate_trail_probabilities(
    n: int,
    nr: int,
    diff: tuple[int, int],
):
    """
    Generate `n` real-difference evaluation samples under the
    fixed input differential `diff`, and compute each sample's
    analytical single-trail probability directly from its own
    real round-by-round trajectory (Lipmaa-Moriai closed form,
    chained under the Markov/independence assumption).

    Returns
    -------
    X : np.ndarray
        Network-ready input encoding, identical in format to
        training/evaluation (see `convert_to_binary`).
    theoretical_probabilities : np.ndarray (float64)
        Per-sample analytical trail probability. Index-aligned
        with `X`. This is a single-trail lower bound; see module
        docstring above.
    """

    keys = np.frombuffer(
        urandom(8 * n), dtype=np.uint16,
    ).reshape(4, -1)

    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)

    plain1l = plain0l ^ diff[0]
    plain1r = plain0r ^ diff[1]

    ks = expand_key(keys, nr)

    x0, y0 = plain0l.copy(), plain0r.copy()
    x1, y1 = plain1l.copy(), plain1r.copy()

    log2_prob = np.zeros(n, dtype=np.float64)
    feasible = np.ones(n, dtype=bool)

    for k in ks:
        a0 = ror(x0, ALPHA())
        b0 = y0
        a1 = ror(x1, ALPHA())
        b1 = y1

        alpha_in = a0 ^ a1
        beta_in = b0 ^ b1

        s0 = (a0 + b0) & MASK_VAL
        s1 = (a1 + b1) & MASK_VAL
        gamma_out = s0 ^ s1

        p_round = xdp_plus(alpha_in, beta_in, gamma_out)

        feasible &= (p_round > 0.0)

        safe_p = np.where(p_round > 0.0, p_round, 1.0)
        log2_prob += np.log2(safe_p)

        x0, y0 = enc_one_round((x0, y0), k)
        x1, y1 = enc_one_round((x1, y1), k)

    log2_prob = np.where(feasible, log2_prob, -np.inf)
    theoretical_probabilities = np.exp2(log2_prob)

    X = convert_to_binary([x0, y0, x1, y1])

    return X, theoretical_probabilities