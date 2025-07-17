
# -----------------------------------------------------------------------------
#  DNA Sequence Property Discovery & Inverse Sampling Demo
# -----------------------------------------------------------------------------

import random
import math
import collections
import numpy as np
from openai import OpenAI



# --- OpenAI API Client ---
client = OpenAI(api_key="sk-...") # Replace with your own key



# --- Global Constants ---
random.seed(541)  # For reproducibility
ALPHABET = ["A", "G", "C", "T"]
SEQ_LENGTH = 10  # Length of the DNA sequence (note that longer sequences may require much longer sampling times)
N_SAMPLES_O2 = 10  # Number of samples when sampling properties
N_ROUNDS = 6  # Number of interaction rounds
REF_SEQ = "".join(random.choices(ALPHABET, k=SEQ_LENGTH))  # Reference sequence for Hamming distance calculations



# --- Pairwise Interaction Matrix ---
interaction_matrix = {
    ('A', 'T'): -1.5, ('T', 'A'): -1.6,
    ('G', 'C'): -1.0, ('C', 'G'): -1.0,
    ('A', 'G'): 0.5,  ('G', 'A'): 0.5,
    ('C', 'T'): 0.5,  ('T', 'C'): 0.5,
}



# --- LLM Model Settings ---
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.3
# MODEL_NAME = "o4-mini"  # For deterministic reasoning
# TEMPERATURE = None



# -----------------------------------------------------------------------------
#  Sequence Properties  f : Seq → [0,1]
# -----------------------------------------------------------------------------

class Properties:
    def __init__(self, seq_length=SEQ_LENGTH, alphabet=ALPHABET, ref_seq=REF_SEQ):
        self.seq_length = seq_length
        self.alphabet = alphabet
        self.ref_seq = ref_seq

    def gc_content(self, seq):
        return (seq.count("G") + seq.count("C")) / self.seq_length

    def alternation_score(self, seq):
        changes = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        return changes / (self.seq_length - 1)

    def masked_alternation_score(self, seq, k=2):
        # Alternation score on positions i % k == 0
        kept = range(0, len(seq), k)
        subseq = [seq[i] for i in kept]
        if len(subseq) <= 1:
            return 0.0
        changes = sum(1 for a, b in zip(subseq, subseq[1:]) if a != b)
        return changes / (len(subseq) - 1)

    def motif_score(self, seq, motif="GTA"):
        windows = self.seq_length - len(motif) + 1
        hits = sum(1 for i in range(windows) if seq[i : i + len(motif)] == motif)
        return hits / windows if windows else 0.0

    def reverse_complement(self, seq):
        comp = str.maketrans("ACGT", "TGCA")
        return seq.translate(comp)[::-1]

    def palindromicity(self, seq):
        rc = self.reverse_complement(seq)
        matches = sum(1 for a, b in zip(seq, rc) if a == b)
        return matches / self.seq_length

    def kmer_entropy(self, seq, k=2):
        kmers = [seq[i : i + k] for i in range(self.seq_length - k + 1)]
        counts = collections.Counter(kmers)
        probs = [c / len(kmers) for c in counts.values()]
        h = -sum(p * math.log2(p) for p in probs)
        h_max = math.log2(len(self.alphabet) ** k)
        return h / h_max if h_max else 0.0

    def hamming_from_ref(self, seq):
        dist = sum(1 for a, b in zip(seq, self.ref_seq) if a != b)
        return dist / self.seq_length

    # Note that interaction_energy is not properly scaled to [0,1] as most of the properties.
    def interaction_energy(self, seq, interaction_matrix=interaction_matrix):
        total_energy = 0.0
        for i in range(len(seq) - 1):
            total_energy += interaction_matrix.get((seq[i], seq[i + 1]), 0.0)
        return -total_energy / (len(seq) - 1)

    def local_propensity(self, seq, window=3, weights={'A': 1.0/1.7, 'C': 0.0, 'G': 0.2/1.7, 'T': 0.5/1.7}):
        scores = []
        for i in range(0, len(seq) - window + 1, window):
            subseq = seq[i : i + window]
            score = sum(weights.get(base, 0.0) for base in subseq) / window
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

# --- Property name to method mapping ---
O1_PROPERTIES = {
    "gc": Properties().gc_content,
    "alt": Properties().alternation_score,
    "motif": Properties().motif_score,
    "pal": Properties().palindromicity,
    "entropy": Properties().kmer_entropy,
    "ham_ref": Properties().hamming_from_ref,
    "interaction_energy": Properties().interaction_energy,
    "local_propensity": Properties().local_propensity,
    "masked_alt": Properties().masked_alternation_score,
}


# -----------------------------------------------------------------------------
#  Inverse Samplers  g : (property, value) → List[Seq]
# -----------------------------------------------------------------------------

class Samplers:
    def __init__(self, properties, n_samples=N_SAMPLES_O2):
        self.properties = properties
        self.n_samples = n_samples

    def sample_gc(self, target, length=SEQ_LENGTH, tol=0.1):
        target = min(1, max(0, target))
        seqs = []
        while len(seqs) < self.n_samples:
            k = np.random.binomial(length, target)
            if abs(k / length - target) > tol:
                continue
            pos = set(random.sample(range(length), k))
            seqs.append(
                "".join(random.choice("GC") if i in pos else random.choice("AT") for i in range(length))
            )
        return seqs

    def sample_alt(self, target):
        seqs = []
        for _ in range(self.n_samples):
            s = [random.choice(self.properties.alphabet)]
            for _ in range(self.properties.seq_length - 1):
                if random.random() < target:
                    s.append(random.choice([b for b in self.properties.alphabet if b != s[-1]]))
                else:
                    s.append(s[-1])
            seqs.append("".join(s))
        return seqs

    def sample_masked_alt(self, target, k=2, tol=0.1):
        alphabet = self.properties.alphabet
        seq_length = self.properties.seq_length
        seqs = []
        m = (self.properties.seq_length + k - 1) // k
        max_attempts = 10000
        for _ in range(self.n_samples):
            for _attempt in range(max_attempts):
                masked = [random.choice(self.properties.alphabet)]
                for _ in range(m - 1):
                    if random.random() < target:
                        masked.append(random.choice([b for b in self.properties.alphabet if b != masked[-1]]))
                    else:
                        masked.append(masked[-1])
                score = self.properties.masked_alternation_score("".join(masked), k=1)
                if abs(score - target) > tol:
                    continue
                seq = [random.choice(alphabet) for _ in range(seq_length)]
                for idx, sym in zip(range(0, seq_length, k), masked):
                    seq[idx] = sym
                seqs.append("".join(seq))
                break
            else:
                raise RuntimeError("Mask-alt sampler hit max_attempts without success.")
        return seqs

    def sample_motif(self, target, motif="GTA"):
        windows = self.properties.seq_length - len(motif) + 1
        hits = int(round(target * windows))
        seqs = []
        for _ in range(self.n_samples):
            s = list(self.random_sequence())
            positions = random.sample(range(windows), k=hits)
            for pos in positions:
                s[pos : pos + len(motif)] = list(motif)
            seqs.append("".join(s))
        return seqs

    def sample_palindromic(self, target):
        def improve(seq):
            idx = random.randrange(self.properties.seq_length)
            rc = self.properties.reverse_complement(seq)
            if seq[idx] != rc[idx]:
                seq = seq[:idx] + rc[idx] + seq[idx + 1:]
            return seq
        seqs = []
        for _ in range(self.n_samples):
            s = Helpers.random_sequence()
            for _ in range(50):
                if self.properties.palindromicity(s) >= target:
                    break
                s = improve(s)
            seqs.append(s)
        return seqs

    def sample_entropy(self, target, tol=0.1, k=2, n=N_SAMPLES_O2, L=SEQ_LENGTH, max_iter=2000, T0=1.0, cooling=0.995):
        
        # the args are: target entropy, tolerance, k-mer size, number of samples, sequence length,
        #               maximum iterations, initial temperature, and cooling factor respectively
        
        def energy(h):
            return abs(h - target)
        seqs = []
        for _ in range(n):
            s = "".join(random.choice(self.properties.alphabet) for _ in range(L))
            h = self.properties.kmer_entropy(s, k)
            best_s, best_e = s, energy(h)
            T = T0
            for _ in range(max_iter):
                if best_e <= tol:
                    break
                i = random.randrange(L)
                c_new = random.choice([c for c in self.properties.alphabet if c != s[i]])
                s2 = s[:i] + c_new + s[i+1:]
                h2 = self.properties.kmer_entropy(s2, k)
                delta = energy(h2) - energy(h)
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    s, h = s2, h2
                    if energy(h) < best_e:
                        best_s, best_e = s, energy(h)
                T *= cooling
            seqs.append(best_s)
        return seqs

    def sample_hamming(self, target, ref=REF_SEQ):
        flips = int(round(target * self.properties.seq_length))
        idxs = list(range(self.properties.seq_length))
        seqs = []
        for _ in range(self.n_samples):
            chosen = random.sample(idxs, flips)
            s = list(ref)
            for i in chosen:
                s[i] = random.choice([b for b in self.properties.alphabet if b != s[i]])
            seqs.append("".join(s))
        return seqs

    def sample_interaction_energy(self, target, tol=0.2, max_trials=10000):
        results = []
        for _ in range(max_trials):
            seq = ''.join(random.choices(self.properties.alphabet, k=self.properties.seq_length))
            e = self.properties.interaction_energy(seq, interaction_matrix)
            if abs(e - target) <= tol:
                results.append(seq)
            if len(results) >= self.n_samples:
                break
        return results

    def sample_local_propensity(self, target, tol=0.1):
        seqs = []
        if target > 0.8:
            target = 0.8
            tol = 0.2
            tries_per_sample = 10000
        else:
            tries_per_sample = 1000
        for _ in range(self.n_samples):
            ref = ''.join(random.choices(list(self.properties.alphabet), k=self.properties.seq_length))
            for _ in range(tries_per_sample):
                s = list(ref)
                n_mutations = random.randint(0, self.properties.seq_length // 2)
                idxs = random.sample(range(self.properties.seq_length), n_mutations)
                for i in idxs:
                    s[i] = random.choice([b for b in self.properties.alphabet if b != s[i]])
                candidate = ''.join(s)
                score = self.properties.local_propensity(candidate)
                if abs(score - target) <= tol:
                    seqs.append(candidate)
                    break
            else:
                print("Warning: Could not find suitable sequence after many tries.")
        return seqs

# --- Property name to sampler mapping ---
O2_SAMPLERS = {
    "gc": Samplers(Properties()).sample_gc,
    "alt": Samplers(Properties()).sample_alt,
    "motif": Samplers(Properties()).sample_motif,
    "pal": Samplers(Properties()).sample_palindromic,
    "entropy": Samplers(Properties()).sample_entropy,
    "ham_ref": Samplers(Properties()).sample_hamming,
    "interaction_energy": Samplers(Properties()).sample_interaction_energy,
    "local_propensity": Samplers(Properties()).sample_local_propensity,
    "masked_alt": Samplers(Properties()).sample_masked_alt,
}


# -----------------------------------------------------------------------------
#  Helper Functions
# -----------------------------------------------------------------------------

class Helpers:
    @staticmethod
    def random_sequence():
        return ''.join(random.choices(ALPHABET, k=SEQ_LENGTH))

    @staticmethod
    def evaluate_property(seq, property_name):
        if property_name not in O1_PROPERTIES:
            raise ValueError("Unknown property: %s" % property_name)
        return round(O1_PROPERTIES[property_name](seq), 3)

def run_property_discovery_chat():
    """
    Interactive property discovery chat loop.
    """

# -----------------------------------------------------------------------------
#  Main Chat Demo
# -----------------------------------------------------------------------------

def run_property_discovery_chat():
    property_name = "gc"  # Property to discover (can change to any key in O1_PROPERTIES)
    conversation = [
        {
            "role": "system",
            "content": (
                f"You are a scientist inferring an unknown property f(s)∈[0,1] of DNA sequences s (length {SEQ_LENGTH}, alphabet {{A,G,C,T}}). "
                f"You will have exactly {N_ROUNDS} rounds of interaction.\n\n"
                "In each round, you must choose ONE of these two options:\n"
                f"1. Say 'SAMPLE {{value}}' to receive {N_SAMPLES_O2} sequences with f(s)≈{{value}}, where {{value}} is a number between 0 and 1\n"
                f"2. Provide exactly {N_SAMPLES_O2} sequences (one per line) to learn their f(s) values\n\n"
                f"After seeing all results from {N_ROUNDS} rounds, you will state your final hypothesis.\n\n"
                "Format your responses carefully, for example:\n"
                "For sampling:\nSAMPLE 0.7\n\n"
                "For providing sequences:\nACGTACGTAC\nGCGCGCGCGC\n..."
            )
        }
    ]

    for round_idx in range(N_ROUNDS):
        print(f"\n=== Round {round_idx + 1} ===")
        chat_params = {
            "model": MODEL_NAME,
            "messages": conversation,
        }
        if TEMPERATURE is not None:
            chat_params["temperature"] = TEMPERATURE

        resp = client.chat.completions.create(**chat_params)
        llm_msg = resp.choices[0].message.content
        print(f"\nLLM Response:\n{llm_msg}\n")
        conversation.append({"role": "assistant", "content": llm_msg})

        # Parse LLM response
        lines = [l.strip() for l in llm_msg.splitlines() if l.strip()]
        if any("sample" in l.lower() for l in lines):
            try:
                sample_line = next(l for l in lines if "sample" in l.lower())
                target_str = sample_line.lower().replace("sample", "").strip()
                target_value = float(target_str)
                target_value = max(0.0, min(1.0, target_value))
                samples = O2_SAMPLERS[property_name](target_value)
                print(f"Sampling sequences with target {property_name} ≈ {target_value}")
            except (ValueError, StopIteration):
                print("!!!! Invalid SAMPLE format. Generating random sequences instead.")
                samples = [Helpers.random_sequence() for _ in range(N_SAMPLES_O2)]
        else:
            valid = []
            for line in lines:
                seq = ''.join(c for c in line if c in ALPHABET)[:SEQ_LENGTH]
                if len(seq) == SEQ_LENGTH:
                    valid.append(seq)
            if len(valid) != N_SAMPLES_O2:
                print("!!! Invalid or wrong number of sequences provided. Generating random sequences instead.")
                samples = [Helpers.random_sequence() for _ in range(N_SAMPLES_O2)]
            else:
                samples = valid

        evals = "\n".join(f"{s} → {Helpers.evaluate_property(s, property_name)}" for s in samples)
        print(f"Evaluations:\n{evals}\n")
        conversation.append({"role": "user", "content": f"Here are the evaluations:\n{evals}"})

    final_resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation + [{"role": "user", "content": "Now please state your final hypothesis and a very concise justification."}],
        **({"temperature": TEMPERATURE} if TEMPERATURE is not None else {})
    )
    final_hypothesis = final_resp.choices[0].message.content
    print("=== Final Hypothesis ===\n", final_hypothesis)
    return final_hypothesis

if __name__ == "__main__":
    run_property_discovery_chat()
   