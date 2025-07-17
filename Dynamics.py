
# -----------------------------------------------------------------------------
#  DNA Sequence Dynamical System Blind-Lab Demo
# -----------------------------------------------------------------------------

import json
import numpy as np
from scipy.spatial.distance import euclidean
from openai import OpenAI
import random

# --- Import Properties and Samplers ---
from Properties import Properties, Samplers, Helpers, O1_PROPERTIES


# -----------------------------------------------------------------------------
#  Configuration
# -----------------------------------------------------------------------------
client = OpenAI(api_key="sk-...")   # Replace with your own key

ALPHABET = ["A", "G", "C", "T"]
SEQ_LENGTH = 30
N_ROUNDS = 6        # max experiments
MAX_STEPS = 20      # per experiment

MODEL_NAME = "gpt-4.1-mini"
TEMPERATURE = 0.5
# MODEL_NAME = "o4-mini"
# TEMPERATURE = None

# -----------------------------------------------------------------------------
#  Blind-Lab Prompt
# -----------------------------------------------------------------------------
BLIND_LAB_PROMPT = """
You are an experimental scientist studying an *unknown* dynamical process that acts on DNA-like sequences.

Environment:
• Sequence length: 30 (alphabet {A,G,C,T})
• You may run *experiments*. One experiment consists of:
   1) Propose an initial sequence (30 chars).
   2) Select a subset of MEASURABLES (listed below) to analyze.
   3) Choose number_of_steps ∈ {1,…,20}.
The lab will return the full trajectory with the requested measurables at each step.

MEASURABLES (random order):
  • gc_content – fraction of G/C bases
  • palindrome_score – fraction of matching bases when the sequence is compared to its reverse complement (formed by swapping A↔T and C↔G and reversing).
  • hamming_distance – number of differences to a hidden reference
  • entropy_score – diversity of 2-mers within the sequence.

Goal:
Infer and explain how the dynamical process behaves, including:
  (i) qualitative description of long-term behaviors or attractors,
  (ii) the rule or objective it optimizes,
  (iii) supporting evidence from your experiments.

Interaction Protocol:
Reply in JSON with either:
  {"type": "experiment", "sequence": <30-mer>, "measurables": [<list>], "steps": <int>, "reasoning": "brief hypothesis/plan/reasoning"}
or
  {"type": "report", "hypothesis": "final hypothesis", "evidence": "supporting evidence"}
In each experiment response, include a brief hypothesis, plan, or reasoning behind your experiment.
If this is your final experiment round, please reveal your final answer based on the goal.
Wait for the lab to respond before the next message.
"""

# -----------------------------------------------------------------------------
#  Dynamical System Class: h : Seq → List[Seq]
# -----------------------------------------------------------------------------

class DynamicalSystemFramework:
    def __init__(self, evolution_properties, fixed_points, basins='inverse_square', inv_covs=None):
        self.evolution_properties = evolution_properties
        self.fixed_points = fixed_points
        self.inv_covs = inv_covs
        self.basin_functions = {
            'inverse_square': self._inverse_square_basin,
            'gaussian': self._gaussian_basin,
            'anisotropic': self._anisotropic_basin
        }
        if basins not in self.basin_functions:
            raise ValueError(f"Unknown basin type: {basins}. Choose from {list(self.basin_functions.keys())}.")
        self.basins_of_attraction = self.basin_functions[basins]

    def _inverse_square_basin(self, distances):
        return [1.0 / (d**2 + 0.01) for d in distances]

    def _gaussian_basin(self, distances, sigma=0.1):
        return [np.exp(-(d**2) / (2 * sigma**2)) for d in distances]

    def _anisotropic_basin(self, prop_vec, fixed_points=None, inv_covs=None):
        if fixed_points is None:
            fixed_points = self.fixed_points
        if inv_covs is None:
            inv_covs = self.inv_covs
        if inv_covs is None:
            raise ValueError("inv_covs must be provided for anisotropic basins.")
        strengths = []
        for fp, P in zip(fixed_points, inv_covs):
            delta = prop_vec - np.array(fp)
            strengths.append(np.exp(-delta.T @ P @ delta / 2))
        return strengths

    def get_property_vector(self, sequence):
        return np.array([func(sequence) for func in self.evolution_properties.values()])

    def distances_to_fixed_points(self, sequence):
        prop_vector = self.get_property_vector(sequence)
        return [euclidean(prop_vector, point) for point in self.fixed_points]

    def calculate_attractor_weights(self, sequence):
        prop_vector = self.get_property_vector(sequence)
        if self.basins_of_attraction == self._anisotropic_basin:
            basin_strengths = self.basins_of_attraction(prop_vec=prop_vector)
        else:
            distances = self.distances_to_fixed_points(sequence)
            basin_strengths = self.basins_of_attraction(distances)
        total = sum(basin_strengths)
        if total == 0:
            return [1.0 / len(basin_strengths)] * len(basin_strengths)
        return [strength / total for strength in basin_strengths]

    def target_direction(self, sequence):
        weights = self.calculate_attractor_weights(sequence)
        current = self.get_property_vector(sequence)
        target = np.zeros_like(current)
        for i, fixed_point in enumerate(self.fixed_points):
            target += weights[i] * np.array(fixed_point)
        return target - current

    def evolve_sequence(self, sequence, steps=1, intensity=0.1):
        seq = sequence
        for _ in range(steps):
            direction = self.target_direction(seq)
            seq = self._apply_directed_mutations(seq, direction, intensity)
        return seq

    def _best_positions(self, sequence, direction, k=3):
        base_vec = self.get_property_vector(sequence)
        scores = []
        for j in range(len(sequence)):
            best_delta = 0.0
            for n in ALPHABET:
                if n == sequence[j]:
                    continue
                trial = sequence[:j] + n + sequence[j+1:]
                delta = self.get_property_vector(trial) - base_vec
                best_delta = max(best_delta, np.dot(delta, direction))
            scores.append(best_delta)
        return np.argsort(scores)[-k:]

    def _apply_directed_mutations(self, sequence, direction, intensity, k=3):
        seq_list = list(sequence)
        n_mutations = max(1, int(intensity * len(sequence)))
        top_positions = self._best_positions(sequence, direction, k=n_mutations)
        for pos in top_positions:
            best_nucleotide = seq_list[pos]
            best_improvement = -float('inf')
            for n in ALPHABET:
                if n == seq_list[pos]:
                    continue
                trial = seq_list[:]
                trial[pos] = n
                trial_seq = ''.join(trial)
                delta = self.get_property_vector(trial_seq) - self.get_property_vector(sequence)
                improvement = np.dot(delta, direction)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_nucleotide = n
            seq_list[pos] = best_nucleotide
        return ''.join(seq_list)

    def analyze_sequence(self, sequence):
        prop_vector = self.get_property_vector(sequence)
        distances = self.distances_to_fixed_points(sequence)
        nearest_idx = np.argmin(distances)
        return {
            'properties': dict(zip(self.evolution_properties.keys(), prop_vector)),
            'nearest_fixed_point': self.fixed_points[nearest_idx],
            'distance_to_nearest': distances[nearest_idx],
            'attractor_weights': self.calculate_attractor_weights(sequence)
        }


def sharp_basin_function(distances, sharpness=10, threshold=0.3):
    """
    Creates sharp transitions between basins of attraction.
    
    Args:
        distances: List of distances to fixed points
        sharpness: Controls how sharp the transition is (higher = sharper)
        threshold: Distance threshold where attraction starts to weaken significantly
        
    Returns:
        List of attraction strengths
    """
    # Sigmoid function creates sharp transitions
    return [1.0 / (1 + np.exp(sharpness * (d - threshold))) for d in distances]


# -----------------------------------------------------------------------------
#  Evolution Utility: Track Sequence and Properties
# -----------------------------------------------------------------------------

def evolve_sequence_with_properties(sys, sequence, props, steps):
    seqs, prop_vals = [sequence], []
    if props:
        prop_vals.append([round(f(sequence), 3) for f in props])
    for _ in range(steps):
        sequence = sys.evolve_sequence(sequence)
        seqs.append(sequence)
        prop_vals.append([round(f(sequence), 3) for f in props])
    return seqs, prop_vals

# -----------------------------------------------------------------------------
#  Main LLM-driven Blind-Lab Loop
# -----------------------------------------------------------------------------

def run_blind_lab(dynamical_system, query_properties):
    conversation = [{"role": "system", "content": BLIND_LAB_PROMPT}]
    print("🔧 SYSTEM PROMPT:")
    print(BLIND_LAB_PROMPT)
    print("")
    for round_idx in range(N_ROUNDS):
        if round_idx == N_ROUNDS - 1:
            conversation.append({
                "role": "system",
                "content": "This is the final round. Please provide your final answer based on the goal."
            })
        if TEMPERATURE is not None:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                temperature=TEMPERATURE
            )
        else:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation
            )
        llm_msg = resp.choices[0].message.content.strip()
        try:
            experiment = json.loads(llm_msg)
            pretty_response = (
                "----- Experiment Response -----\n"
                f"Type: {experiment.get('type')}\n"
                f"Sequence: {experiment.get('sequence')}\n"
                "Measurables:\n" +
                "\n".join(f"  - {m}" for m in experiment.get('measurables', [])) + "\n" +
                f"Steps: {experiment.get('steps')}\n"
                f"Reasoning: {experiment.get('reasoning')}\n"
                "-------------------------------"
            )
            print("🤖 LLM Response:")
            print(pretty_response)
        except Exception as e:
            print("Could not parse JSON from LLM response.", e)
            break
        if experiment.get("type") == "report":
            pretty_report = (
                "----- Final Report -----\n"
                f"Hypothesis: {experiment.get('hypothesis', 'N/A')}\n"
                f"Evidence: {experiment.get('evidence', 'N/A')}\n"
                "------------------------"
            )
            print("Final Report from LLM:")
            print(pretty_report)
            return experiment
        seq = experiment["sequence"].upper()
        meas = experiment["measurables"]
        steps = experiment["steps"]
        props = [query_properties[p] for p in meas if p in query_properties]
        seqs, vals = evolve_sequence_with_properties(sys=dynamical_system,
                                                     sequence=seq,
                                                     props=props,
                                                     steps=steps)
        print("RESULTS:")
        for step_idx, (sequence, prop_values) in enumerate(zip(seqs, vals)):
            properties = dict(zip(meas, prop_values))
            print(f"Step {step_idx}: {sequence}, Properties: {properties}")
        lab_reply = {
            "type": "lab_results",
            "trajectory": seqs,
            "measurements": {
                name: [round(v, 3) for v in series]
                for name, series in zip(meas, zip(*vals))
            }
        }
        conversation.append({"role": "user", "content": json.dumps(lab_reply)})
        print("")
    print("No report produced after max rounds.")
    return None

# -----------------------------------------------------------------------------
#  Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Evolution properties (drives the dynamics) ---
    evolution_properties = {
        "gc_content":           O1_PROPERTIES["gc"],
        # "alternation_score":    O1_PROPERTIES["alt"],
        "palindrome_score":     O1_PROPERTIES["pal"],
        # "hamming_distance":     O1_PROPERTIES["ham_ref"],
        # "entropy_score":            O1_PROPERTIES["entropy"],
    }

    # --- Properties available for LLM to query (keep comments for clarity) ---
    evolution_properties_available_to_query = {
        "gc_content":           O1_PROPERTIES["gc"],
        # "alternation_score":    O1_PROPERTIES["alt"],
        "palindrome_score":     O1_PROPERTIES["pal"],
        "hamming_distance":     O1_PROPERTIES["ham_ref"],
        "entropy_score":            O1_PROPERTIES["entropy"],
    }

    # these are the measurables the LLM may query
    query_properties = evolution_properties_available_to_query.copy()

    fixed_points = [(0.8, 0.8), (0.4, 0.2)]

    ds = DynamicalSystemFramework(
        evolution_properties=evolution_properties,
        fixed_points=fixed_points,
        basins='gaussian'
    )

    result = run_blind_lab(dynamical_system=ds, query_properties=query_properties)
