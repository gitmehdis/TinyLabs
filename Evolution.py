
# -----------------------------------------------------------------------------
#  DNA Sequence Dynamical System Framework & LLM Discovery Demo
# -----------------------------------------------------------------------------

import numpy as np
from scipy.spatial.distance import euclidean
from openai import OpenAI



# --- Import Properties and Samplers ---
from Properties import Properties, Samplers, Helpers, O1_PROPERTIES, O2_SAMPLERS


# --- OpenAI API Client ---
client = OpenAI(api_key="sk-...") # Replace with your OpenAI API key


# --- Global Constants ---
ALPHABET = ["A", "G", "C", "T"]
SEQ_LENGTH = 10
N_SAMPLES_O2 = 10
N_ROUNDS = 8



# --- LLM Model Settings ---
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0.3
# MODEL_NAME = "o4-mini"  # For deterministic reasoning
# TEMPERATURE = None





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



# -----------------------------------------------------------------------------
#  Utility: Sharp Basin Function
# -----------------------------------------------------------------------------

def sharp_basin_function(distances, sharpness=10, threshold=0.3):
    # Sigmoid function creates sharp transitions
    return [1.0 / (1 + np.exp(sharpness * (d - threshold))) for d in distances]


# -----------------------------------------------------------------------------
#  Evolution Utility: Track Sequence and Properties
# -----------------------------------------------------------------------------

def evolve_sequence_with_properties(dynamical_system, sequence, properties, steps=1, temperature=None):
    sequences = [sequence]
    if properties is not None:
        prop_vector = np.array([round(prop(sequence), 3) for prop in properties])
        properties_list = [prop_vector]
    else:
        properties_list = []
    for _ in range(steps):
        sequence = dynamical_system.evolve_sequence(sequence)
        sequences.append(sequence)
        if properties is not None:
            prop_vector = np.array([round(prop(sequence), 3) for prop in properties])
            properties_list.append(prop_vector)
    return sequences, properties_list


# -----------------------------------------------------------------------------
#  Main LLM Chat Loop
# -----------------------------------------------------------------------------

def run_dynamics_discovery_chat(dynamical_system, query_properties, n_rounds=N_ROUNDS, n_steps=7):
    conversation = [
        {
            "role": "system",
            "content": (
                f"You are a scientist studying a dynamical system over DNA sequences (length {SEQ_LENGTH}, alphabet {{A,G,C,T}}).\n\n"
                f"In each round, you may propose:\n"
                f"- An input sequence ({SEQ_LENGTH} letters using A, G, C, T)\n"
                "- A list of properties to monitor during the dynamics.\n\n"
                f"Available properties you can ask about are ONLY:\n{list(query_properties.keys())}\n\n"
                "Here are definitions of the properties:\n"
                "(1) gc_content: Fraction of G and C bases (sequence composition).\n"
                "(2) palindrome_score: Similarity to the reverse complement of the sequence where reverse complement of a sequence is the sequence obtained by replacing each base with its complement (A↔T, C↔G) and reversing the order.\n"
                "(3) entropy_score: Diversity of short subsequences (2-mers).\n"
                "Examples of sequence with palindrome_score = 1.0: AAAAAAATTTTTTT, GGGGGGGCCCCCCC, AGAGAGTACTCTCT \n"
                f"For your input sequence, we will evolve it for {n_steps} steps and report:\n"
                "- The sequence at each step\n"
                "- The values of your selected properties at each step\n\n"
                f"Your ultimate task is to **form a hypothesis about the underlying dynamics given the following HINT:**\n"
                f"The system roughly tends toward two different convergence points. Your goal is to strategically design queries that help you effectively span the space, so that you can locate the location of these rought convergence points and the rules governing the evolution.\n\n"
                "DON'T FORGET to query sequences with extreme values of the properties you are interested in.\n"
                f"---\n"
                "\n\nFormat your proposals like:\n"
                "SEQUENCE: ACGTACGTAC\n"
                "PROPERTIES: ['gc_content', 'alternation_score']\n\n"
                "Respond carefully!"
            )
        }
    ]
    print("\n=== Prompt Given to LLM ===\n", conversation[0]['content'])

    for round_idx in range(n_rounds):
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

        # Parse LLM message
        sequence = None
        selected_props = []
        for line in llm_msg.splitlines():
            if line.strip().lower().startswith("sequence:"):
                sequence = line.split(":", 1)[1].strip().upper()
            elif line.strip().lower().startswith("properties:"):
                props_line = line.split(":", 1)[1].strip()
                try:
                    selected_props = eval(props_line)
                except Exception:
                    selected_props = []

        valid_seq = sequence and set(sequence) <= set(ALPHABET) and len(sequence) == SEQ_LENGTH
        valid_props = [p for p in selected_props if p in query_properties]

        if not valid_seq:
            print("Invalid sequence proposed. Using a random sequence instead.")
            sequence = Helpers.random_sequence()
        if not valid_props:
            print("No valid properties selected. Defaulting to all available properties.")
            valid_props = list(query_properties.keys())

        query_funcs = {p: query_properties[p] for p in valid_props}
        evolved_seqs, evolved_props = evolve_sequence_with_properties(
            dynamical_system=dynamical_system,
            sequence=sequence,
            properties=list(query_funcs.values()),
            steps=n_steps
        )
        results = "RESULTS:\n"
        for idx, (s, props) in enumerate(zip(evolved_seqs, evolved_props)):
            prop_dict = {name: round(val, 3) for name, val in zip(valid_props, props)}
            results += f"Step {idx}: {s}, Properties: {prop_dict}\n"
        print(results)
        conversation.append({"role": "user", "content": results})

    final_prompt = {"role": "user", "content": "Now, based on your observations, please state your final hypothesis about the dynamics very concisely."}
    final_resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=conversation + [final_prompt],
        **({"temperature": TEMPERATURE} if TEMPERATURE is not None else {})
    )
    final_hypothesis = final_resp.choices[0].message.content
    print("\n=== Final Hypothesis ===\n", final_hypothesis)
    return final_hypothesis



# -----------------------------------------------------------------------------
#  Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Evolution properties (used by the dynamical system) ---
    evolution_properties = {
        "gc_content": O1_PROPERTIES["gc"],
        # "local_propensity": O1_PROPERTIES["local_propensity"]
        # "masked_alter": O1_PROPERTIES["masked_alt"]
        "palindrome_score": O1_PROPERTIES["pal"],
        # "alternation_score": O1_PROPERTIES["alt"]
        # "interaction_score": O1_PROPERTIES["interaction_energy"],
        # "entropy_score": O1_PROPERTIES["entropy"]
    }

    # --- Query properties (LLM can query these during evolution) ---
    query_properties = {
        "gc_content": O1_PROPERTIES["gc"],
        # "local_propensity": O1_PROPERTIES["local_propensity"],
        "palindrome_score": O1_PROPERTIES["pal"],
        # "masked_alter": O1_PROPERTIES["masked_alt"],
        # "alternation_score": O1_PROPERTIES["alt"],
        # "motif_score": O1_PROPERTIES["motif"],
        "entropy_score": O1_PROPERTIES["entropy"]
        # "interaction_score": O1_PROPERTIES["interaction_energy"]
    }

    # --- Fixed points (correspond to actual sequences) ---
    fixed_sequences = [
        "ATAGAGAAGA",
        "CTCGTGCTGG"
    ]
    print(f"MODEL_NAME= {MODEL_NAME}, TEMPERATURE={TEMPERATURE}\n\n")

    fixed_points = [(0.8, 0.8), (0.4, 0.2)]
    print("Fixed Points in Property Space:")
    for seq, fp in zip(fixed_sequences, fixed_points):
        print(f"Sequence: {seq}, Fixed Point: {fp}")

    ds_framework = DynamicalSystemFramework(
        evolution_properties=evolution_properties,
        fixed_points=fixed_points,
        basins='gaussian',  # or 'inverse_square', 'anisotropic'
    )

    # Run the LLM interaction
    final_hypothesis = run_dynamics_discovery_chat(
        dynamical_system=ds_framework,
        query_properties=query_properties,
        n_rounds=N_ROUNDS,
        n_steps=15)
    