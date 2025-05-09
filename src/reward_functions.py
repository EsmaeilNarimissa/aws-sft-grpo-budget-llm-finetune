"""
Reward functions for the DeepSeek R1 GRPO training pipeline.
"""

import re
from typing import Dict, Optional, List, Tuple
import math
from collections import Counter
import logging

# Imports for mathematical accuracy checking
from latex2sympy2_extended.latex2sympy2 import latex2sympy
from latex2sympy2_extended.math_normalization import normalize_latex

from math_verify import verify

# Setup logging
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """Extracts content within the first occurrence of <tag>...</tag>."""
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1).strip() if match else None

def _generate_ngrams(text: str, ngram_size: int) -> List[Tuple[str, ...]]:
    """Generates n-grams from a string after cleaning and splitting.

    Args:
        text: The input string.
        ngram_size: The size of the n-grams (e.g., 3 for trigrams).

    Returns:
        A list of n-grams tuples.
    """
    if not text:
        return []
    # Basic cleaning: lowercase and split by space
    words = text.lower().split()
    if len(words) < ngram_size:
        return []
    # Efficient n-gram generation using zip
    return list(zip(*[words[i:] for i in range(ngram_size)]))

# --- Reward Functions ---

def accuracy_reward(output: str, label: str) -> float:
    """
    Calculates reward based on mathematical equivalence between the content 
    of the <answer> tag and the ground-truth label.
    Uses latex2sympy2_extended for parsing and normalization, and math_verify for comparison.
    Returns 1.0 if equivalent, 0.0 otherwise.
    """
    answer_content = extract_tag_content(output, 'answer')
    
    if answer_content is None:
        logger.debug("AccuracyReward: No <answer> tag found in output.")
        return 0.0
    if not label:
        logger.debug("AccuracyReward: Ground truth label is empty.")
        return 0.0 # Cannot compare if label is empty

    try:
        # Attempt to parse both the extracted answer and the label
        # latex2sympy can handle some basic math expressions directly too
        normalized_answer = normalize_latex(answer_content)
        normalized_label = normalize_latex(label)

        parsed_answer = latex2sympy(normalized_answer)
        parsed_label = latex2sympy(normalized_label)

        # Use math_verify to compare the parsed expressions
        is_equivalent = verify(parsed_answer, parsed_label)
        
        if is_equivalent:
            logger.debug(f"AccuracyReward: Match found. Answer: {parsed_answer}, Label: {parsed_label}")
            return 1.0
        else:
            logger.debug(f"AccuracyReward: No match. Answer: {parsed_answer}, Label: {parsed_label}")
            return 0.0

    except (SyntaxError, TypeError, Exception) as e:
        # Catch potential parsing errors from latex2sympy or comparison errors
        logger.warning(f"AccuracyReward: Error during comparison. Answer: '{answer_content}', Label: '{label}'. Error: {e}")
        # Fallback to strict string comparison if parsing/comparison fails?
        # Or return 0.0 as the reference likely expects successful parsing.
        # Let's return 0.0 on error for now, assuming robust parsing is required.
        return 0.0 

def format_reward(output: str) -> float:
    """
    Checks if the output contains both <think>...</think> and <answer>...</answer> tags,
    in the correct order.
    Returns 1.0 if format is correct, 0.0 otherwise.
    """
    think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)

    # Check for presence and basic content
    if think_match and answer_match and think_match.group(1).strip() and answer_match.group(1).strip():
        # Check if <think> appears before <answer>
        if think_match.start() < answer_match.start():
            return 1.0
    return 0.0

def reasoning_steps_reward(output: str, min_steps: int = 2) -> float:
    """
    Calculates reward based on the presence and number of reasoning steps
    within the <think> tag.
    A simple heuristic: counts non-empty lines within the tag.
    Returns 1.0 if step count >= min_steps, scaled down otherwise.
    Needs refinement for actual step quality analysis.
    """
    think_content = extract_tag_content(output, 'think')
    if not think_content:
        return 0.0

    # Simple step counting: split by newline, filter empty lines
    steps = [line for line in think_content.strip().split('\n') if line.strip()]
    num_steps = len(steps)

    # Reward scaling (example: linear scaling up to min_steps)
    if num_steps >= min_steps:
        return 1.0
    elif min_steps > 0:
        return float(num_steps) / min_steps
    else:
        return 0.0 # Avoid division by zero if min_steps is 0 or negative

def conciseness_reward(output: str, min_len: int = 20, max_len: int = 500) -> float:
    """
    Calculates reward based on the total length of the output.
    Returns 1.0 if within [min_len, max_len], 0.0 otherwise.
    Penalizes outputs that are too short or too long.
    """
    output_len = len(output)
    if min_len <= output_len <= max_len:
        return 1.0
    return 0.0

def combine_rewards(rewards: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """
    Combines individual reward scores into a single weighted score.
    If weights are not provided, assumes equal weight for all rewards.
    """
    if not rewards:
        return 0.0

    if weights is None:
        # Default to equal weights if none provided
        num_rewards = len(rewards)
        weights = {key: 1.0 / num_rewards for key in rewards.keys()} if num_rewards > 0 else {}
        
    total_reward = 0.0
    total_weight = 0.0

    # Calculate weighted sum, ensuring weights exist for provided rewards
    for key, score in rewards.items():
        weight = weights.get(key, 0.0) # Default to 0 weight if key missing in weights dict
        if weight < 0:
             raise ValueError(f"Reward weight for '{key}' cannot be negative: {weight}")
        total_reward += score * weight
        total_weight += weight

    # Normalize by total weight to handle cases where weights don't sum to 1
    # or some reward keys were missing from the weights dict.
    if total_weight > 0:
        # Ensure reward is not arbitrarily inflated if total_weight < 1
        # Clamp the effective total weight used for normalization to max 1.0
        # Although typically weights should sum to 1 or be handled appropriately
        # normalization_factor = max(total_weight, 1.0)
        # return total_reward / normalization_factor 
        # Sticking to direct normalization by sum of weights used:
        return total_reward / total_weight
    else:
        # Avoid division by zero if no valid weights found or rewards dict was empty
        # Or if all weights provided were 0.
        # Fallback: return average of scores if weights are problematic
        if len(rewards) > 0:
            # Ensure scores are valid numbers before averaging
            valid_scores = [s for s in rewards.values() if isinstance(s, (int, float))]
            return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        else:
            return 0.0

# --- New Reward Functions based on Reference ---

def cosine_scaled_reward(accuracy_score: float, output_text: str, max_len: int = 512) -> float:
    """
    Scales the accuracy reward based on the length of the output text using a cosine function.
    Shorter correct solutions get higher rewards (closer to accuracy_score * 1.0).
    Longer correct solutions get slightly lower rewards (closer to accuracy_score * 0.5).
    Incorrect solutions (accuracy_score=0) remain 0.
    Based on the description in Fareedkhan-dev-train-deepseek-r1.txt.

    Args:
        accuracy_score: The raw accuracy reward (0.0 or 1.0).
        output_text: The generated text (used for length calculation).
        max_len: The reference maximum length for scaling.

    Returns:
        The scaled accuracy reward.
    """
    if accuracy_score == 0.0:
        return 0.0

    output_len = len(output_text)
    if output_len == 0 or max_len <= 0:
        return accuracy_score # Return raw score if length is zero or max_len invalid

    # Scale factor based on cosine
    # cos term ranges from 1 (len=0) to -1 (len=max_len) -> (1+cos)/2 ranges from 1 to 0
    # We want scaling from 1 (short) down towards 0.5 (long), so adjust the formula slightly
    # Let's use the formula hinted in the reference's diagram context (Reward * (1+cos)/2)
    scale_factor = (1 + math.cos(math.pi * min(1.0, output_len / max_len))) / 2.0
    
    # Apply scaling to the accuracy score
    return accuracy_score * scale_factor

def repetition_penalty_reward(output_text: str, ngram_size: int = 4, penalty_strength: float = 1.0) -> float:
    """
    Calculates a reward that penalizes repetition of n-grams in the output text.
    Reward is 1.0 for no repetition, decreasing towards 0 as repetition increases.
    Based on the description in Fareedkhan-dev-train-deepseek-r1.txt.

    Args:
        output_text: The generated text.
        ngram_size: The size of n-grams to check for repetition (e.g., 3 or 4).
        penalty_strength: A factor to scale the penalty (0 to 1). 1.0 means full penalty.

    Returns:
        A reward score between 0.0 and 1.0.
    """
    if not output_text or len(output_text.split()) < ngram_size:
        return 1.0 # No repetition possible

    ngrams = _generate_ngrams(output_text, ngram_size)
    if not ngrams:
        return 1.0 # Not enough words to form n-grams

    total_ngrams = len(ngrams)
    unique_ngrams_count = len(set(ngrams))

    # Calculate repetition ratio (0 means no repetition, approaches 1 for high repetition)
    repetition_ratio = 1.0 - (unique_ngrams_count / total_ngrams)

    # Calculate penalty (0 to 1)
    penalty = repetition_ratio * penalty_strength
    
    # Reward is 1 minus penalty
    reward = 1.0 - penalty

    # Ensure reward is within [0, 1]
    return max(0.0, min(1.0, reward))


# --- Example Usage and Basic Tests ---
if __name__ == '__main__':
    sample_good_output = "<think>Step 1: Understand the question. Step 2: Formulate answer.</think><answer>This is the answer.</answer>"
    sample_bad_format = "<answer>Just the answer.</answer><think>Thinking later?</think>"
    sample_no_tags = "Plain text without tags."
    sample_only_think = "<think>Thinking...</think>"
    sample_only_answer = "<answer>Answer.</answer>"
    sample_long_think = "<think>" + "a"*100 + "</think><answer>Ans.</answer>"
    sample_short_output = "<think>a</think><answer>b</answer>"

    label = "This is the answer."

    print(f"--- Testing Output: '{sample_good_output[:50]}...' ---")
    r_acc = accuracy_reward(sample_good_output, label)
    r_fmt = format_reward(sample_good_output)
    r_rsn = reasoning_steps_reward(sample_good_output)
    r_cnc = conciseness_reward(sample_good_output)
    rewards_good = {'accuracy': r_acc, 'format': r_fmt, 'reasoning': r_rsn, 'conciseness': r_cnc}
    print(f"Individual: {rewards_good}")
    print(f"Combined (equal weights): {combine_rewards(rewards_good)}")
    custom_weights = {'accuracy': 0.5, 'format': 0.2, 'reasoning': 0.2, 'conciseness': 0.1}
    print(f"Combined (custom weights): {combine_rewards(rewards_good, custom_weights)}\n")

    print(f"--- Testing Output: '{sample_bad_format[:50]}...' ---")
    r_acc_bf = accuracy_reward(sample_bad_format, "Different Label") # Test wrong label
    r_fmt_bf = format_reward(sample_bad_format)
    rewards_bad_format = {'accuracy': r_acc_bf, 'format': r_fmt_bf}
    print(f"Individual: {rewards_bad_format}")
    print(f"Combined (equal): {combine_rewards(rewards_bad_format)}\n")

    print(f"--- Testing Output: '{sample_no_tags[:50]}...' ---")
    r_fmt_nt = format_reward(sample_no_tags)
    print(f"Format Reward: {r_fmt_nt}\n") # Expect 0
    
    print(f"--- Testing Output: '{sample_short_output[:50]}...' ---")
    r_cnc_short = conciseness_reward(sample_short_output)
    print(f"Conciseness Reward: {r_cnc_short}\n") # Expect 0

    print(f"--- Testing Combine with Missing Weights ---")
    rewards_partial = {'accuracy': 1.0, 'format': 1.0}
    weights_partial = {'accuracy': 0.8} # Missing format weight
    print(f"Rewards: {rewards_partial}, Weights: {weights_partial}")
    # Should normalize based on provided weights (total_weight=0.8)
    # Expected: (1.0 * 0.8 + 1.0 * 0.0) / 0.8 = 1.0
    print(f"Combined: {combine_rewards(rewards_partial, weights_partial)}") 

    print(f"--- Testing Combine with Zero Weights ---")
    rewards_zero_w = {'accuracy': 1.0, 'format': 0.5}
    weights_zero_w = {'accuracy': 0.0, 'format': 0.0} 
    print(f"Rewards: {rewards_zero_w}, Weights: {weights_zero_w}")
    # Should fallback to average score
    # Expected: (1.0 + 0.5) / 2 = 0.75
    print(f"Combined: {combine_rewards(rewards_zero_w, weights_zero_w)}")

    # Test new reward functions
    print(f"--- Testing Cosine Scaled Reward ---")
    r_acc_scaled = cosine_scaled_reward(r_acc, sample_good_output)
    print(f"Cosine Scaled Accuracy Reward: {r_acc_scaled}")

    print(f"--- Testing Repetition Penalty Reward ---")
    r_rep_penalty = repetition_penalty_reward(sample_good_output)
    print(f"Repetition Penalty Reward: {r_rep_penalty}")
