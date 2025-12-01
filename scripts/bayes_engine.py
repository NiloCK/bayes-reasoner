#!/usr/bin/env python3
"""
Bayesian reasoning engine for hypothesis evaluation.

State is stored in .claude/.bayes_state.json relative to the current working directory.
"""

import argparse
import json
import math
import os
import sys

STATE_DIR = ".claude"
STATE_FILE = os.path.join(STATE_DIR, ".bayes_state.json")


def ensure_state_dir():
    """Create .claude directory if it doesn't exist."""
    if not os.path.exists(STATE_DIR):
        os.makedirs(STATE_DIR)


def load_state():
    if not os.path.exists(STATE_FILE):
        return {"hypotheses": {}, "tests": {}}
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    ensure_state_dir()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def output(data, as_json=False):
    """Output data as JSON or human-readable text."""
    if as_json:
        print(json.dumps(data, indent=2))
    else:
        if "error" in data:
            print(f"Error: {data['error']}")
        if "warning" in data:
            print(f"⚠️  {data['warning']}")
        if "message" in data:
            print(data["message"])
        if "beliefs" in data:
            print("\n--- CURRENT BELIEFS ---")
            for item in data["beliefs"]:
                bar = "#" * int(item["probability"] * 20)
                print(f"{item['hypothesis']:20} | {bar} {item['probability']:.2%}")
        if "recommendations" in data:
            print("\n--- TEST RECOMMENDATIONS (Maximize Information Gain) ---")
            for item in data["recommendations"]:
                print(f"[{item['info_gain']:.4f} bits] {item['test']}")
        if "tests_defined" in data and not as_json:
            print(
                f"\nDefined tests: {', '.join(data['tests_defined']) if data['tests_defined'] else '(none)'}"
            )


def get_beliefs_list(hypotheses):
    """Convert hypotheses dict to sorted list for output."""
    sorted_h = sorted(hypotheses.items(), key=lambda x: x[1], reverse=True)
    return [{"hypothesis": h, "probability": p} for h, p in sorted_h]


# Recognized names for the "other" hypothesis (catch-all for unlisted causes)
OTHER_HYPOTHESIS_NAMES = {"other", "unknown", "surprise", "unlisted", "none_of_above"}


def has_other_hypothesis(hypotheses):
    """Check if any hypothesis represents the 'other' category (unlisted causes)."""
    return any(h.lower() in OTHER_HYPOTHESIS_NAMES for h in hypotheses)


def cmd_init(args):
    """Initialize hypotheses. Args: H1:0.5 H2:0.5"""
    hypotheses = {}
    for item in args.pairs:
        if ":" not in item:
            return {"error": f"Invalid format '{item}'. Use H1:0.5 format."}
        key, val = item.split(":", 1)
        try:
            hypotheses[key] = float(val)
        except ValueError:
            return {"error": f"Invalid probability '{val}' for hypothesis '{key}'."}

    if not hypotheses:
        return {"error": "No hypotheses provided."}

    # Normalize
    total = sum(hypotheses.values())
    if total == 0:
        return {"error": "Total probability cannot be zero."}
    hypotheses = {k: v / total for k, v in hypotheses.items()}

    save_state({"hypotheses": hypotheses, "tests": {}})

    # Check for "Other" hypothesis
    warning = None
    if not has_other_hypothesis(hypotheses):
        warning = (
            "No 'Other' hypothesis included. "
            "Consider adding one (e.g., Other:0.15) to leave room for unlisted causes. "
            "A closed hypothesis space assumes one of your guesses is definitely correct."
        )

    result = {
        "message": f"Initialized model with {len(hypotheses)} hypotheses.",
        "beliefs": get_beliefs_list(hypotheses),
        "tests_defined": [],
    }
    if warning:
        result["warning"] = warning
    return result


def cmd_define_test(args):
    """
    Define a test's likelihoods.

    Args: TestName H1:0.9 H2:0.1

    Likelihoods represent P(test passes | hypothesis is true).
    Unmentioned hypotheses default to 0.5 (neutral evidence).
    """
    state = load_state()

    if not state["hypotheses"]:
        return {"error": "No hypotheses defined. Run 'init' first."}

    likelihoods = {}
    for item in args.pairs:
        if ":" not in item:
            return {"error": f"Invalid format '{item}'. Use H1:0.9 format."}
        key, val = item.split(":", 1)
        try:
            prob = float(val)
            if not 0 <= prob <= 1:
                return {
                    "error": f"Likelihood must be between 0 and 1, got {prob} for '{key}'."
                }
            likelihoods[key] = prob
        except ValueError:
            return {"error": f"Invalid likelihood '{val}' for hypothesis '{key}'."}

    # Warn about unknown hypotheses
    unknown = set(likelihoods.keys()) - set(state["hypotheses"].keys())
    if unknown:
        return {
            "error": f"Unknown hypotheses: {unknown}. Known: {list(state['hypotheses'].keys())}"
        }

    # Default unmentioned hypotheses to 0.5 (neutral)
    for h in state["hypotheses"]:
        if h not in likelihoods:
            likelihoods[h] = 0.5

    state["tests"][args.name] = likelihoods
    save_state(state)

    return {
        "message": f"Defined test '{args.name}'. Likelihoods: {likelihoods}",
        "tests_defined": list(state["tests"].keys()),
    }


def cmd_undefine_test(args):
    """Remove a test definition."""
    state = load_state()

    if args.name not in state["tests"]:
        return {
            "error": f"Test '{args.name}' not defined. Defined tests: {list(state['tests'].keys())}"
        }

    del state["tests"][args.name]
    save_state(state)

    return {
        "message": f"Removed test '{args.name}'.",
        "tests_defined": list(state["tests"].keys()),
    }


def cmd_split(args):
    """
    Split a named hypothesis into sub-hypotheses, redistributing its probability mass.

    Example: split Network NetworkTimeout:0.6 NetworkDNS:0.4

    This takes the source hypothesis's probability and distributes it among the
    new hypotheses according to the given ratios. The source is removed and
    replaced by the new set.

    Use this when refining a named, concrete hypothesis into sub-categories.
    Do NOT use this on "Other"—use `inject` instead for new candidate causes.
    """
    state = load_state()
    hypotheses = state["hypotheses"]

    if not hypotheses:
        return {"error": "No hypotheses defined. Run 'init' first."}

    source = args.source
    if source not in hypotheses:
        return {
            "error": f"Hypothesis '{source}' not found. Current hypotheses: {list(hypotheses.keys())}"
        }

    # Guard: do not split "Other" or equivalent
    if source.lower() in OTHER_HYPOTHESIS_NAMES:
        return {
            "error": f"Cannot split '{source}'. 'Other' represents model incompleteness, not a refinable hypothesis. "
            f"Use 'inject NewHypothesis:probability' to add new candidate causes at evidence-appropriate probabilities."
        }

    source_prob = hypotheses[source]

    # Parse the new distribution
    new_distribution = {}
    for item in args.pairs:
        if ":" not in item:
            return {
                "error": f"Invalid format '{item}'. Use NewHypothesis:ratio format."
            }
        key, val = item.split(":", 1)
        try:
            new_distribution[key] = float(val)
        except ValueError:
            return {"error": f"Invalid ratio '{val}' for hypothesis '{key}'."}

    if not new_distribution:
        return {"error": "No new hypotheses provided."}

    # Check for conflicts with existing hypotheses (except source)
    existing_conflicts = set(new_distribution.keys()) & (
        set(hypotheses.keys()) - {source}
    )
    if existing_conflicts:
        return {
            "error": f"Cannot split into existing hypotheses: {existing_conflicts}. Use different names or include them in the split."
        }

    # Normalize the ratios
    total_ratio = sum(new_distribution.values())
    if total_ratio == 0:
        return {"error": "Total ratio cannot be zero."}

    # Remove the source hypothesis
    del hypotheses[source]

    # Add new hypotheses with probability proportional to source's probability
    for h, ratio in new_distribution.items():
        allocated_prob = source_prob * (ratio / total_ratio)
        if h in hypotheses:
            hypotheses[h] += allocated_prob
        else:
            hypotheses[h] = allocated_prob

    # Update test likelihoods: new hypotheses default to 0.5, source is removed
    for test_name, likelihoods in state["tests"].items():
        if source in likelihoods:
            del likelihoods[source]
        for h in new_distribution:
            if h not in likelihoods:
                likelihoods[h] = 0.5

    state["hypotheses"] = hypotheses
    save_state(state)

    # Check if we still have an "Other" hypothesis
    warning = None
    if not has_other_hypothesis(hypotheses):
        warning = (
            "No 'Other' hypothesis remains after split. "
            "Consider including Other:0.1 (or similar) to preserve room for unlisted causes."
        )

    result = {
        "message": f"Split '{source}' ({source_prob:.1%}) into {len(new_distribution)} hypotheses.",
        "beliefs": get_beliefs_list(hypotheses),
        "tests_defined": list(state["tests"].keys()),
        "note": "Existing tests now use likelihood=0.5 (neutral) for new hypotheses. Consider redefining tests with specific likelihoods.",
    }
    if warning:
        result["warning"] = warning
    return result


def cmd_inject(args):
    """
    Inject a new hypothesis at a specified probability, shrinking all others proportionally.

    Example: inject BuildCache:0.95
    Example: inject DNS:0.25 --likelihoods CheckLogs:0.7 PingTest:0.3

    Use this when a smoking gun emerges—new evidence that demands a new hypothesis
    at high probability. All existing hypotheses shrink proportionally to make room,
    preserving their relative ordering.

    If tests are already defined, you MUST provide likelihoods for the new hypothesis
    via --likelihoods, to ensure future test recommendations remain valid.
    """
    state = load_state()
    hypotheses = state["hypotheses"]

    if not hypotheses:
        return {"error": "No hypotheses defined. Run 'init' first."}

    # Parse the injection
    injections = {}
    for item in args.pairs:
        if ":" not in item:
            return {
                "error": f"Invalid format '{item}'. Use Hypothesis:probability format."
            }
        key, val = item.split(":", 1)
        try:
            prob = float(val)
            if not 0 < prob < 1:
                return {
                    "error": f"Probability must be between 0 and 1 (exclusive), got {prob}."
                }
            injections[key] = prob
        except ValueError:
            return {"error": f"Invalid probability '{val}' for hypothesis '{key}'."}

    if not injections:
        return {"error": "No hypotheses to inject."}

    total_injection = sum(injections.values())
    if total_injection >= 1:
        return {
            "error": f"Total injected probability ({total_injection}) must be less than 1."
        }

    # Check for conflicts with existing hypotheses
    conflicts = set(injections.keys()) & set(hypotheses.keys())
    if conflicts:
        return {
            "error": f"Cannot inject existing hypotheses: {conflicts}. Use 'update' to change beliefs about existing hypotheses, or 'reset' and 'init' to start fresh."
        }

    # Parse likelihoods for existing tests
    provided_likelihoods = {}
    if args.likelihoods:
        for item in args.likelihoods:
            if ":" not in item:
                return {
                    "error": f"Invalid likelihood format '{item}'. Use TestName:likelihood format."
                }
            test_name, val = item.split(":", 1)
            try:
                likelihood = float(val)
                if not 0 <= likelihood <= 1:
                    return {
                        "error": f"Likelihood must be between 0 and 1, got {likelihood} for '{test_name}'."
                    }
                provided_likelihoods[test_name] = likelihood
            except ValueError:
                return {"error": f"Invalid likelihood '{val}' for test '{test_name}'."}

    # Check that likelihoods are provided for all existing tests
    existing_tests = set(state["tests"].keys())
    if existing_tests:
        provided_tests = set(provided_likelihoods.keys())
        missing_tests = existing_tests - provided_tests
        unknown_tests = provided_tests - existing_tests

        if unknown_tests:
            return {
                "error": f"Unknown tests in --likelihoods: {unknown_tests}. Defined tests: {list(existing_tests)}"
            }

        if missing_tests:
            return {
                "error": f"Must provide likelihoods for all defined tests. Missing: {list(missing_tests)}. "
                f"Use --likelihoods {' '.join(f'{t}:0.5' for t in missing_tests)} (adjust values as appropriate)."
            }

    # Shrink existing hypotheses proportionally
    shrink_factor = 1 - total_injection
    for h in hypotheses:
        hypotheses[h] *= shrink_factor

    # Add injected hypotheses
    for h, prob in injections.items():
        hypotheses[h] = prob

    # Update test likelihoods for new hypotheses
    for test_name, likelihoods in state["tests"].items():
        for h in injections:
            # Use provided likelihood, or this shouldn't happen (we checked above)
            likelihoods[h] = provided_likelihoods.get(test_name, 0.5)

    state["hypotheses"] = hypotheses
    save_state(state)

    # Check if we still have an "Other" hypothesis
    warning = None
    if not has_other_hypothesis(hypotheses):
        warning = (
            "No 'Other' hypothesis present. "
            "Consider adding one to leave room for unlisted causes."
        )

    result = {
        "message": f"Injected {len(injections)} hypothesis(es). Existing hypotheses shrunk by {shrink_factor:.1%}.",
        "beliefs": get_beliefs_list(hypotheses),
        "tests_defined": list(state["tests"].keys()),
    }
    if warning:
        result["warning"] = warning
    return result


def cmd_update(args):
    """
    Update beliefs based on test result.

    Args: TestName result

    Result can be: true/pass/yes/1 for positive, false/fail/no/0 for negative.
    The test definition is preserved for potential reuse.
    """
    state = load_state()
    hypotheses = state["hypotheses"]

    if not hypotheses:
        return {"error": "No hypotheses defined. Run 'init' first."}

    test_likelihoods = state["tests"].get(args.name)

    if not test_likelihoods:
        return {
            "error": f"Test '{args.name}' not defined. Defined tests: {list(state['tests'].keys())}"
        }

    is_positive = args.result.lower() in ["true", "pass", "yes", "1"]
    is_negative = args.result.lower() in ["false", "fail", "no", "0"]

    if not is_positive and not is_negative:
        return {
            "error": f"Invalid result '{args.result}'. Use true/false, pass/fail, yes/no, or 1/0."
        }

    new_priors = {}
    total_prob = 0

    for h, prior in hypotheses.items():
        likelihood = test_likelihoods.get(h, 0.5)
        if is_negative:
            likelihood = 1.0 - likelihood

        posterior = likelihood * prior
        new_priors[h] = posterior
        total_prob += posterior

    if total_prob == 0:
        return {
            "error": "Evidence impossible under all hypotheses. Your likelihood definitions may need revision.",
            "hint": "This means P(evidence|H)=0 for all H. Re-examine your assumptions.",
        }

    # Normalize
    state["hypotheses"] = {h: p / total_prob for h, p in new_priors.items()}

    # Test definition is preserved (not deleted)
    save_state(state)

    result_str = "PASS" if is_positive else "FAIL"
    return {
        "message": f"Updated beliefs based on {args.name}={result_str}",
        "beliefs": get_beliefs_list(state["hypotheses"]),
        "tests_defined": list(state["tests"].keys()),
    }


def get_entropy(beliefs):
    """Calculate Shannon entropy in bits."""
    return -sum(p * math.log2(p) for p in beliefs.values() if p > 0)


def cmd_recommend(args):
    """Calculate Expected Information Gain for all defined tests."""
    state = load_state()
    hypotheses = state["hypotheses"]

    if not hypotheses:
        return {"error": "No hypotheses defined. Run 'init' first."}

    if not state["tests"]:
        return {"error": "No tests defined. Use 'define' to add tests."}

    current_entropy = get_entropy(hypotheses)

    results = []

    for test_name, likelihoods in state["tests"].items():
        # P(Test passes)
        p_pass = sum(hypotheses[h] * likelihoods.get(h, 0.5) for h in hypotheses)
        p_fail = 1.0 - p_pass

        # Entropy if Pass
        if p_pass > 0:
            post_pass = {
                h: (likelihoods.get(h, 0.5) * hypotheses[h]) / p_pass
                for h in hypotheses
            }
            ent_pass = get_entropy(post_pass)
        else:
            ent_pass = 0

        # Entropy if Fail
        if p_fail > 0:
            post_fail = {
                h: ((1.0 - likelihoods.get(h, 0.5)) * hypotheses[h]) / p_fail
                for h in hypotheses
            }
            ent_fail = get_entropy(post_fail)
        else:
            ent_fail = 0

        expected_entropy = (p_pass * ent_pass) + (p_fail * ent_fail)
        info_gain = current_entropy - expected_entropy
        results.append({"test": test_name, "info_gain": info_gain, "p_pass": p_pass})

    results.sort(key=lambda x: x["info_gain"], reverse=True)

    return {"current_entropy": current_entropy, "recommendations": results}


def cmd_status(args):
    """Show current beliefs and defined tests."""
    state = load_state()

    if not state["hypotheses"]:
        return {"message": "No model initialized. Run 'init' to begin."}

    return {
        "beliefs": get_beliefs_list(state["hypotheses"]),
        "tests_defined": list(state["tests"].keys()),
        "entropy": get_entropy(state["hypotheses"]),
    }


def cmd_reset(args):
    """Clear all state and start fresh."""
    save_state({"hypotheses": {}, "tests": {}})
    return {"message": "State cleared. Ready for new investigation."}


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian reasoning engine for hypothesis evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize with hypotheses (always include Other!)
  %(prog)s init Database:0.35 Network:0.30 CodeBug:0.20 Other:0.15

  # Define a test: P(logs show timeout | Network issue) = 0.9
  %(prog)s define CheckLogs Network:0.9 Database:0.3 CodeBug:0.1 Other:0.5

  # Get recommendation for most informative test
  %(prog)s recommend

  # Update beliefs after test result
  %(prog)s update CheckLogs pass

  # Split a named hypothesis into sub-categories
  %(prog)s split Network NetworkTimeout:0.6 NetworkDNS:0.4

  # Add a new hypothesis (with likelihoods for existing tests)
  %(prog)s inject DNS:0.25 --likelihoods CheckLogs:0.7 PingTest:0.3

  # Inject a smoking-gun hypothesis (shrinks all others proportionally)
  %(prog)s inject BuildCache:0.90 --likelihoods CheckLogs:0.1 PingTest:0.1

  # View current state
  %(prog)s status

  # Output as JSON for programmatic use
  %(prog)s --json status
""",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    subparsers = parser.add_subparsers(dest="command")

    p_init = subparsers.add_parser(
        "init", help="Initialize hypotheses (H1:prob H2:prob ...)"
    )
    p_init.add_argument(
        "pairs",
        nargs="+",
        metavar="H:prob",
        help="Hypothesis:probability pairs (will be normalized)",
    )

    p_def = subparsers.add_parser("define", help="Define a test's likelihoods")
    p_def.add_argument("name", help="Test name")
    p_def.add_argument(
        "pairs",
        nargs="+",
        metavar="H:likelihood",
        help="P(test passes | hypothesis) for each hypothesis",
    )

    p_undef = subparsers.add_parser("undefine", help="Remove a test definition")
    p_undef.add_argument("name", help="Test name to remove")

    p_split = subparsers.add_parser(
        "split", help="Split a hypothesis into new hypotheses"
    )
    p_split.add_argument("source", help="Hypothesis to split (e.g., Other)")
    p_split.add_argument(
        "pairs",
        nargs="+",
        metavar="H:ratio",
        help="NewHypothesis:ratio pairs (ratios are normalized)",
    )

    p_inject = subparsers.add_parser(
        "inject", help="Inject new hypothesis at given probability, shrinking others"
    )
    p_inject.add_argument(
        "pairs",
        nargs="+",
        metavar="H:prob",
        help="Hypothesis:probability pairs to inject (others shrink proportionally)",
    )
    p_inject.add_argument(
        "--likelihoods",
        nargs="+",
        metavar="Test:likelihood",
        help="Likelihoods for new hypothesis on existing tests (required if tests are defined)",
    )

    p_upd = subparsers.add_parser("update", help="Update beliefs with test result")
    p_upd.add_argument("name", help="Test name")
    p_upd.add_argument("result", help="Test result: true/false, pass/fail, yes/no, 1/0")

    p_rec = subparsers.add_parser(
        "recommend", help="Recommend highest information-gain test"
    )

    p_stat = subparsers.add_parser("status", help="Show current beliefs and tests")

    p_reset = subparsers.add_parser("reset", help="Clear all state")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "define": cmd_define_test,
        "undefine": cmd_undefine_test,
        "split": cmd_split,
        "inject": cmd_inject,
        "update": cmd_update,
        "recommend": cmd_recommend,
        "status": cmd_status,
        "reset": cmd_reset,
    }

    if args.command in commands:
        result = commands[args.command](args)
        output(result, as_json=args.json)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
