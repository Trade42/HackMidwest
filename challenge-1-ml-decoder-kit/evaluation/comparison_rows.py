"""Availability catalog for the eleven agreed comparisons; no placeholder scores."""
import argparse
import json
from pathlib import Path

ROWS = (
    (1, "Official BP-OSD", "bposd", False, False, "implemented"),
    (2, "Official Relay-BP five solutions", "relay-bp", False, False, "implemented"),
    (3, "Relay-BP first solution", "relay-first-solution", False, False, "implemented"),
    (4, "First leg or full Relay-BP", "relay-first-leg-or-full", False, False, "implemented"),
    (5, "Learned router without predecoder", "evaluation.router:make_tiered_decoder", False, True, "requires_frozen_router"),
    (6, "Static-prior Relay-BP under loss", None, False, False, "unavailable_pending_loss_owner"),
    (7, "Herald-aware priors plus Relay-BP", None, False, False, "unavailable_pending_loss_owner"),
    (8, "Herald priors, tracker, and router", None, False, True, "unavailable_pending_loss_and_tracker"),
    (9, "BP-confidence commits then row 5", "evaluation.predecoder:make_composed_decoder", False, True, "requires_frozen_bp_and_router"),
    (10, "Learned predecoder commits then row 5", "evaluation.predecoder:make_composed_decoder", True, True, "requires_frozen_cnn_and_router"),
    (11, "Learned predecoder commits then BP-OSD", "evaluation.predecoder:make_composed_decoder", True, False, "requires_frozen_cnn"),
)


def catalog():
    return [{"row": n, "comparison": label, "decoder_spec": spec, "preprocessor_is_ml": pre_ml,
             "router_is_ml": router_ml, "availability": status, "score": None}
            for n, label, spec, pre_ml, router_ml, status in ROWS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"{output}: catalog already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog(), indent=2))
    print(output)


if __name__ == "__main__":
    main()
