from __future__ import annotations

import argparse

from citrinet_vast import OfferSearchSpec, VastClient

from ._common import dump_json, tupled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search Vast offers for P003 Citrinet using the typed SDK wrapper."
    )
    parser.add_argument("--gpu-name", type=str, default=None, help="GPU model name.")
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs required."
    )
    parser.add_argument(
        "--storage-gb",
        type=float,
        default=150.0,
        help="Allocated storage requested during search.",
    )
    parser.add_argument(
        "--limit", type=int, default=20, help="Maximum offers to return."
    )
    parser.add_argument(
        "--order",
        type=str,
        default="score-",
        help="Vast order expression, e.g. score- or dph.",
    )
    parser.add_argument(
        "--offer-type",
        type=str,
        default="on-demand",
        help="Offer type passed to Vast search_offers().",
    )
    parser.add_argument(
        "--query-clause",
        action="append",
        default=[],
        help="Additional Vast query clause, repeated as needed.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    client = VastClient.from_env()
    offers = client.search_offers(
        OfferSearchSpec(
            gpu_name=args.gpu_name,
            num_gpus=args.num_gpus,
            storage_gb=args.storage_gb,
            limit=args.limit,
            order=args.order,
            offer_type=args.offer_type,
            query_clauses=tupled(args.query_clause),
        )
    )
    dump_json([offer.to_dict() for offer in offers])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
