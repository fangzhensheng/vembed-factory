NAME = "jsonl"


def add_generate_arguments(parser):
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument(
        "--retrieval_mode", type=str, default="t2i", choices=["t2i", "i2i", "i2t", "t2t"]
    )


def build_data(args):
    return {
        "data_source": args.data_path,
        "image_root": args.image_root,
        "retrieval_mode": args.retrieval_mode,
        "extras": {},
    }
