from model_server.kueue_rl import build_parser, train


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
