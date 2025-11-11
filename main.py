import sys
from planner import MAPLLMPipeline
from config import Config


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-dir", required=True)
    parser.add_argument("--domain-file", required=True)
    parser.add_argument("--problem-file", required=True)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = Config()
    pipeline = MAPLLMPipeline(config)
    pipeline.run(
        domain_dir=args.domain_dir,
        domain_file=args.domain_file,
        problem_file=args.problem_file,
        mode=args.mode,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
