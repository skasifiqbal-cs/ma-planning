from planner import MAPLLMPipeline
from config import Config

def main():
    config = Config()
    pipeline = MAPLLMPipeline(config)
    # You can iterate over domains/problems as needed
    pipeline.run(domain_dir="/path/to/blocksworld", domain_file="domain.pddl", problem_file="probBLOCKS-10-0.pddl", out_dir="/path/to/out", mode="soft-val", max_steps=30)

if __name__ == "__main__":
    main()