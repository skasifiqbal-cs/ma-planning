class Config:
    """
    Global configuration holder for the ma-planning pipeline.
    Supports extension with command-line arguments or env vars if desired.
    """

    def __init__(self):
        # MAPDDL to PDDL converter settings
        self.converter = {
            # Path to ma-to-pddl.py script (update as needed)
            "converter_script": "/home/rr/ma-planning/codmap-2015/competition/centalized/ma-to-pddl.py",
            # Command used to run Python 2 for the converter
            "python_cmd": "python2",
        }

        # LLM model name for Ollama/OpenAI, etc.
        self.llm_model = "llama3:8b"  # e.g., "llama3", "phi3", "openai/gpt-4", etc.

        # URL for LLM API endpoint (default is Ollama local)
        self.llm_url = "http://localhost:11434/api/chat"

        # VAL validator binary location (if not in PATH, use absolute path)
        self.val_bin = "Validate"  # e.g., "/usr/local/bin/Validate"

        # SentenceTransformer embedding model for soft validation
        self.embed_model = "paraphrase-MiniLM-L6-v2"

        # Default mode for validation ("strict", "no-val", "soft-val")
        self.validation_mode = "no-val"

        # Directory with unfactored MAPDDL domains/problems
        self.unfactored_root = (
            "/home/rr/Downloads/pddl-data-master/codmap-2015/unfactored/"
        )

        # Directory for storing centralized PDDL output files
        self.centralized_root = "/home/rr/ma-planning/centralized"

        # Output directory for experiment results/plans
        self.results_root = "/home/rr/ma-planning/results"

        # Planning loop options
        self.max_steps = 30
        self.plan_timeout = 300  # seconds, if you support timeouts

        # Debug flag for verbose output/logging
        self.debug = True

        # Default temperature value
        self.temperature = 0.1

        # Add more options as needed for batch jobs, logging, etc.

    def as_dict(self):
        """
        Returns all config fields as a dictionary for easy serialization/logging.
        """
        return self.__dict__
