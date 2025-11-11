class Config:
    def __init__(self):
        self.converter = {
            "converter_script": "/path/to/ma-to-pddl.py",
            "python_cmd": "python2"
        }
        self.llm_model = "llama3"
        self.val_bin = "Validate"
        # etc. all options/modes