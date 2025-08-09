import subprocess
from typing import Tuple


def _run_code_in_container(code: str, container_name: str = "sandbox") -> Tuple[str, str]:
        """
        Helper function that actually runs Python code inside a Docker container named `sandbox` (by default).
        """
        print("executing the code!!!!")

        code = code.strip('"""')
        cmd = [
            "docker", "exec", "-i",
            container_name,
            "python", "-c", "import sys; exec(sys.stdin.read())"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = process.communicate(code)
        return out, err