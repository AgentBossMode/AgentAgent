# Promptius AI

Repository of experiments:

1. This repo is set up to use uv. Run `uv sync` to install required dependencies:
    ```bash
    uv sync
    ```
    Or, if you prefer not to use uv:
    ```bash
    pip install -r requirements.txt
    ```

2. To get the environment variables file, do the following:
    - Create a gpg key - https://gnupg.org/download/
    ```bash
    brew install gpg
    gpg --full-generate-key
    ```
    Choose:
    - RSA and RSA, size 4096
    - Expiration: optional
    - Name/email: your actual info
    - Passphrase: something safe

    Share the pubkey with admin (Kanishk/Abhishek).
    Once they encrypt the file with your gpg key, then using 'sops' - http://getsops.io/:
    ```bash
    brew install sops # for mac
    export GPG_TTY=$(tty)
    sops --decrypt --input-type binary --output-type binary .env.enc > .env.decrypted
    ```

3. To run langgraph server, go to the root folder and run the following:
    ```bash
    uv run langgraph dev
    ```

4. To run unit tests:
    ```bash
    pytest -s tests
    ```