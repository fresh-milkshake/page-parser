from subprocess import run, CalledProcessError


def main() -> None:
    commands = [
        [
            "uv",
            "run",
            "ruff",
            "check",
            "src",
            "--fix",
            "--unsafe-fixes",
        ],
        [
            "uv",
            "run",
            "ruff",
            "format",
            "src",
        ],
    ]
    try:
        for command in commands:
            run(command, check=True)
    except CalledProcessError as exc:
        print(f"Process failed with exit code {exc.returncode}")
        raise


if __name__ == "__main__":
    main()
