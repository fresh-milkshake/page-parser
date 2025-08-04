from subprocess import run, CalledProcessError


def main() -> None:
    """
    Run the main.py script with specified arguments using uv,
    redirecting output to the terminal.
    """
    try:
        run(
            [
                "uv",
                "run",
                "python",
                "main.py",
                "data/2507.21509v1.pdf",
                "models/yolov12l-doclaynet.pt",
                "output",
                "output.json",
            ],
            check=True,
        )
    except CalledProcessError as exc:
        print(f"Process failed with exit code {exc.returncode}")
        raise


if __name__ == "__main__":
    main()
